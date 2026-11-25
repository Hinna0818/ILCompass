"""Frozen tissue encoder, full-gene residual head; C train / D development evaluation."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='4'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ilc2_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/ilc2_numba')
import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from models import ROOT
from prepare import write_json
from posttrain_data import encode,file_hash
from genecompass_coverage import map_genes
from qc_utils import _raw_counts
from residual import ResidualPredictor
from il33_lung_metrics import correlation,differential,evaluate,r2

CFG=json.loads((ROOT/'configs/drugpred_il33_lung_tissue.json').read_text())
CACHE=ROOT/CFG['cache_directory']
OUT=ROOT/CFG['result_directory']


def prepare_data():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    prep=json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    paths={role:ROOT/CFG['data_directory']/role/f'GSE136154_{role}.h5ad' for role in ['train','validation']}
    signature=dict(inputs={r:file_hash(p) for r,p in paths.items()},
                   checkpoint=file_hash(ROOT/CFG['checkpoint']/'pytorch_model.bin'),
                   tokens=file_hash(ROOT/prep['token_dictionary']),medians=file_hash(ROOT/prep['gene_median_reference']),
                   encoding=prep['encoding'],max_gene_tokens=prep['max_gene_tokens'],precision='FP32',pooling='masked_gene_mean_no_species_CLS')
    if (CACHE/'signature.json').exists() and json.loads((CACHE/'signature.json').read_text())!=signature:
        raise ValueError('Cache inputs changed; use a new version')
    if (CACHE/'prepared.json').exists():
        print('PREPARE complete cache reused',flush=True);return
    tokens=pickle.loads((ROOT/prep['token_dictionary']).read_bytes())
    medians=pickle.loads((ROOT/prep['gene_median_reference']).read_bytes())
    seen=set();metadata=[];inputs=[];train_genes=None;counts_summary=[]
    for role,batch in [('train','C'),('validation','D')]:
        a=ad.read_h5ad(paths[role]);a=a[a.obs.condition.isin(['PBS','IL33'])].copy()
        if (not a.obs.split.eq(role).all() or not a.obs.author_batch.eq(batch).all()
            or not a.obs.study.eq('GSE136154').all() or not a.obs.qc_version.eq('ilc2_balanced_v1').all()
            or json.loads(a.uns['split_provenance_json'])['split_version']!=CFG['split_version']):
            raise ValueError('Incorrect C/D roles, QC or study')
        if not a.obs_names.is_unique or not a.var_names.is_unique or seen.intersection(a.obs_names):
            raise ValueError('Duplicate or overlapping cells/genes')
        if not a.var_names.str.match(r'^ENSMUSG\d+$').all():raise ValueError('Unstable gene IDs')
        seen.update(a.obs_names)
        x=_raw_counts(a.layers['counts'])
        if (x!=a.X).nnz:raise ValueError('Original raw counts mismatch')
        if role=='train':
            train_genes=a.var_names.copy()
            genes=a.var[['gene_symbol']].copy()
            genes['train_detected_fraction']=np.asarray((x>0).mean(0)).ravel()
            genes['DEG_eligible']=genes.train_detected_fraction>=CFG['deg_train_detection_fraction']
            genes.to_csv(CACHE/'genes.tsv',sep='\t',index_label='gene_id')
        elif not a.var_names.equals(train_genes):raise ValueError('C/D gene order differs')
        depth=np.asarray(x.sum(1)).ravel()
        if (depth<=0).any():raise ValueError('Zero-count cells')
        y=x.astype('float32').multiply((1e4/depth)[:,None]).tocsr();y.data=np.log1p(y.data)
        np.save(CACHE/f'{role}_y.npy',y.toarray().astype('float32'))
        obs=a.obs[['study','library_id','gsm','author_batch','condition','split','qc_version']].copy()
        obs.to_csv(CACHE/f'{role}_metadata.tsv.gz',sep='\t',index_label='cell_id')
        pbs=np.flatnonzero(a.obs.condition.eq('PBS'))
        col,med,_=map_genes(a.var_names,tokens,medians)
        ids,vals,length=encode(x[pbs],col,med,prep['max_gene_tokens'],prep['encoding'])
        inputs.append((ids,vals,length))
        pm=obs.iloc[pbs].copy();pm['source_row']=pbs;metadata.append(pm)
        for condition,n in obs.condition.value_counts().items():
            counts_summary.append(dict(split=role,batch=batch,condition=condition,cells=int(n),genes=a.n_vars))
        print('PREPARED',role,a.shape,'PBS',len(pbs),flush=True)
        del a,x,y;gc.collect()
    np.savez_compressed(CACHE/'PBS_inputs.npz',ids=np.concatenate([v[0] for v in inputs]),
                        vals=np.concatenate([v[1] for v in inputs]),length=np.concatenate([v[2] for v in inputs]))
    pd.concat(metadata).to_csv(CACHE/'PBS_metadata.tsv.gz',sep='\t',index_label='cell_id')
    pd.DataFrame(counts_summary).to_csv(OUT/'cell_counts.tsv',sep='\t',index=False)
    # This checkpoint used the unchanged six-study posttraining pool; GSE136154 is downstream-only.
    pool=pd.read_csv(ROOT/'data/processed/ILC2_posttrain_update/ILC2_posttrain_20261021_update_metadata.tsv.gz',
                     sep='\t',usecols=['study'])
    if pool.study.eq('GSE136154').any():raise ValueError('Downstream study present in posttraining pool')
    write_json(CACHE/'signature.json',signature)
    write_json(CACHE/'prepared.json',dict(PBS_cells=sum(len(m) for m in metadata),counts_integer=True,
        train_validation_disjoint=True,genes=len(train_genes),conditions=['PBS','IL33'],posttrain_study_overlap=False))
    write_json(OUT/'config.json',dict(**CFG,signature=signature,
        encoder='Frozen selected tissue-supervised update14000; original tissue/MLM heads discarded',
        output='All 27998 measured genes; nonnegative predicted log1p(CP10K), not generated raw counts',
        model='clamp(PBS expression + C empirical mean delta + ResidualPredictor(768+32 ->128 GELU->G),min=0)',
        condition='Only IL33 [1,0]; fixed single stimulus, no learned unseen-drug representation',
        loss='standardized unpaired group mean MSE + 0.1 standardized group variance MSE + 0.01 scaled residual square',
        selection='Final epoch15 for every seed; D monitored only, no checkpoint/architecture/DEG tuning',
        DEG='>=1% detected in C PBS/IL33; nominal cell-level Mann-Whitney/Wilcoxon rank test, BH<0.05; p ascending then absolute mean change descending',
        metrics='Pearson of gene means and control-centered deltas; coefficient-of-determination R2 for gene means/variances, ddof0; true D top20/50/100 DEG subsets; overlap and sign accuracy',
        limitations=['D repeatedly used for project development; no new independent test',
                     'One training response pair and one validation pair; cells are not biological replicates',
                     'Unpaired moment matching does not identify true single-cell trajectories',
                     'Cell-level DEG p-values are descriptive, not animal-level significance'],
        combination_data_read=False,external_studies_read=False))


def encode_controls():
    sys.path.insert(0,str(ROOT/'src/downstream/compare'))
    from evaluate_tissue import fill_embeddings
    meta=pd.read_csv(CACHE/'PBS_metadata.tsv.gz',sep='\t',index_col=0)
    destination=CACHE/'PBS_h.npy'
    if destination.exists():
        h=np.load(destination,mmap_mode='r')
        if h.shape!=(len(meta),768) or not np.isfinite(h).all():raise ValueError('Invalid complete cache')
        print('ENCODE complete cache reused',flush=True);return
    inp=np.load(CACHE/'PBS_inputs.npz')
    partial=CACHE/'PBS_h.partial.npy';exists=partial.exists()
    h=np.lib.format.open_memmap(partial,mode='r+' if exists else 'w+',dtype='float32',shape=(len(meta),768))
    if not exists:h[:]=np.nan;h.flush()
    fill_embeddings(ROOT/CFG['checkpoint'],h,np.arange(len(meta)),inp['ids'],inp['vals'],inp['length'],'ILC2_tissue_PBS')
    if not np.isfinite(h).all():raise ValueError('Incomplete encoder cache')
    h.flush();del h;partial.replace(destination)


def load_data():
    meta={r:pd.read_csv(CACHE/f'{r}_metadata.tsv.gz',sep='\t',index_col=0) for r in ['train','validation']}
    y={r:np.load(CACHE/f'{r}_y.npy',mmap_mode='r') for r in meta}
    pm=pd.read_csv(CACHE/'PBS_metadata.tsv.gz',sep='\t',index_col=0)
    h=np.load(CACHE/'PBS_h.npy',mmap_mode='r')
    groups={}
    for role in meta:
        rows=pm.split.eq(role).to_numpy();p=np.flatnonzero(meta[role].condition.eq('PBS'));t=np.flatnonzero(meta[role].condition.eq('IL33'))
        if not pm.index[rows].equals(meta[role].index[p]) or not np.array_equal(pm.source_row[rows],p):
            raise ValueError('PBS embedding/expression/cell identity mismatch')
        groups[role]=dict(h=np.asarray(h[rows]),pbs=np.asarray(y[role][p]),target=np.asarray(y[role][t]))
    return groups


@torch.no_grad()
def predict(model,h,y,device):
    model.eval();out=[]
    for left in range(0,len(h),128):
        hh=torch.as_tensor(h[left:left+128],device=device)
        yy=torch.as_tensor(y[left:left+128],device=device)
        c=torch.tensor([1.,0.],device=device).expand(len(hh),-1)
        out.append(model(hh,yy,c).clamp_min(0).cpu().numpy())
    return np.concatenate(out)


def fit():
    if not torch.cuda.is_available():raise RuntimeError('CUDA needed for this training run')
    groups=load_data();tr,va=groups['train'],groups['validation']
    h0=torch.tensor(tr['h'],device='cuda');y0=torch.tensor(tr['pbs'],device='cuda');y1=torch.tensor(tr['target'],device='cuda')
    delta=y1.mean(0)-y0.mean(0)
    combined=torch.cat([y0,y1]);scale=combined.std(0,unbiased=False).clamp_min(CFG['minimum_expression_scale']);del combined
    np.save(CACHE/'C_mean_delta.npy',delta.cpu().numpy())
    for seed in CFG['seeds']:
        folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():print('FIT completed seed reused',seed,flush=True);continue
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=ResidualPredictor(h0,torch.stack([delta,torch.zeros_like(delta)]),CFG['hidden_size']).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        # The zero head must start at exactly the clipped expression-space mean-delta baseline.
        with torch.no_grad():
            c=torch.tensor([1.,0.],device='cuda').expand(8,-1)
            torch.testing.assert_close(model(h0[:8],y0[:8],c).clamp_min(0),(y0[:8]+delta).clamp_min(0))
        history=[];start=time.monotonic()
        for epoch in range(1,CFG['epochs']+1):
            model.train();order=rng.permutation(len(y1));totals=np.zeros(4)
            for left in range(0,len(order),CFG['batch_size']):
                target=order[left:left+CFG['batch_size']];src=rng.choice(len(y0),len(target),replace=True)
                c=torch.tensor([1.,0.],device='cuda').expand(len(src),-1)
                raw=model(h0[src],y0[src],c);pred=raw.clamp_min(0);truth=y1[target]
                mean_loss=((pred.mean(0)-truth.mean(0))/scale).square().mean()
                var_loss=((pred.var(0,unbiased=False)-truth.var(0,unbiased=False))/scale.square()).square().mean()
                penalty=((raw-y0[src]-delta)/scale).square().mean()
                loss=mean_loss+CFG['variance_loss_weight']*var_loss+CFG['residual_penalty']*penalty
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();totals+=np.array([loss.item(),mean_loss.item(),var_loss.item(),penalty.item()])*len(target)/len(y1)
            vp=predict(model,va['h'],va['pbs'],'cuda')
            rmse=float(np.sqrt(np.mean((vp.mean(0,dtype=np.float64)-va['target'].mean(0,dtype=np.float64))**2)))
            row=dict(epoch=epoch,loss=totals[0],mean_loss=totals[1],variance_loss=totals[2],residual_penalty=totals[3],D_delta_RMSE=rmse)
            history.append(row);print('TRAIN',seed,row,'elapsed',round(time.monotonic()-start),flush=True)
            pd.DataFrame(history).to_csv(folder/'history.tsv',sep='\t',index=False)
        torch.save(dict(model=model.state_dict(),config=CFG,epoch=epoch,signature=json.loads((CACHE/'signature.json').read_text()),
                        gene_ids=pd.read_csv(CACHE/'genes.tsv',sep='\t').gene_id.tolist()),folder/'head.pt')
        np.save(folder/'D_predicted_log1p.npy',vp.astype('float32'))
        train_prediction=predict(model,tr['h'],tr['pbs'],'cuda')
        write_json(folder/'complete.json',dict(seed=seed,epochs=epoch,trainable_parameters=sum(p.numel() for p in model.parameters()),
            elapsed_seconds=time.monotonic()-start,train_mean_RMSE=float(np.sqrt(np.mean((train_prediction.mean(0)-tr['target'].mean(0))**2))),
            encoder_frozen=True,checkpoint_selection='fixed_final_epoch',D_used_for_gradients=False))
        del model,optimizer,vp,train_prediction;torch.cuda.empty_cache()
    print('FIT COMPLETE',flush=True)


def score():
    groups=load_data();va=groups['validation']
    genes=pd.read_csv(CACHE/'genes.tsv',sep='\t').set_index('gene_id')
    def save_top(name,de,order):
        top=de.iloc[order[:100]].copy()
        top['gene_id']=genes.index[order[:100]]
        top['gene_symbol']=genes.gene_symbol.iloc[order[:100]].to_numpy()
        top.insert(0,'DEG_rank',np.arange(1,len(top)+1))
        top.to_csv(OUT/f'{name}_top100_DEG.tsv',sep='\t',index=False)
    true_de,true_order=differential(va['target'],va['pbs'],genes.DEG_eligible.to_numpy())
    pd.concat([genes.reset_index(),true_de],axis=1).to_csv(OUT/'D_observed_DEG.tsv.gz',sep='\t',index=False)
    np.save(OUT/'D_true_DEG_order.npy',true_order)
    save_top('observed',true_de,true_order)
    delta=np.load(CACHE/'C_mean_delta.npy')
    models=['control_copy','C_mean_delta_unclipped','C_mean_delta_clipped']+[f'head_seed{s}' for s in CFG['seeds']]
    rows=[];export=genes.copy()
    for name in models:
        if name=='control_copy':pred=va['pbs']
        elif name=='C_mean_delta_unclipped':pred=va['pbs']+delta
        elif name=='C_mean_delta_clipped':pred=np.maximum(va['pbs']+delta,0)
        else:pred=np.load(OUT/f"seed{name.rsplit('seed',1)[1]}"/'D_predicted_log1p.npy',mmap_mode='r')
        de,order=differential(pred,va['pbs'],genes.DEG_eligible.to_numpy())
        metrics,values=evaluate(pred,va['target'],va['pbs'],true_order,order)
        eligible=genes.DEG_eligible.to_numpy()
        metrics.update(Pearson_delta_train_expressed=correlation(values['predicted_delta'][eligible],values['observed_delta'][eligible]),
                       R2_mean_train_expressed=r2(values['observed_mean'][eligible],values['predicted_mean'][eligible]),
                       R2_var_train_expressed=r2(values['observed_var'][eligible],values['predicted_var'][eligible]))
        rows.append(dict(model=name,**metrics))
        de.insert(0,'gene_id',genes.index)
        de.to_csv(OUT/f'{name}_DEG.tsv.gz',sep='\t',index=False)
        save_top(name,de,order)
        for key,value in values.items():export[f'{name}_{key}']=value
        print('SCORED',name,metrics,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.tsv',sep='\t',index=False)
    export.to_csv(OUT/'D_gene_predictions.tsv.gz',sep='\t')
    report()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(ROOT/'external/fonts/Arial.ttf')
    plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':9})
    table=pd.read_csv(OUT/'metrics.tsv',sep='\t')
    seeds=table[table.model.str.startswith('head_seed')]
    means=seeds.drop(columns='model').mean();sd=seeds.drop(columns='model').std(ddof=1)
    pd.DataFrame(dict(mean=means,seed_SD=sd)).to_csv(OUT/'head_summary.tsv',sep='\t')
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for seed in CFG['seeds']:
        h=pd.read_csv(OUT/f'seed{seed}/history.tsv',sep='\t')
        axes[0].plot(h.epoch,h.loss,label=f'seed {seed}')
        axes[1].plot(h.epoch,h.D_delta_RMSE)
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Train moment loss');axes[0].legend(frameon=False)
    axes[1].set_xlabel('Epoch');axes[1].set_ylabel('D mean-expression RMSE')
    axes[1].axhline(table.set_index('model').loc['C_mean_delta_clipped','delta_RMSE'],
                   color='#666666',linestyle='--',linewidth=.8,label='Mean delta')
    axes[1].legend(frameon=False)
    base=table[table.model.isin(['control_copy','C_mean_delta_clipped'])]
    values=list(base.delta_RMSE)+[means.delta_RMSE]
    axes[2].bar(range(3),values,color=['#999999','#4477AA','#228877'],yerr=[0,0,sd.delta_RMSE],capsize=3)
    axes[2].set_xticks(range(3),['Control','Mean delta','Encoder head'],rotation=20);axes[2].set_ylabel('D delta RMSE')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'training.{ext}',dpi=250)
    plt.close(fig)
    prediction=pd.read_csv(OUT/'D_gene_predictions.tsv.gz',sep='\t',index_col=0)
    eligible=prediction.DEG_eligible.to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(6.6,3))
    for ax,name,label in zip(axes,['C_mean_delta_clipped','head_seed0'],['Mean delta','Frozen ILC2 head (seed 0)']):
        x=prediction.loc[eligible,f'{name}_observed_delta']
        y=prediction.loc[eligible,f'{name}_predicted_delta']
        ax.scatter(x,y,s=3,alpha=.3,c='#4477AA',rasterized=True,linewidths=0)
        lower=min(x.min(),y.min());upper=max(x.max(),y.max())
        ax.plot([lower,upper],[lower,upper],color='#555555',lw=.7)
        ax.set_xlabel('Observed IL-33 change');ax.set_ylabel('Predicted change');ax.set_title(label)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'response_scatter.{ext}',dpi=250)
    plt.close(fig)
    baseline=table.set_index('model').loc['C_mean_delta_clipped']
    linear=table.set_index('model').loc['C_mean_delta_unclipped']
    improvement=100*(1-means.delta_RMSE/baseline.delta_RMSE)
    wins=int((seeds.delta_RMSE<baseline.delta_RMSE).sum())
    summary=['| 指标 | Control-copy | Mean delta（非负） | 新版头：3 seed均值 ± SD |',
             '|---|---:|---:|---:|']
    for key in ['Pearson','Pearson_delta','delta_RMSE','R2_mean_all','R2_var_all','R2_mean_DEG100','R2_var_DEG100',
                'DEG_overlap_20','DEG_overlap_50','DEG_overlap_100','DEG_direction_20','DEG_direction_50','DEG_direction_100']:
        ctrl=table.set_index('model').loc['control_copy',key]
        fmt=lambda v:'NA' if pd.isna(v) else f'{v:.4f}'
        summary.append(f'| {key} | {fmt(ctrl)} | {fmt(baseline[key])} | {fmt(means[key])} ± {fmt(sd[key])} |')
    columns=['model','Pearson','Pearson_delta','R2_mean_all','R2_var_all','delta_RMSE','DEG_overlap_20','DEG_overlap_50','DEG_overlap_100',
             'DEG_direction_20','DEG_direction_50','DEG_direction_100']
    text=['# 冻结新版ILC2 encoder：肺IL33全基因响应预测','',
        '只用GSE136154：C PBS3762/IL333674训练，D PBS3329/IL334224开发评价。原balanced_v1，27998个基因全部输出。未读CGRP、双刺激或其他研究用于拟合/评分。原始counts与角色保持不变。',
        'encoder为组织监督新版update14000，FP32/median/top2048/有效gene token均值；完全冻结，只编码PBS，不把真实IL33表示输入预测器。',
        '头复用ResidualPredictor：标准化768维PBS embedding+固定IL33编码32维→128 GELU→27998；加到PBS表达和C训练均值效应上，再逐细胞非负截断。单刺激条件编码是常量，不代表学到药物化学或未见刺激泛化。',
        '固定15epoch、seed0/1/2，batch64。同C背景独立采样PBS与IL33，均值MSE+0.1方差MSE+0.01残差惩罚；均按C表达尺度归一化。每epoch遍历C IL33一次，PBS有放回。无逐对MSE或真实单细胞匹配；方差来自模型对不同PBS的确定性输出，不是验证过的单细胞命运分布。',
        '仅最终epoch15评分，不根据D选择checkpoint/DEG或调整参数。D曾反复用于项目开发，因此本轮不是独立盲测；C/D各条件是一个作者批次文库，不能把细胞数当生物重复数。','',
        '## 主要结果',
        f'新版头相对非负mean-delta的D delta-RMSE改善为{improvement:.2f}%，{wins}/3个seed误差更低（负改善即退步）。模型最终固定15epoch，不因这个结果重选训练轮数。',
        f'均值误差基本持平：head RMSE {means.delta_RMSE:.6f} ± {sd.delta_RMSE:.6f}，非负mean delta {baseline.delta_RMSE:.6f}。保留负值的线性mean delta更低（{linear.delta_RMSE:.6f}），但有{100*linear.negative_fraction:.2f}%逐细胞预测值为负，故作为数学参考与非负对照同时报告。',
        f'主要收益在方差及DEG排序：all-gene variance R2 {baseline.R2_var_all:.4f}→{means.R2_var_all:.4f}，Top100 DEG overlap {baseline.DEG_overlap_100:.1%}→{means.DEG_overlap_100:.1%}。方差也是训练目标之一，不能把收益直接归因于新版encoder。',
        'Mean delta与head的top20/50/100方向准确率都达到100%，该指标在本次数据上没有区分它们。Pearson delta提高也不能替代幅度误差。当前仅比较新版头与简单基线，没有证明新版encoder优于原始GeneCompass或其他表示。',
        '',*summary,'',
        '## D逐模型结果','```tsv',table[columns].to_csv(sep='\t',index=False,float_format='%.6f'),'```','',
        '## 三个head种子汇总','```tsv',pd.DataFrame(dict(mean=means,seed_SD=sd)).to_csv(sep='\t',float_format='%.6f'),'```','',
        '## 指标定义',
        '表达空间为log1p(CP10K)，归一化分母含全部27998个原始基因。Pearson比较预测/真实处理组的逐基因均值；Pearson delta均减去同一个D PBS均值。control-copy的预测delta恒零，因此其Pearson delta未定义，保存NA，不填0。',
        'R2_mean_all是跨全部基因均值的决定系数，R2_var_all是跨基因细胞间方差的决定系数（ddof0）；不是相关系数平方，允许负值。DEG20/50/100的R2_mean/R2_var在真实D相应top DEG集合计算，详见metrics.tsv。另报delta RMSE/MAE和总方差比。',
        f'输出{len(prediction)}基因；其中{int(prediction.train_detected_fraction.eq(0).sum())}个在C完全未检出，{int(eligible.sum())}个满足C至少1%检出。另报训练表达基因子集的delta Pearson和R2，避免全基因大量低/零表达掩盖问题。',
        'DEG评分仅在训练C中至少1%细胞检出的基因中进行。真实D IL33与D PBS、预测与同D PBS各自用双侧Mann-Whitney/Wilcoxon秩和检验，BH<0.05；按p升序、绝对均值变化降序、稳定基因顺序打破同值。Top-N overlap=两组top-N交集数/真实候选数，方向准确率在真实top-N上比较平均变化符号，预测零变化计不正确。若显著候选不足N，报告实际分母。',
        '这些DEG是细胞层面的描述性排序，不是独立动物的统计DE真值。预测细胞继承PBS且缺乏真实配对，其名义p值也不应解释成实验显著性。未读取D DEG用于训练、输出基因选择或选模。',
        'C_mean_delta_unclipped：D PBS每细胞+C(IL33均值−PBS均值)，保留负值作线性基线；clipped版本截断到0，作为非负模型的主要对照。非负截断会改变均值与方差，两者不能混用。',
        'seed_SD仅为训练随机性，不是动物置信区间。高绝对表达Pearson可能来自保留PBS背景，需结合Pearson delta、delta RMSE、DEG及方差指标判断。','',
        '## 产物',
        'metrics.tsv含全部指标，head_summary.tsv为均值/seed_SD；D_gene_predictions.tsv.gz保留逐基因均值、方差与变化；各模型DEG表、D_observed_DEG.tsv.gz保留完整候选排序信息；*_top100_DEG.tsv保存可直接审阅的前100候选。',
        'seed*/head.pt、history.tsv、D_predicted_log1p.npy为实际预测头/训练曲线/逐细胞预测表达。训练前配置与缓存签名、数据角色和元信息均保留。',
        'training.png/pdf为训练及D监测曲线；response_scatter.png/pdf固定展示seed0与mean delta的变化散点，不按效果挑seed，三个seed的指标分别计算后汇总。',
        '代码src/downstream/drugpred/train_il33_lung_full.py；指标il33_lung_metrics.py；配置configs/drugpred_il33_lung_tissue.json。',
        '复现阶段：prepare / encode / fit / score / report / all。GPU用于encode和fit。','']
    (OUT/'report.md').write_text('\n'.join(text))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','encode','fit','score','report','all'])
    args=parser.parse_args();torch.set_num_threads(CFG['cpu_threads'])
    with threadpool_limits(limits=CFG['cpu_threads']):
        if args.stage in ['prepare','all']:prepare_data()
        if args.stage!='prepare':
            recorded=json.loads((OUT/'config.json').read_text())
            if any(recorded[k]!=v for k,v in CFG.items()):
                raise ValueError('Experiment settings changed; use a new result version')
        if args.stage in ['encode','all']:encode_controls()
        if args.stage in ['fit','all']:fit()
        if args.stage in ['score','all']:score()
        if args.stage=='report':report()
