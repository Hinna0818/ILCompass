"""IL33-only population baselines; read explicit development files, never external targets."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k] = '4'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/ilc2_numba'
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'
import json
import time
import hashlib
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from train_group_head import GroupHead, metrics
from prepare import write_json
from models import ROOT

DATA = Path(__file__).resolve().parent / 'downstream_data/IL33_expanded_v1'
OUT = ROOT / 'results/IL33_drugpred_baseline_v1'
CFG = dict(seeds=[0,1,2], n_hvg=2000, markers=['Il5','Il13','Areg','Gata3','Rora','Il1rl1','Il7r','Il2ra'],
    pca_components=32, hidden_size=64, group_size=128, max_steps=5000,
    validation_every=500, patience=4, learning_rate=.001, weight_decay=.01,
    study_weights={'GSE136154_C':.5,'GSE102299_rep1':.25,'GSE102299_rep2':.25})


def write_report():
    summary=json.loads((OUT/'summary.json').read_text())
    table=pd.read_csv(OUT/'comparison.tsv',sep='\t')
    mlp=table.loc[table.model.eq('expression_PCA_MLP')]
    means=mlp.mean(numeric_only=True);sd=mlp.std(numeric_only=True,ddof=1)
    baseline=table.set_index('model').loc['C_mean_effect','delta_rmse']
    pooled=table.set_index('model').loc['study_balanced_mean_effect','delta_rmse']
    prediction=pd.read_csv(OUT/'D_predictions.tsv.gz',sep='\t')
    # Validate saved predictions against reported scores without accessing any new labels.
    for row in table.itertuples():
        name=f'mlp_seed{int(row.seed)}' if row.model=='expression_PCA_MLP' else row.model
        rmse=np.sqrt(np.mean((prediction[name+'_delta']-prediction.observed_delta)**2))
        if not np.isclose(rmse,row.delta_rmse,rtol=1e-5):
            raise ValueError('Prediction export/metric mismatch')
    markers=prediction.loc[prediction.gene_symbol.isin(CFG['markers'])].copy()
    markers['MLP_mean_delta']=markers[[f'mlp_seed{s}_delta' for s in CFG['seeds']]].mean(axis=1)
    markers[['gene_id','gene_symbol','observed_delta','C_mean_effect_delta','study_balanced_mean_effect_delta','MLP_mean_delta']].to_csv(
        OUT/'functional_genes.tsv',sep='\t',index=False)
    lines=['# IL33 扩充数据：表达基线训练结果', '',
        f"训练 {summary['train_cells']:,}、开发验证 {summary['validation_cells']:,} 个细胞；输出 {summary['panel_genes']:,} 个训练集HVG+预设marker。",
        'PBS/IL33 only；训练C+GSE102299两个重复，D用于早停。外部GSE189285与双刺激未读取。',
        '', '| 模型 | D ΔRMSE↓ | ΔPearson↑ | ΔSpearman↑ | Top20交集 |',
        '|---|---:|---:|---:|---:|']
    for r in table.iloc[:4].itertuples():
        f=lambda v:'—' if pd.isna(v) else f'{v:.6f}'
        lines.append(f'| {r.model} | {r.delta_rmse:.6f} | {f(r.delta_pearson)} | {f(r.delta_spearman)} | {f(r.top20_overlap)} |')
    lines += [f"| PCA-MLP，3 seeds | {means.delta_rmse:.6f} ± {sd.delta_rmse:.6f} | {means.delta_pearson:.6f} | {means.delta_spearman:.6f} | {means.top20_overlap:.2f}/20 |",'',
        f'MLP比扩充研究平均效应RMSE低 {100*(1-means.delta_rmse/pooled):.2f}%，比仅C平均效应低 {100*(1-means.delta_rmse/baseline):.2f}%。'
        f'直接平均新增研究响应使D RMSE比仅C高 {100*(pooled/baseline-1):.2f}%，扩充数据不保证改善同研究D。',
        f'MLP MAE={means.mae:.6f}，仅C平均效应MAE={table.set_index("model").loc["C_mean_effect","mae"]:.6f}；'
        'RMSE的小幅改善没有转化为所有指标的提升。', '',
        '结构：训练PBS表达PCA32 + 固定IL33条件向量2 → Linear64 → GELU → LinearG。'
        '输出为 max(0, PBS群体均值 + 研究平衡平均效应 + MLP残差)，共132,435个参数。'
        '本轮没有使用Foundation Model编码器，是后续ILC2模型的表达基线。',
        '训练按C/旧研究rep1/rep2以0.5/0.25/0.25抽样。每步从同一response_group独立抽128个PBS和128个IL33，'
        '最小化处理群体平均log1p(CP10K)的MSE，不假设真实细胞配对。'
        'AdamW lr=0.001、weight_decay=0.01、最多5000步，每500步验证，patience=4。初始平均效应也参与选模。',
        '三个seed均选第500步，均在2500步早停。训练重采样步数不等于完整数据集epoch。', '',
        f'最佳MLP在D预测均值中约 {100*means.zero_mean_gene_fraction:.1f}% 基因为0；'
        f'D真实处理均值零基因比例为 {100*prediction.observed_IL33_mean.eq(0).mean():.1f}%。'
        '非负截断使许多低表达基因归零，是当前模型的限制；不能把RMSE提升理解成全转录组准确重建。',
        'Top20按平均表达变化绝对值排序；D没有充分独立生物重复，因此这是观测响应排序，不是统计显著DE的ground truth。'
        '三个seed标准差反映训练随机性，不是生物学置信区间。',
        '所有模型使用同一输出面板；与旧版2008基因实验的数值不可直接横向比较。仅有两个研究、三个训练响应对，'
        'D又与C同研究，当前结果不能证明跨研究泛化，更不能证明ILC2后训练收益。', '',
        '下一步：固定当前面板和基线，比较冻结ILC2表示接同尺寸简单头；然后再按README的2026-10-25计划增加生成模块。'
        '生成模型除平均响应外还需分布指标，并检查非负截断/低表达基因；外部IL33继续留到架构、超参和评价方案冻结后。',
        '', '复现：`python src/downstream/drugpred/train_il33_baselines.py`。完成目录不可覆盖。'
        '逐seed结果、训练日志、最佳checkpoint、PCA、gene panel、D逐基因预测和功能基因表均在本目录。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def score(pred, truth, control):
    r = metrics(pred,truth,control)
    dp=(pred-control).numpy();dt=(truth-control).numpy()
    r['mae']=float(np.abs(dp-dt).mean())
    r['delta_spearman']=float(spearmanr(dp,dt).statistic) if np.ptp(dp)>0 else None
    order_p=np.argsort(-np.abs(dp),kind='stable')[:20]
    order_t=np.argsort(-np.abs(dt),kind='stable')[:20]
    # A zero-response model has no ranked perturbation candidates.
    r['top20_overlap']=int(len(set(order_p)&set(order_t))) if np.ptp(dp)>0 else None
    r['top20_direction_accuracy']=float((np.sign(dp[order_t])==np.sign(dt[order_t])).mean())
    return r


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():
        raise ValueError('Completed experiment exists; do not overwrite')
    torch.set_num_threads(4)
    manifest=json.loads((DATA/'development_manifest.json').read_text())
    write_json(OUT/'config.json',dict(**CFG, model='32 PCA features + fixed 2D condition ->64 GELU->G residual',
        target='population mean log1p(CP10K); all input genes in normalization denominator',
        selection='HVG from training PBS/IL33 only; PCA from training PBS only',
        loss='MSE of independently sampled PBS/IL33 group means within one response_group',
        external_accessed=False, foundation_encoder_used=False))
    matrices={};metadata={};hashes={};start=time.monotonic()
    for split, expected in [('train','train'),('validation','development_validation')]:
        path=DATA/manifest[split]['path']
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        a=ad.read_h5ad(path)
        if not a.obs.split.eq(expected).all() or not a.obs.qc_version.eq('ilc2_balanced_v1').all():
            raise ValueError('Unexpected split/QC')
        a=a[a.obs.condition.isin(['PBS','IL33'])].copy()
        if split=='train':
            train_cells=set(a.obs_names)
        elif train_cells & set(a.obs_names):
            raise ValueError('Overlapping splits')
        metadata[split]=a.obs.copy()
        sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
        if split=='train':
            sc.pp.highly_variable_genes(a,n_top_genes=CFG['n_hvg'],flavor='seurat')
            panel=a.var.loc[a.var.highly_variable | a.var.gene_symbol.isin(CFG['markers'])].copy()
            ids=panel.index
            panel.to_csv(OUT/'gene_panel.tsv',sep='\t',index_label='gene_id')
        matrices[split]=torch.from_numpy(a[:,ids].X.toarray().astype('float32'))
        print('PREPARED',split,matrices[split].shape,flush=True)
        del a
    y=matrices['train'];v=matrices['validation'];obs=metadata['train'];vo=metadata['validation']
    pools={g:{c:np.flatnonzero(obs.response_group.eq(g)&obs.condition.eq(c)) for c in ['PBS','IL33']}
           for g in CFG['study_weights']}
    vp={c:np.flatnonzero(vo.condition.eq(c)) for c in ['PBS','IL33']}
    pbs=np.flatnonzero(obs.condition.eq('PBS'))
    pca=PCA(n_components=CFG['pca_components'],svd_solver='randomized',random_state=0)
    pca.fit(y[pbs].numpy())
    h=torch.from_numpy(pca.transform(y.numpy()).astype('float32'))
    vh=torch.from_numpy(pca.transform(v.numpy()).astype('float32'))
    np.savez_compressed(OUT/'pca.npz',mean=pca.mean_,components=pca.components_)
    effects={g:y[p['IL33']].mean(0)-y[p['PBS']].mean(0) for g,p in pools.items()}
    effect=sum(CFG['study_weights'][g]*effects[g] for g in pools)
    ctrl=v[vp['PBS']].mean(0);truth=v[vp['IL33']].mean(0);hm=vh[vp['PBS']].mean(0)
    predictions={'control_copy':ctrl,'C_mean_effect':(ctrl+effects['GSE136154_C']).clamp_min(0),
        'GSE102299_mean_effect':(ctrl+(effects['GSE102299_rep1']+effects['GSE102299_rep2'])/2).clamp_min(0),
        'study_balanced_mean_effect':(ctrl+effect).clamp_min(0)}
    results=[dict(model=name,seed=None,**score(pred,truth,ctrl)) for name,pred in predictions.items()]
    condition=torch.tensor([1.,0.]);run_details=[]
    for seed in CFG['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=GroupHead(h[pbs],torch.stack([effect,torch.zeros_like(effect)]),CFG['hidden_size'],True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        folder=OUT/f'mlp_seed{seed}';folder.mkdir(exist_ok=True)
        best=score(predictions['study_balanced_mean_effect'],truth,ctrl)['delta_rmse'];best_step=0;bad=0
        torch.testing.assert_close(model(hm,ctrl,condition),predictions['study_balanced_mean_effect'])
        def save(step):
            torch.save(dict(model=model.state_dict(),seed=seed,step=step,config=CFG,gene_ids=list(ids),
                input_hashes=hashes),folder/'best.pt')
        save(0)
        with (folder/'history.jsonl').open('w',buffering=1) as log:
            for step in range(1,CFG['max_steps']+1):
                group=rng.choice(list(pools),p=list(CFG['study_weights'].values()))
                pi=rng.choice(pools[group]['PBS'],CFG['group_size'],replace=True)
                ti=rng.choice(pools[group]['IL33'],CFG['group_size'],replace=True)
                pred=model(h[pi].mean(0),y[pi].mean(0),condition)
                loss=(pred-y[ti].mean(0)).square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
                if step%CFG['validation_every']==0:
                    with torch.no_grad():
                        val_score=score(model(hm,ctrl,condition),truth,ctrl)
                        train_rmse={g:score(model(h[p['PBS']].mean(0),y[p['PBS']].mean(0),condition),
                            y[p['IL33']].mean(0),y[p['PBS']].mean(0))['delta_rmse'] for g,p in pools.items()}
                    row=dict(step=step,group_mse=float(loss.detach()),train_rmse=train_rmse,validation=val_score)
                    log.write(json.dumps(row)+'\n')
                    if val_score['delta_rmse']<best:
                        best=val_score['delta_rmse'];best_step=step;bad=0;save(step)
                    else:bad+=1
                    print('VALIDATION',seed,step,val_score['delta_rmse'],'best',best_step,best,flush=True)
                    if bad>=CFG['patience']:break
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
        with torch.no_grad():
            pred=model(hm,ctrl,condition)
            train_final={g:score(model(h[p['PBS']].mean(0),y[p['PBS']].mean(0),condition),
                y[p['IL33']].mean(0),y[p['PBS']].mean(0)) for g,p in pools.items()}
        row=dict(model='expression_PCA_MLP',seed=seed,**score(pred,truth,ctrl))
        results.append(row);predictions[f'mlp_seed{seed}']=pred
        details=dict(**row,best_step=best_step,stopped_step=step,train=train_final,
                     parameters=sum(p.numel() for p in model.parameters()))
        run_details.append(details);write_json(folder/'result.json',details)
    table=pd.DataFrame(results);table.to_csv(OUT/'comparison.tsv',sep='\t',index=False)
    frame=pd.DataFrame(dict(gene_id=ids,gene_symbol=panel.gene_symbol.to_numpy(),PBS_mean=ctrl.numpy(),
        observed_IL33_mean=truth.numpy(),observed_delta=(truth-ctrl).numpy()))
    for name,pred in predictions.items():frame[name+'_delta']=(pred-ctrl).numpy()
    frame.to_csv(OUT/'D_predictions.tsv.gz',sep='\t',index=False)
    write_json(OUT/'summary.json',dict(results=results,runs=run_details,panel_genes=len(ids),
        train_cells=len(obs),validation_cells=len(vo),input_hashes=hashes,
        elapsed_seconds=time.monotonic()-start,external_accessed=False,
        limitations=['Only three training control-treatment contrasts from two studies',
            'D is development validation; no independent external score',
            'Top20 ground truth is observed mean change, not replicated significant DE',
            'Seed variation is training randomness, not biological confidence interval',
            'No ILC2 foundation encoder used in expression baselines']))
    print(table.to_string(index=False),flush=True)
    write_report()


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
