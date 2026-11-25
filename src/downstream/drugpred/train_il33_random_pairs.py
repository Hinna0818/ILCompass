"""Frozen ILC2 + random within-background pairs, B=64; development data only."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k]='4'
os.environ['NUMBA_CACHE_DIR']='/tmp/ilc2_numba'
os.environ['MPLCONFIGDIR']='/tmp/ilc2_mpl'
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import torch
from threadpoolctl import threadpool_limits
from models import ROOT, FrozenILC2Encoder, OFFICIAL_CHECKPOINT
from prepare import write_json
from posttrain_data import encode, file_hash
from genecompass_coverage import map_genes
from train_group_head import GroupHead
from train_il33_baselines import DATA, CFG, score

OUT=ROOT/'results/IL33_drugpred_random_pairs_v1'
CACHE=DATA/'prepared_official_PBS_v1'
BASELINE=ROOT/'results/IL33_drugpred_baseline_v1'


def report():
    table=pd.read_csv(OUT/'comparison.tsv',sep='\t')
    previous=pd.read_csv(BASELINE/'comparison.tsv',sep='\t')
    rows=[]
    for name in ['C_mean_effect','study_balanced_mean_effect','expression_PCA_MLP']:
        values=previous.loc[previous.model.eq(name)]
        rows.append(dict(model=name,rmse=values.delta_rmse.mean(),sd=values.delta_rmse.std(ddof=1),
            pearson=values.delta_pearson.mean(),spearman=values.delta_spearman.mean(),
            mae=values.mae.mean(),top20=values.top20_overlap.mean(),zero=values.zero_mean_gene_fraction.mean()))
    for mode,values in table.groupby('mode',sort=False):
        rows.append(dict(model=mode,rmse=values.delta_rmse.mean(),sd=values.delta_rmse.std(ddof=1),
            pearson=values.delta_pearson.mean(),spearman=values.delta_spearman.mean(),
            mae=values.mae.mean(),top20=values.top20_overlap.mean(),zero=values.zero_mean_gene_fraction.mean()))
    comparison=pd.DataFrame(rows);comparison.to_csv(OUT/'baseline_comparison.tsv',sep='\t',index=False)
    trained=[]
    for seed in [0,1,2]:
        history=[json.loads(line) for line in (OUT/f'random_pair_cell_mse_seed{seed}/history.jsonl').read_text().splitlines()]
        best=min(history,key=lambda r:r['validation']['delta_rmse'])
        trained.append(dict(seed=seed,best_trained_step=best['step'],best_trained_rmse=best['validation']['delta_rmse']))
    write_json(OUT/'random_pair_trained_only.json',trained)
    predictions=pd.read_csv(OUT/'D_predictions.tsv.gz',sep='\t')
    for r in table.itertuples():
        pred=predictions[f'{r.mode}_seed{r.seed}'].to_numpy()
        if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid exported predictions')
        if not np.isclose(np.sqrt(np.mean((pred-predictions.target_mean)**2)),r.delta_rmse,rtol=1e-5):
            raise ValueError('Prediction/metric inconsistency')
    panel=pd.read_csv(BASELINE/'gene_panel.tsv',sep='\t').set_index('gene_id')
    predictions['gene_symbol']=predictions.gene_id.map(panel.gene_symbol)
    predictions['observed_delta']=predictions.target_mean-predictions.control_mean
    for mode in ['random_pair_cell_mse','group_mean_mse']:
        predictions[mode+'_delta']=predictions[[f'{mode}_seed{s}' for s in [0,1,2]]].mean(axis=1)-predictions.control_mean
    predictions.loc[predictions.gene_symbol.isin(CFG['markers']),
        ['gene_id','gene_symbol','observed_delta','random_pair_cell_mse_delta','group_mean_mse_delta']].to_csv(
            OUT/'functional_genes.tsv',sep='\t',index=False)
    lines=['# 冻结 ILC2：64个随机配对与群体均值对照','',
        '仅PBS/IL33：训练24,649、D开发验证7,553细胞。固定上一轮训练集选择的2,003基因。'
        '没有读取GSE189285外部样本、双刺激或利用其标签调参。','',
        '| 模型 | D ΔRMSE↓ | seed SD | ΔPearson↑ | ΔSpearman↑ | MAE↓ | Top20命中 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison.itertuples():
        sd='—' if pd.isna(r.sd) else f'{r.sd:.6f}'
        lines.append(f'| {r.model} | {r.rmse:.6f} | {sd} | {r.pearson:.4f} | {r.spearman:.4f} | {r.mae:.6f} | {r.top20:.2f}/20 |')
    lines+=['',
        '**随机配对三个seed最佳均为第0步，表中0.079469是未训练初始头，并非训练收益。** '
        '只看实际训练后的检查点，各seed最优D RMSE分别为 '+
        ', '.join(f"{r['best_trained_rmse']:.6f} (step {r['best_trained_step']})" for r in trained)+
        '；随机配对实验没有改善。',
        'ILC2群体均值头RMSE均值0.053941，比旧PCA均值头低约0.62%，但MAE、Spearman更差；'
        '输入维度和batch也不同，不能声称FM明确优于PCA。建议保留群体响应目标，暂不把随机配对MSE作为主方案。',
        '', '## 实现与训练',
        '冻结0.8–0.2后训练ILC2：原始counts按官方gene median编码、top2048、FP32，'
        '最后层有效基因mean pooling排除species CLS，得到768维。C/D复用签名与来源校验通过的旧缓存，'
        '仅新增研究9,481个PBS细胞需要GPU编码；处理细胞无需编码。',
        '头：768维PBS表示按训练PBS标准化，concat固定[1,0]刺激编码→Linear64→GELU→Linear2003。'
        '预测为clamp(PBS表达+研究平衡平均效应+残差, min=0)。编码器冻结；只优化头。',
        '每步选择一个背景，C/rep1/rep2概率0.5/0.25/0.25；在该背景抽64个不同IL33细胞，'
        '为每个处理细胞独立、有放回抽一个PBS细胞。一批内PBS可重复，配对每步重新抽样。'
        '逐对模式对64×2003表达误差求MSE均值；不先平均64个细胞。'
        '均值模式用同一抽样序列先取群体均值再求MSE。',
        '两个模式均为seeds0/1/2，AdamW lr=0.001、weight_decay=0.01，batch64，最多5000步，'
        '每500步验证、patience4。初始未训练头参与选模。不是按完整epoch遍历；每步研究内重采样。',
        '逐对模式推理：分别预测全部D PBS细胞，再平均预测表达，对比D真实IL33均值；'
        '群体模式推理：先平均PBS表达/表示，再预测。非负截断与平均不可交换，因此两个模式初始值可能不同，'
        '每轮result.json均记录initial作为对应零修正基线。','',
        '## 最佳步与限制']
    for r in json.loads((OUT/'summary.json').read_text())['runs']:
        lines.append(f"- {r['mode']} seed{r['seed']}：best={r['best_step']}，stop={r['stopped_step']}；训练RMSE="+
            ', '.join(f"{g}:{v['delta_rmse']:.5f}" for g,v in r['train'].items()))
    lines+=['','随机配对并没有建立真实细胞对应。在背景内独立抽样时，逐对MSE的最优解倾向该背景的处理均值，'
        '所以这里检查的是这种训练策略的实际效果，不能将模型输出称为已验证的单细胞命运预测。',
        'D仅开发验证；seed SD不等于生物学置信区间。Top20按平均log1p表达变化绝对值排序，不是显著DE标签。'
        'PCA上一轮用group128且输入维度32，本轮FM用64且输入768；不能把二者差异全部归因于后训练。'
        '本轮两个FM模式具有相同头与抽样，比较的是逐对损失/逐细胞推理和群体均值目标。',
        '表达零比例（预测群体均值）：'+', '.join(f'{r.model}={r.zero:.1%}' for r in comparison.itertuples())+
        f'；D真实={predictions.target_mean.eq(0).mean():.1%}。',
        '', '脚本：src/downstream/drugpred/train_il33_random_pairs.py；每轮best.pt、result.json、history.jsonl以及D预测表在本目录。']
    (OUT/'report.md').write_text('\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n')
    write_json(OUT/'verification.json',dict(export_metrics_match=True,predictions_finite_nonnegative=True,
        external_accessed=False,gene_panel_identical_to_baseline=True))


def prepare():
    manifest=json.loads((DATA/'development_manifest.json').read_text())
    cfg=json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    old=DATA.parent/'GSE136154/prepared_official_E0_v1'
    oldsig=json.loads((old/'signature.json').read_text())
    signature=dict(checkpoint=file_hash(OFFICIAL_CHECKPOINT/'pytorch_model.bin'),
        tokens=file_hash(ROOT/cfg['token_dictionary']),medians=file_hash(ROOT/cfg['gene_median_reference']),
        encoding=cfg['encoding'],max_gene_tokens=cfg['max_gene_tokens'],precision='float32',pooling='masked_gene_mean')
    if any(oldsig[k]!=v for k,v in signature.items()) or oldsig!=json.loads((old/'complete.json').read_text())['signature']:
        raise ValueError('Old encoder cache signature mismatch')
    signature['inputs']={s:file_hash(DATA/manifest[s]['path']) for s in ['train','validation']}
    CACHE.mkdir(exist_ok=True)
    if (CACHE/'signature.json').exists() and json.loads((CACHE/'signature.json').read_text())!=signature:
        raise ValueError('Cache source changed')
    write_json(CACHE/'signature.json',signature)
    genes=pd.read_csv(BASELINE/'gene_panel.tsv',sep='\t').gene_id.tolist()
    tokens=pickle.loads((ROOT/cfg['token_dictionary']).read_bytes())
    medians=pickle.loads((ROOT/cfg['gene_median_reference']).read_bytes())
    encoder=None;data={}
    for split,role in [('train','train'),('validation','development_validation')]:
        a=ad.read_h5ad(DATA/manifest[split]['path'])
        if not a.obs.split.eq(role).all() or not a.obs.selected_ILC2.all():raise ValueError('Invalid split')
        a=a[a.obs.condition.isin(['PBS','IL33'])].copy()
        obs=a.obs.copy();x=a.X
        if not np.issubdtype(x.dtype,np.integer) or (x!=a.layers['counts']).nnz:raise ValueError('Invalid counts')
        y=x.astype('float32');y=y.multiply((1e4/np.asarray(y.sum(1)).ravel())[:,None]).tocsr();y.data=np.log1p(y.data)
        y=torch.from_numpy(y[:,a.var_names.get_indexer(genes)].toarray())
        pbs=np.flatnonzero(obs.condition.eq('PBS'));po=obs.iloc[pbs]
        dest=CACHE/f'{split}_pbs_h.npy'
        if not dest.exists():
            # Reuse C/D embeddings only after source file and cell ordering checks.
            oldfile=DATA.parent/'GSE136154'/split/f'GSE136154_{split}.h5ad'
            if file_hash(oldfile)!=oldsig['inputs'][split]:raise ValueError('Old source changed')
            om=pd.read_csv(old/f'{split}_metadata.tsv',sep='\t').set_index('cell_uid')
            oh=np.load(old/f'{split}_h.npy',mmap_mode='r')
            partial=CACHE/f'{split}_pbs_h.partial.npy';progress=CACHE/f'{split}_progress.json'
            h=np.lib.format.open_memmap(partial,mode='r+' if partial.exists() else 'w+',dtype='float32',shape=(len(pbs),768))
            isold=po.study.eq('GSE136154').to_numpy()
            indices=om.index.get_indexer(po.index[isold])
            if (indices<0).any():raise ValueError('Missing old cell ID')
            h[isold]=oh[indices]
            new=np.flatnonzero(~isold)
            if len(new):
                col,med,_=map_genes(a.var_names,tokens,medians)
                ids,vals,lengths=encode(x[pbs[new]],col,med,cfg['max_gene_tokens'],cfg['encoding'])
                if encoder is None:
                    if not torch.cuda.is_available():raise RuntimeError('GPU required for new frozen embeddings')
                    torch.backends.cuda.matmul.allow_tf32=False
                    encoder=FrozenILC2Encoder().to('cuda').eval()
                start=json.loads(progress.read_text())['completed_rows'] if progress.exists() else 0
                for left in range(start,len(new),8):
                    right=min(left+8,len(new));length=int(lengths[left:right].max())
                    ti=torch.as_tensor(ids[left:right,:length],dtype=torch.long,device='cuda')
                    tv=torch.as_tensor(vals[left:right,:length],device='cuda')
                    mask=torch.arange(length,device='cuda')[None,:]<torch.as_tensor(lengths[left:right],device='cuda')[:,None]
                    features=encoder(ti,tv,mask).cpu().numpy()
                    if not np.isfinite(features).all():raise ValueError('Nonfinite embeddings')
                    h[new[left:right]]=features
                    if right%200==0 or right==len(new):
                        h.flush();write_json(progress,dict(completed_rows=right))
                        print('ENCODER',split,right,'/',len(new),flush=True)
            h.flush();del h;partial.replace(dest)
            po.to_csv(CACHE/f'{split}_pbs_metadata.tsv',sep='\t')
        saved=pd.read_csv(CACHE/f'{split}_pbs_metadata.tsv',sep='\t',index_col=0)
        if not saved.index.equals(po.index):raise ValueError('Cache order mismatch')
        h=torch.from_numpy(np.load(dest))
        data[split]=(obs,y,pbs,h)
    del encoder;torch.cuda.empty_cache()
    write_json(CACHE/'complete.json',signature)
    return data,genes,signature


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise ValueError('Do not overwrite completed experiment')
    torch.set_num_threads(4)
    data,genes,signature=prepare()
    obs,y,pbs,h=data['train'];vo,vy,vpbs,vh=data['validation']
    if set(obs.index)&set(vo.index):raise ValueError('Split overlap')
    pools={g:dict(PBS=np.flatnonzero(obs.iloc[pbs].response_group.eq(g)),
        IL33=np.flatnonzero(obs.response_group.eq(g)&obs.condition.eq('IL33'))) for g in CFG['study_weights']}
    ypbs=y[pbs];vcontrol=vy[vpbs];truth=vy[np.flatnonzero(vo.condition.eq('IL33'))].mean(0);ctrl=vcontrol.mean(0)
    effect=sum(CFG['study_weights'][g]*(y[p['IL33']].mean(0)-ypbs[p['PBS']].mean(0)) for g,p in pools.items())
    c=torch.tensor([1.,0.]);results=[];predictions={}
    config=dict(batch_size=64,encoder=str(OFFICIAL_CHECKPOINT),seeds=[0,1,2],max_steps=5000,
        validation_every=500,patience=4,learning_rate=.001,study_weights=CFG['study_weights'],
        architecture='768 frozen embedding + 2 condition ->64 GELU->2003 residual; PBS expression skip and fixed mean effect',
        modes=['random_pair_cell_mse','group_mean_mse'],
        sampling='one background per step, 64 distinct treated cells; 64 controls independently with replacement',
        external_accessed=False,input_signature=signature)
    write_json(OUT/'config.json',config)
    @torch.no_grad()
    def evaluate(model,mode,ch,cy):
        if mode=='group_mean_mse':return model(ch.mean(0),cy.mean(0),c)
        return torch.cat([model(ch[i:i+256],cy[i:i+256],c) for i in range(0,len(ch),256)]).mean(0)
    for seed in config['seeds']:
        for mode in config['modes']:
            folder=OUT/f'{mode}_seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():
                r=json.loads((folder/'result.json').read_text());results.append(r)
                predictions[f'{mode}_seed{seed}']=np.load(folder/'D_prediction.npy');continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=GroupHead(h,torch.stack([effect,torch.zeros_like(effect)]),64,True)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            def save(step):torch.save(dict(model=model.state_dict(),step=step,config=config,gene_ids=genes),folder/'best.pt')
            initial=score(evaluate(model,mode,vh,vcontrol),truth,ctrl)
            best=initial['delta_rmse'];best_step=0;bad=0;save(0)
            with (folder/'history.jsonl').open('w',buffering=1) as log:
                for step in range(1,5001):
                    group=rng.choice(list(pools),p=list(CFG['study_weights'].values()));p=pools[group]
                    target=rng.choice(p['IL33'],64,replace=False);source=rng.choice(p['PBS'],64,replace=True)
                    if mode=='random_pair_cell_mse':pred=model(h[source],ypbs[source],c);target_y=y[target]
                    else:pred=model(h[source].mean(0),ypbs[source].mean(0),c);target_y=y[target].mean(0)
                    loss=(pred-target_y).square().mean()
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
                    if step%500==0:
                        prediction=evaluate(model,mode,vh,vcontrol);scores=score(prediction,truth,ctrl)
                        if scores['delta_rmse']<best:best=scores['delta_rmse'];best_step=step;bad=0;save(step)
                        else:bad+=1
                        log.write(json.dumps(dict(step=step,pair_or_group_mse=float(loss.detach()),validation=scores))+'\n')
                        print('VALIDATION',mode,seed,step,scores['delta_rmse'],'best',best_step,flush=True)
                        if bad>=4:break
            model.load_state_dict(torch.load(folder/'best.pt',weights_only=False)['model'])
            pred=evaluate(model,mode,vh,vcontrol)
            train_scores={g:score(evaluate(model,mode,h[p['PBS']],ypbs[p['PBS']]),y[p['IL33']].mean(0),ypbs[p['PBS']].mean(0)) for g,p in pools.items()}
            r=dict(mode=mode,seed=seed,best_step=best_step,stopped_step=step,initial=initial,
                validation=score(pred,truth,ctrl),train=train_scores,parameters=sum(p.numel() for p in model.parameters()))
            write_json(folder/'result.json',r);np.save(folder/'D_prediction.npy',pred.numpy())
            results.append(r);predictions[f'{mode}_seed{seed}']=pred.numpy()
    frame=pd.DataFrame([dict(mode=r['mode'],seed=r['seed'],best_step=r['best_step'],**r['validation']) for r in results])
    frame.to_csv(OUT/'comparison.tsv',sep='\t',index=False)
    export=pd.DataFrame(dict(gene_id=genes,control_mean=ctrl.numpy(),target_mean=truth.numpy(),**predictions))
    export.to_csv(OUT/'D_predictions.tsv.gz',sep='\t',index=False)
    write_json(OUT/'summary.json',dict(runs=results,external_accessed=False,
        caution='Random pairing does not create longitudinal labels. Pairwise MSE tends toward conditional means; no single-cell accuracy claim.'))
    print(frame.to_string(index=False),flush=True)
    report()


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
