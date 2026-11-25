"""C+D+GSE102299 training, fixed 15 epochs; final GSE189285 evaluation only."""
from train_il33_random_pairs import prepare, DATA, CACHE
from train_il33_baselines import score
from train_group_head import GroupHead
from models import ROOT, FrozenILC2Encoder, OFFICIAL_CHECKPOINT
from prepare import write_json
from posttrain_data import encode, file_hash
from genecompass_coverage import map_genes
import argparse
import json
import pickle
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

OUT=ROOT/'results/IL33_drugpred_fixed15_v1'
SPLIT=DATA.parent/'IL33_C_D_train_v1'
GROUPS=['GSE136154_C','GSE136154_D','GSE102299_rep1','GSE102299_rep2']
CONFIG=dict(epochs=15,seeds=[0,1,2],batch_size=64,learning_rate=.001,weight_decay=.01,
    hidden_size=64,gradient_clip=10,group_weights={g:.25 for g in GROUPS},
    architecture='frozen 768 ILC2 + 2 condition ->64 GELU->G residual; PBS mean + mean effect + residual, clamp>=0',
    checkpoint_selection='final epoch15 only; no validation, early stopping, or seed selection',
    epoch_definition='each training IL33 cell exactly once, shuffled within background; matched PBS drawn with replacement per batch',
    loss='group mean log1p(CP10K) MSE; batch weights account for group size and final partial batch',
    output_panel='unchanged baseline_v1 train-selected 2003 genes',
    encoder=str(OFFICIAL_CHECKPOINT),external_test='GSE189285 lung PBS -> IL33')


def train():
    OUT.mkdir(parents=True,exist_ok=True);SPLIT.mkdir(exist_ok=True)
    if (OUT/'training_complete.json').exists():raise ValueError('Training already complete; use --evaluate')
    write_json(OUT/'config.json',CONFIG)
    data,genes,signature=prepare()
    # Old filenames stay unchanged for provenance; both source roles now contribute to training.
    ys=[];hs=[];metadata=[];pmeta=[];ycontrol=[]
    for source in ['train','validation']:
        obs,y,pbs,h=data[source]
        metadata.append(obs.copy());pmeta.append(obs.iloc[pbs].copy());ys.append(y);hs.append(h);ycontrol.append(y[pbs])
    obs=pd.concat(metadata);po=pd.concat(pmeta);y=torch.cat(ys);h=torch.cat(hs);ypbs=torch.cat(ycontrol)
    if not obs.index.is_unique or not obs.condition.isin(['PBS','IL33']).all():raise ValueError('Unexpected training cells')
    obs['previous_split']=obs['split'].astype(str);obs['split']='train';obs['allowed_for_fit']=True;obs['allowed_for_tuning']=False
    obs.to_csv(SPLIT/'train_metadata.tsv.gz',sep='\t')
    groups={g:dict(PBS=np.flatnonzero(po.response_group.eq(g)),IL33=np.flatnonzero(obs.response_group.eq(g)&obs.condition.eq('IL33'))) for g in GROUPS}
    if any(not len(p[k]) for p in groups.values() for k in ['PBS','IL33']):raise ValueError('Missing condition')
    effects={g:y[p['IL33']].mean(0)-ypbs[p['PBS']].mean(0) for g,p in groups.items()}
    effect=sum(effects.values())/4
    panel=pd.read_csv(ROOT/'results/IL33_drugpred_baseline_v1/gene_panel.tsv',sep='\t')
    if panel.gene_id.tolist()!=genes:raise ValueError('Changed output panel')
    panel.to_csv(OUT/'gene_panel.tsv',sep='\t',index=False)
    manifest=dict(version='IL33_C_D_train_v1',train_sources=signature['inputs'],train_cells=len(obs),
        condition_counts=obs.condition.value_counts().to_dict(),group_counts=obs.groupby(['response_group','condition'],observed=True).size().rename('cells').reset_index().to_dict('records'),
        external_input=str(DATA/'external_test/input/external_input.h5ad'),
        external_target=str(DATA/'external_test/sealed/external_target_sealed.h5ad'),
        no_validation=True,roles='Both former development roles are training; external target only for final scoring',
        panel_genes=len(genes),encoder_signature=signature)
    write_json(SPLIT/'manifest.json',manifest);write_json(OUT/'data_manifest.json',manifest)
    obs.groupby(['study','response_group','condition'],observed=True).size().rename('cells').to_csv(SPLIT/'library_counts.tsv',sep='\t')
    weights=CONFIG['group_weights'];c=torch.tensor([1.,0.]);nsteps=sum((len(p['IL33'])+63)//64 for p in groups.values())
    # This constant integrates to one equally weighted group objective over an epoch.
    history=[]
    for seed in CONFIG['seeds']:
        folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
        if (folder/'final.pt').exists():raise ValueError('Existing seed checkpoint; use new experiment directory')
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=GroupHead(h,torch.stack([effect,torch.zeros_like(effect)]),64,True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        with (folder/'history.jsonl').open('w',buffering=1) as log:
            for epoch in range(1,16):
                batches=[]
                for g,p in groups.items():
                    ids=rng.permutation(p['IL33'])
                    batches.extend((g,ids[i:i+64]) for i in range(0,len(ids),64))
                visited=np.concatenate([ids for _,ids in batches])
                if len(np.unique(visited))!=len(visited) or len(visited)!=int(obs.condition.eq('IL33').sum()):
                    raise ValueError('Epoch does not visit each target exactly once')
                total_loss=0.
                for j in rng.permutation(len(batches)):
                    g,target=batches[j];source=rng.choice(groups[g]['PBS'],len(target),replace=True)
                    prediction=model(h[source].mean(0),ypbs[source].mean(0),c)
                    mse=(prediction-y[target].mean(0)).square().mean()
                    factor=weights[g]*len(target)/len(groups[g]['IL33'])
                    loss=mse*(factor*nsteps)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
                    total_loss+=float(mse.detach())*factor
                with torch.no_grad():
                    scores={g:score(model(h[p['PBS']].mean(0),ypbs[p['PBS']].mean(0),c),
                        y[p['IL33']].mean(0),ypbs[p['PBS']].mean(0)) for g,p in groups.items()}
                row=dict(seed=seed,epoch=epoch,steps=epoch*nsteps,weighted_minibatch_mse=total_loss,
                    train=scores,train_macro_rmse=sum(v['delta_rmse'] for v in scores.values())/4)
                history.append(row);log.write(json.dumps(row)+'\n')
                # Latest is recovery state, never a checkpoint selected by metric.
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,seed=seed,
                    gene_ids=genes,config=CONFIG,signature=signature),folder/'latest.pt')
                print('TRAIN',seed,'epoch',epoch,'/',15,'macro_rmse',row['train_macro_rmse'],flush=True)
        (folder/'latest.pt').replace(folder/'final.pt')
    pd.DataFrame([{k:v for k,v in row.items() if k!='train'} for row in history]).to_csv(OUT/'train_history.tsv',sep='\t',index=False)
    np.savez_compressed(OUT/'training_effects.npz',gene_ids=np.array(genes),**{g:v.numpy() for g,v in effects.items()})
    write_json(OUT/'training_complete.json',dict(epochs=15,steps_per_epoch=nsteps,seeds=CONFIG['seeds'],
        final_checkpoint_sha256={str(seed):file_hash(OUT/f'seed{seed}/final.pt') for seed in CONFIG['seeds']},
        test_opened_during_training=False))


def external_embedding(a, cache=None, input_path=None):
    cache=SPLIT/'external_PBS_cache' if cache is None else cache
    cache.mkdir(parents=True,exist_ok=True)
    input_path=DATA/'external_test/input/external_input.h5ad' if input_path is None else input_path
    cfg=json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    signature=dict(input=file_hash(input_path),
        encoder=file_hash(OFFICIAL_CHECKPOINT/'pytorch_model.bin'),tokens=file_hash(ROOT/cfg['token_dictionary']),
        medians=file_hash(ROOT/cfg['gene_median_reference']),encoding='median',length=2048,precision='float32')
    if (cache/'signature.json').exists() and json.loads((cache/'signature.json').read_text())!=signature:raise ValueError('External cache changed')
    write_json(cache/'signature.json',signature)
    dest=cache/'pbs_h.npy'
    if dest.exists():
        meta=pd.read_csv(cache/'metadata.tsv.gz',sep='\t',index_col=0)
        if not meta.index.equals(a.obs_names):raise ValueError('PBS embedding order mismatch')
        return torch.from_numpy(np.load(dest))
    if not torch.cuda.is_available():raise RuntimeError('GPU required to encode external PBS')
    tokens=pickle.loads((ROOT/cfg['token_dictionary']).read_bytes());median=pickle.loads((ROOT/cfg['gene_median_reference']).read_bytes())
    col,med,_=map_genes(a.var_names,tokens,median)
    ids,vals,lengths=encode(a.X,col,med,2048,'median')
    torch.backends.cuda.matmul.allow_tf32=False;encoder=FrozenILC2Encoder().to('cuda').eval()
    partial=cache/'pbs_h.partial.npy';progress=cache/'progress.json'
    start=json.loads(progress.read_text())['rows'] if progress.exists() else 0
    h=np.lib.format.open_memmap(partial,mode='r+' if start else 'w+',dtype='float32',shape=(a.n_obs,768))
    for left in range(start,a.n_obs,8):
        right=min(left+8,a.n_obs);length=int(lengths[left:right].max())
        ti=torch.as_tensor(ids[left:right,:length],dtype=torch.long,device='cuda');tv=torch.as_tensor(vals[left:right,:length],device='cuda')
        mask=torch.arange(length,device='cuda')[None,:]<torch.as_tensor(lengths[left:right],device='cuda')[:,None]
        features=encoder(ti,tv,mask).cpu().numpy()
        if not np.isfinite(features).all():raise ValueError('Nonfinite embedding')
        h[left:right]=features
        if right%200==0 or right==a.n_obs:
            h.flush();write_json(progress,dict(rows=right));print('EXTERNAL PBS ENCODE',right,'/',a.n_obs,flush=True)
    h.flush();del h,encoder;partial.replace(dest);torch.cuda.empty_cache()
    a.obs.to_csv(cache/'metadata.tsv.gz',sep='\t')
    return torch.from_numpy(np.load(dest))


def mean_expression(a,genes):
    if not np.issubdtype(a.X.dtype,np.integer) or (a.X!=a.layers['counts']).nnz:raise ValueError('Invalid external counts')
    x=a.X.astype('float32');x=x.multiply((1e4/np.asarray(x.sum(1)).ravel())[:,None]).tocsr();x.data=np.log1p(x.data)
    values=np.full(len(genes),np.nan,dtype='float32');locations=a.var_names.get_indexer(genes);valid=locations>=0
    values[valid]=np.asarray(x[:,locations[valid]].mean(0)).ravel()
    return torch.from_numpy(values)


def evaluate():
    if (OUT/'test_results.json').exists():raise ValueError('Final external test already completed')
    stamp=json.loads((OUT/'training_complete.json').read_text())
    for seed in CONFIG['seeds']:
        if file_hash(OUT/f'seed{seed}/final.pt')!=stamp['final_checkpoint_sha256'][str(seed)]:raise ValueError('Checkpoint modified before test')
    pbs=ad.read_h5ad(DATA/'external_test/input/external_input.h5ad')
    if not pbs.obs.condition.eq('PBS').all() or not pbs.obs.study.eq('GSE189285').all():raise ValueError('Wrong external input')
    h=external_embedding(pbs)
    genes=pd.read_csv(OUT/'gene_panel.tsv',sep='\t').gene_id.tolist()
    ctrl=mean_expression(pbs,genes);predictions={};c=torch.tensor([1.,0.])
    for seed in CONFIG['seeds']:
        saved=torch.load(OUT/f'seed{seed}/final.pt',weights_only=False)
        if saved['epoch']!=15 or saved['gene_ids']!=genes:raise ValueError('Not fixed final checkpoint')
        model=GroupHead(torch.zeros(2,768),torch.zeros(2,len(genes)),64,True)
        model.load_state_dict(saved['model']);model.eval()
        with torch.no_grad():predictions[f'ILC2_seed{seed}']=model(h.mean(0),ctrl,c)
    # Model predictions are fixed before opening the treated test file.
    target=ad.read_h5ad(DATA/'external_test/sealed/external_target_sealed.h5ad')
    if not target.obs.condition.eq('IL33').all() or not target.obs.study.eq('GSE189285').all():raise ValueError('Wrong test target')
    training_ids=pd.read_csv(SPLIT/'train_metadata.tsv.gz',sep='\t',usecols=[0],index_col=0).index
    if len(training_ids.intersection(pbs.obs_names)) or len(training_ids.intersection(target.obs_names)):raise ValueError('Test/train overlap')
    truth=mean_expression(target,genes);measured=torch.isfinite(ctrl)&torch.isfinite(truth)
    effects=np.load(OUT/'training_effects.npz')
    avg=sum(torch.from_numpy(effects[g]) for g in GROUPS)/4
    predictions['control_copy']=ctrl
    predictions['train_balanced_mean_effect']=(ctrl+avg).clamp_min(0)
    predictions['GSE136154_mean_effect']=(ctrl+(torch.from_numpy(effects[GROUPS[0]])+torch.from_numpy(effects[GROUPS[1]]))/2).clamp_min(0)
    predictions['GSE102299_mean_effect']=(ctrl+(torch.from_numpy(effects[GROUPS[2]])+torch.from_numpy(effects[GROUPS[3]]))/2).clamp_min(0)
    rows=[]
    for name,pred in predictions.items():
        if not torch.isfinite(pred[measured]).all() or (pred[measured]<0).any():raise ValueError('Invalid test prediction')
        rows.append(dict(model=name,**score(pred[measured],truth[measured],ctrl[measured])))
    table=pd.DataFrame(rows);table.to_csv(OUT/'test_comparison.tsv',sep='\t',index=False)
    panel=pd.read_csv(OUT/'gene_panel.tsv',sep='\t');frame=panel[['gene_id','gene_symbol']].copy()
    frame['measured_in_both_test_conditions']=measured.numpy();frame['PBS_mean']=ctrl.numpy();frame['IL33_mean']=truth.numpy()
    frame['observed_delta']=(truth-ctrl).numpy()
    for name,pred in predictions.items():frame[name+'_delta']=(pred-ctrl).numpy()
    frame.to_csv(OUT/'test_gene_predictions.tsv.gz',sep='\t',index=False)
    frame.loc[frame.measured_in_both_test_conditions].assign(abs_change=lambda x:x.observed_delta.abs()).nlargest(20,'abs_change').to_csv(OUT/'test_observed_top20.tsv',sep='\t',index=False)
    frame.loc[frame.gene_symbol.isin(['Il5','Il13','Areg','Gata3','Rora','Il1rl1','Il7r','Il2ra'])].to_csv(OUT/'functional_genes.tsv',sep='\t',index=False)
    write_json(OUT/'test_results.json',dict(results=rows,test_PBS_cells=pbs.n_obs,test_IL33_cells=target.n_obs,
        panel_genes=len(genes),evaluated_genes=int(measured.sum()),missing_genes=frame.loc[~frame.measured_in_both_test_conditions,'gene_id'].tolist(),
        selection='all three prespecified seed epoch15 checkpoints, no test-based checkpoint selection',
        limits='one pooled test library per condition; descriptive gene ranking, no replicated DE significance',
        checkpoint_hashes=stamp['final_checkpoint_sha256']))
    print(table.to_string(index=False),flush=True)
    report()


def report():
    summary=json.loads((OUT/'test_results.json').read_text());table=pd.read_csv(OUT/'test_comparison.tsv',sep='\t')
    model=table[table.model.str.startswith('ILC2_seed')];means=model.mean(numeric_only=True);sd=model.std(numeric_only=True,ddof=1)
    lines=['# 合并C/D训练15 epoch：GSE189285外部测试','',
        '训练为GSE136154 C+D和GSE102299两个重复的PBS/IL33；不再有开发验证，不纳入CGRP或双刺激。'
        '三个预设seed均固定训练15 epoch，使用最终权重；训练全过程未读取外部目标。','',
        '| 模型 | 外部ΔRMSE↓ | ΔPearson↑ | ΔSpearman↑ | MAE↓ | Top20命中 |','|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples():
        fmt=lambda x:'—' if pd.isna(x) else f'{x:.4f}'
        lines.append(f'| {r.model} | {r.delta_rmse:.6f} | {fmt(r.delta_pearson)} | {fmt(r.delta_spearman)} | {r.mae:.6f} | {fmt(r.top20_overlap)} |')
    lines+=['',f'ILC2三seed ΔRMSE={means.delta_rmse:.6f}±{sd.delta_rmse:.6f}；Pearson={means.delta_pearson:.4f}。'
        'SD为训练随机性，不是生物学置信区间；没有根据外部结果挑选seed。',
        '本轮外部结果未超过PBS原样复制和训练平均效应基线，显示当前15-epoch模型未实现有效的跨研究平均响应迁移。'
        '训练内拟合好不等同外部泛化好；品系、处理途径/时长及群体组成差异均可能参与，当前实验不能单独确定原因。',
        f"外部PBS {summary['test_PBS_cells']}、IL33 {summary['test_IL33_cells']}；固定面板 {summary['panel_genes']}，"
        f"共同测量 {summary['evaluated_genes']} 基因；缺失feature保持NaN并排除评分，未补零。",
        '模型沿用冻结0.8–0.2 ILC2、官方median输入编码、768维表示、770→64 GELU→G残差头。'
        '输出clamp(PBS均值+四背景等权平均效应+MLP修正,0)。标准化与平均效应重新用C+D+GSE102299训练池计算。',
        '每epoch把每个训练IL33细胞使用一次，文库内打乱、每批最多64；从相同背景PBS有放回抽等量细胞。'
        '群体均值MSE，按组样本量及尾批校正权重，使四背景每epoch各占1/4；两个研究各占1/2。'
        'PBS不会保证每epoch恰好遍历一次。AdamW lr0.001、weight_decay0.01，无早停、无最佳checkpoint选择。',
        '本次外部测试包含实验方案/品系变化；此前D已并入训练，其原来的验证分数不能当本模型泛化结果。'
        '外部单条件一个混池文库，Top20是观测平均变化而非显著DEG。此次已解封评分；若按结果改模型，'
        'GSE189285后续应称反复使用的外部评估集，不能继续称完全未见测试。',
        '复现：python src/downstream/drugpred/train_il33_fixed15.py --train；随后 --evaluate（首次编码外部PBS需要GPU）。'
        '输出包括固定配置、15轮训练日志、各seed最终checkpoint、外部逐基因预测和来源清单。']
    (OUT/'report.md').write_text('\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true');parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.train==args.evaluate:parser.error('Choose exactly one phase')
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        train() if args.train else evaluate()
