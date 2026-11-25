"""Full real-data CPU acceptance of sampling/PAD/MLM; no model or updates.

From scratch/: python -u src/validate_input_pipeline.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from input_pipeline import build_loader, DEFAULT_CONFIG, ROOT

sys.path.insert(0,str(ROOT/"src"))
from posttrain_data import file_hash


def signature(batch):
    digest=hashlib.sha256()
    for key in ['source_rows','input_ids','values','target_mask','corruption_type','labels','value_labels']:
        digest.update(batch[key].numpy().tobytes())
    return digest.hexdigest()


def check_batch(batch,dataset,first,n_genes):
    x={k:v.numpy() for k,v in batch.items() if isinstance(v,torch.Tensor)}
    valid=x['attention_mask'];target=x['target_mask'];kind=x['corruption_type']
    np.testing.assert_array_equal(valid,np.arange(valid.shape[1])[None,:]<x['lengths'][:,None])
    assert valid.shape[1]==int(x['lengths'].max())
    assert not target[~valid].any() and not kind[~valid].any()
    assert (x['input_ids'][~valid]==0).all() and (x['values'][~valid]==0).all()
    assert (x['position_ids'][~valid]==0).all()
    assert (x['labels'][~target]==-100).all() and (x['target_gene_ids'][~target]==-100).all()
    assert (x['value_labels'][~target]==0).all()
    np.testing.assert_array_equal(x['target_counts'],target.sum(axis=1))
    np.testing.assert_array_equal(x['loss_cell_mask'],target.any(axis=1))
    np.testing.assert_array_equal(kind>0,target)
    np.testing.assert_array_equal(x['labels'][target]+first,x['target_gene_ids'][target])
    assert ((x['labels'][target]>=0)&(x['labels'][target]<n_genes)).all()
    assert (x['input_ids'][kind==1]==1).all()
    assert (x['values'][(kind==1)|(kind==2)]==-1).all()
    assert ((x['input_ids'][kind==2]>=first)&(x['input_ids'][kind==2]<first+n_genes)).all()
    for j,idx in enumerate(x['sample_indices']):
        sample=dataset[(int(idx),int(x['epochs'][j]))]
        n=len(sample['gene_ids']);t=target[j,:n];k=kind[j,:n]
        assert sample['cell_id']==batch['cell_ids'][j] and sample['source_row']==x['source_rows'][j]
        np.testing.assert_array_equal(x['position_ids'][j,:n],np.arange(n))
        np.testing.assert_array_equal(x['target_gene_ids'][j,:n][t],sample['gene_ids'][t])
        np.testing.assert_array_equal(x['value_labels'][j,:n][t],sample['values'][t])
        unchanged=(~t)|(k==3)
        np.testing.assert_array_equal(x['input_ids'][j,:n][unchanged],sample['gene_ids'][unchanged])
        np.testing.assert_array_equal(x['values'][j,:n][unchanged],sample['values'][unchanged])


def run_split(loader,first,n_genes,save_masks=False):
    n=len(loader.dataset);seen=np.zeros(n,dtype=np.int8)
    masked=np.zeros(n_genes,dtype=np.int64)
    records=[None]*n if save_masks else None
    totals=dict(cells=0,batches=0,tokens=0,padded_tokens=0,target_tokens=0,
                mask_tokens=0,random_tokens=0,unchanged_targets=0,random_same_as_original=0,
                empty_target_cells=0,mixed_shard_batches=0,attention_pair_units=0)
    fingerprints=[]
    for step,b in enumerate(loader):
        check_batch(b,loader.dataset,first,n_genes)
        ix=b['sample_indices'].numpy();seen[ix]+=1
        if (seen[ix]>1).any():raise ValueError('Duplicate cells in loader epoch')
        target=b['target_mask'].numpy();kind=b['corruption_type'].numpy()
        ids=b['input_ids'].numpy();gene=b['target_gene_ids'].numpy()
        labels=b['value_labels'].numpy()
        masked+=np.bincount(gene[target]-first,minlength=n_genes)
        totals['cells']+=len(ix);totals['batches']+=1
        totals['tokens']+=int(b['lengths'].sum())
        totals['padded_tokens']+=b['input_ids'].numel()
        totals['target_tokens']+=int(target.sum())
        totals['mask_tokens']+=int((kind==1).sum())
        totals['random_tokens']+=int((kind==2).sum())
        totals['unchanged_targets']+=int((kind==3).sum())
        totals['random_same_as_original']+=int(((kind==2)&(ids==gene)).sum())
        totals['empty_target_cells']+=int((b['target_counts']==0).sum())
        totals['mixed_shard_batches']+=int(loader.dataset.frame.iloc[ix].shard.nunique()>1)
        totals['attention_pair_units']+=len(ix)*ids.shape[1]**2
        if step<8:fingerprints.append(signature(b))
        if records is not None:
            for j,idx in enumerate(ix):
                positions=np.flatnonzero(target[j]).astype(np.int32)
                records[idx]=(positions,kind[j,positions].copy(),ids[j,positions].astype(np.int32),
                              gene[j,positions].astype(np.int32),labels[j,positions].copy())
        if step%2048==0:
            print(f"{loader.dataset.split}: {totals['cells']:,}/{n:,} cells checked",flush=True)
    assert (seen==1).all()
    totals['padding_fraction']=1-totals['tokens']/totals['padded_tokens']
    totals['target_fraction']=totals['target_tokens']/totals['tokens']
    totals['mask_fraction_among_targets']=totals['mask_tokens']/totals['target_tokens']
    totals['random_fraction_among_targets']=totals['random_tokens']/totals['target_tokens']
    totals['keep_fraction_among_targets']=totals['unchanged_targets']/totals['target_tokens']
    assert abs(totals['target_fraction']-.15)<.001
    for key,p in [('mask_fraction_among_targets',.8),('random_fraction_among_targets',.1),('keep_fraction_among_targets',.1)]:
        assert abs(totals[key]-p)<.005
    cache=None
    if records is not None:
        cache=dict(source_rows=loader.dataset.frame.source_row.to_numpy(dtype=np.int64),
                   offsets=np.r_[0,np.cumsum([len(r[0]) for r in records])].astype(np.int64))
        for j,key in enumerate(['positions','corruption_type','input_gene_ids','target_gene_ids','value_labels']):
            cache[key]=np.concatenate([r[j] for r in records])
    return totals,masked,cache,fingerprints


def mdtable(df):
    rows=[list(df.columns),['---']*len(df.columns)]
    rows += [[f'{v:.6g}' if isinstance(v,float) else str(v) for v in line]
             for line in df.itertuples(index=False,name=None)]
    return '\n'.join('| '+' | '.join(r)+' |' for r in rows)


def main():
    torch.set_num_threads(4)
    cfg=json.loads(DEFAULT_CONFIG.read_text())
    report_dir=ROOT/cfg['report'];fixed=ROOT/cfg['fixed_eval_masks']
    staging=fixed.with_name(fixed.name+'.partial')
    if fixed.exists() or staging.exists():raise FileExistsError('Fixed eval masks already exist; refusing overwrite')
    report_dir.mkdir(parents=True,exist_ok=True)
    tokenized=ROOT/cfg['tokenized'];prepared=tokenized.parent
    norm=json.loads((prepared/'config.json').read_text())
    vocab=pd.read_csv(prepared/'gene_vocab.tsv',sep='\t')
    n_genes=len(vocab);first=norm['first_gene_token_id']
    loader=build_loader('train',cfg)
    sampler=loader.batch_sampler
    orders=[]
    for epoch in (0,1):
        sampler.set_epoch(epoch)
        order=np.array([i for batch in sampler for i,e in batch])
        np.testing.assert_array_equal(np.sort(order),np.arange(len(loader.dataset)))
        orders.append(order)
    assert not np.array_equal(orders[0],orders[1])
    sampler.set_epoch(0)
    train,masked,_,fingerprints=run_split(loader,first,n_genes)
    train['split']='train'
    original_gene_events=vocab.training_positive_count.sum()
    assert train['tokens']==original_gene_events
    # Restore from actually consumed batch=3, independent of prefetched/dispatched work.
    state=sampler.state_dict(consumed_batches=3)
    resumed=build_loader('train',cfg)
    resumed.batch_sampler.load_state_dict(state)
    it=iter(resumed)
    for expected in fingerprints[3:8]:assert signature(next(it))==expected
    del it,resumed
    # Persistent workers must receive the new epoch in queued dataset keys.
    sampler.set_epoch(1)
    it=iter(loader);next_epoch=next(it)
    assert (next_epoch['epochs']==1).all()
    assert signature(next_epoch)!=fingerprints[0]
    del it,loader
    staging.mkdir(parents=True)
    summaries=[train];all_masked={'train':masked};fixed_hashes={}
    for split in ('validation','test'):
        eval_loader=build_loader(split,cfg)
        stat,counts,cache,_=run_split(eval_loader,first,n_genes,save_masks=True)
        stat['split']=split;summaries.append(stat);all_masked[split]=counts
        np.savez_compressed(staging/(split+'.npz'),**cache)
        with np.load(staging/(split+'.npz'),allow_pickle=False) as saved:
            for key in cache:np.testing.assert_array_equal(saved[key],cache[key])
        fixed_hashes[split+'.npz']=file_hash(staging/(split+'.npz'))
        if split=='validation':
            # Fixed cell-local masks must survive changed batch sizes/order and workers.
            repeat_cfg={**cfg,'num_workers':0,'batch_size':7,'bucket_size':280}
            repeat=build_loader(split,repeat_cfg);repeat.batch_sampler.set_epoch(9)
            _,_,repeated,_=run_split(repeat,first,n_genes,save_masks=True)
            for key in cache:np.testing.assert_array_equal(cache[key],repeated[key])
            del repeat,repeated
        del eval_loader
    stats=pd.DataFrame(summaries)
    stats.to_csv(report_dir/'batch_mask_statistics.tsv',sep='\t',index=False)
    genes=vocab[['gene_id','gene_symbol','token_id','training_positive_count','is_functional67']].copy()
    for split,counts in all_masked.items():genes[split+'_masked_targets']=counts
    genes['train_expected_targets_per_epoch']=norm['mask_probability']*genes.training_positive_count
    genes.to_csv(report_dir/'gene_mask_targets_epoch0.tsv',sep='\t',index=False)
    lengths=pd.read_csv(tokenized/'cell_manifest.tsv.gz',sep='\t').query("split=='train'").length.to_numpy()
    random=np.random.default_rng(cfg['sampler_seed']).permutation(len(lengths))
    random_padded=random_attention=0
    for start in range(0,len(random),cfg['batch_size']):
        ls=lengths[random[start:start+cfg['batch_size']]]
        random_padded+=len(ls)*int(ls.max());random_attention+=len(ls)*int(ls.max())**2
    padding=dict(random_padding_fraction=1-lengths.sum()/random_padded,
                 bucket_padding_fraction=train['padding_fraction'],
                 random_attention_pairs=random_attention,bucket_attention_pairs=train['attention_pair_units'],
                 bucket_to_random_attention_ratio=train['attention_pair_units']/random_attention)
    (report_dir/'padding_comparison.json').write_text(json.dumps(padding,indent=2)+'\n')
    provenance=dict(config=cfg,normalization_source='frozen preparation/config.json',
        epoch_definition='zero-based; train seed varies with epoch, eval ignores epoch',
        eval_masks='deterministically regenerated per cell; exported cache verifies exact targets/corruptions',
        CE_labels='labels are real-gene class IDs (token_id-first_gene_token_id); target_gene_ids preserve token IDs',
        branch_codes={'0':'non-target/PAD','1':'MASK','2':'random real gene','3':'unchanged target'},
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in [
            DEFAULT_CONFIG,prepared/'config.json',prepared/'gene_vocab.tsv',tokenized/'config.json',
            tokenized/'cell_manifest.tsv.gz',tokenized/'shard_manifest.tsv',
            Path(__file__).with_name('input_pipeline.py'),Path(__file__)]},
        fixed_mask_hashes=fixed_hashes,torch_version=torch.__version__,numpy_version=np.__version__,
        limitations=['data pipeline only; no actual Encoder/heads/loss/optimizer/CUDA benchmark',
                     'input and masking diagnostics on Test, no predictions or model selection',
                     'exact optimizer checkpoint/RNG restoration belongs to future trainer'])
    (staging/'config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    staging.rename(fixed)
    (report_dir/'input_pipeline_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    checks=dict(passed=True,all_real_cells_and_batches_checked=True,train_epoch0_and_1_sampler_exact_coverage=True,
        dynamic_pad_excluded_from_targets_and_labels=True,original_targets_and_nonmasked_context_preserved=True,
        label_class_offset_verified=True,corruption_rates_verified=True,random_ids_exclude_specials=True,
        multiworker_persistent_loader_epoch_change_verified=True,prefetch_safe_sampler_resume_verified=True,
        validation_masks_identical_workers_2_vs_0_and_batch_4_vs_7=True,
        fixed_validation_test_masks_saved_and_reloaded=True,
        actual_cells=int(stats.cells.sum()),actual_tokens=int(stats.tokens.sum()),num_workers=cfg['num_workers'])
    (report_dir/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    text=f'''# Scratch v0.3 随机分桶、动态 PAD 与 Joint MLM 输入流水线

已实现并在全部 **{int(stats.cells.sum()):,}细胞 / {int(stats.tokens.sum()):,}有效Token** 上完成CPU数据验收。没有运行预训练模型、反向传播或参数更新。

## 抽样与实际 PAD

Train每epoch先对全体细胞随机排列，再将每{cfg['bucket_size']}细胞组成一个pool，pool内按长度稳定排序分成batch={cfg['batch_size']}，最后全局打乱batch顺序。每细胞恰好一次，不丢弃尾batch；不按study或shard固定遍历，不做组织重采样。pool不是按固定长度边界删除细胞。
Validation/Test按长度确定性排序，遍历每细胞一次。

Train epoch0随机batch的PAD比例{padding['random_padding_fraction']:.3%}，随机长度分桶后为 **{padding['bucket_padding_fraction']:.3%}**。
分桶后的attention-pair代理为随机batch的{padding['bucket_to_random_attention_ratio']:.3%}；不是模型测速或正式训练加速比。

## 全量 Mask 统计

{mdtable(stats[['split','cells','batches','tokens','padded_tokens','padding_fraction','target_tokens','target_fraction','mask_fraction_among_targets','random_fraction_among_targets','keep_fraction_among_targets','empty_target_cells']])}

15%为独立Bernoulli目标率，不要求每细胞刚好15%。80/10/10按目标划分；随机ID只来自真实基因，可偶然抽回原ID，仍属random分支。MASK/random分支value=-1，keep分支保留原值；Mask之后不排序。
训练统计来自一次CPU输入遍历的真实Mask抽样，不是理论期望，也不是已完成模型训练。逐基因实际目标数见`gene_mask_targets_epoch0.tsv`，其中保留15%理论期望供对照。

## 接口约定

模型输入只有 `input_ids`(int64)、`values`(float32)、`attention_mask`(bool True=有效)、`position_ids`(int64)。PAD为0、value=0、position=0，不能仅看value判定有效性。
SDPA使用`attention_mask[:,None,None,:]`，True表示允许的key；若未来使用nn.Transformer的key_padding_mask，其语义相反，应传`~attention_mask`。

监督输出：

- `labels`：目标ID减2的真实基因分类下标0..26724；非目标/PAD=-100。与不含特殊符号的26725类ID头对应。
- `target_gene_ids`：原始目标Token ID，非目标=-100，便于解码/曝光统计；不得作为模型特征。
- `value_labels`：仅目标位置的原始归一化值，其他位置为0；必须用`target_mask`选择，不能直接全矩阵MSE。
- `target_mask`、`corruption_type`：目标和80/10/10分支；分支0非目标/PAD、1MASK、2random、3keep。
- `target_counts`、`loss_cell_mask`：Cell-macro每细胞分母和非空目标细胞。全空batch不强制补Mask，未来trainer须跳过更新。
- `lengths`、`source_rows`、`sample_indices`、`cell_ids`、`epochs`：对齐与恢复元数据，不能当作模型生物学输入。

## 随机性、workers和恢复

Train的每细胞污染由mask_seed、epoch、源行号、split、训练模式确定，不依赖batch顺序或worker RNG。每epoch换Mask，同一epoch重访同一细胞可精确复现。epoch由sampler的(index,epoch)键进入worker，避免persistent workers拿到旧epoch。
Eval固定使用epoch=0，在不同batch组合、batch size和worker数下同一细胞得到同样目标与替换。缓存位于`{cfg['fixed_eval_masks']}`的validation/test.npz；缓存保存source_rows、offsets、目标positions、分支、污染ID、原始目标ID/表达。运行时按同一确定性算法再生，并已验证与导出缓存一致。

```python
import sys
sys.path.insert(0, "src")
from input_pipeline import build_loader
train_loader = build_loader("train")
train_loader.batch_sampler.set_epoch(0)
for batch_index, batch in enumerate(train_loader):
    # future model: only input_ids, values, attention_mask, position_ids
    # future trainer: normalize Cell-macro across consumed valid cells
    pass
state = train_loader.batch_sampler.state_dict(consumed_batches=batch_index + 1)
# future checkpoint also needs model/optimizer/scheduler/RNG and frozen-data hashes
restored = build_loader("train")
restored.batch_sampler.load_state_dict(state)
```

`consumed_batches`是当前迭代器起点后实际消费的batch数，不能用DataLoader预取/派发数量。恢复sampler已验证；这不等于尚未实现的optimizer/checkpoint恢复已通过。

## 验收范围与接下来的关口

已通过：Train两个epoch抽样覆盖及顺序变化；全部分区逐细胞输入/标签/分支校验；每个batch动态PAD；2个persistent workers的epoch传递；真实前缀batch恢复；Validation跨2/0 workers、4/7 batch size固定Mask一致；固定Eval缓存保存重读。
单元测试另覆盖空目标、随机替换范围、源数组不被污染、SDPA布尔mask屏蔽PAD key。

**训练所需的数据准备与输入层已就绪。正式训练启动前仍需完成E1b Encoder/双头/Cell-macro损失、梯度累计、模型checkpoint以及真实最大长度的完整forward/backward/optimizer benchmark。** 本阶段没有实现E4 Query、没有检查模型本体PAD不变性或训练显存，不能将数据验收称为整个预训练系统已验收。

复现：从scratch运行 `/root/.conda/envs/ilc2_model/bin/python -u src/validate_input_pipeline.py`。固定Eval缓存已存在时拒绝覆盖。
'''
    (report_dir/'report.md').write_text(text)
    print(json.dumps(checks,indent=2),flush=True)
    print(json.dumps(padding,indent=2),flush=True)


if __name__=='__main__':main()
