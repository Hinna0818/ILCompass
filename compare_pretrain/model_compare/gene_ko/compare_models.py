"""Matched Piezo1 interventions: raw GeneCompass / LoRA / Scratch v3.

Reuse the completed LoRA intervention roster, random genes and cosine scorer.
"""
import sys,argparse,shutil,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared import *
from posttrain_data import encode as encode_counts
import anndata as ad
spec=importlib.util.spec_from_file_location('piezo1_original_protocol',Path(__file__).with_name('run.py'))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
BASE_CFG=json.loads(Path(__file__).with_name('config.json').read_text())
DEST=Path(__file__).parent/'results/GSE279338_Piezo1_model_comparison'

def paths(smoke):
    base=ROOT/BASE_CFG['output'];out=DEST
    if smoke:base=base.with_name(base.name+'_smoke');out=out.with_name(out.name+'_smoke')
    return base,out

def prepare(smoke):
    base,out=paths(smoke);common=out/'common';common.mkdir(parents=True,exist_ok=True)
    if (common/'preparation.json').exists():raise FileExistsError('Common inputs already fixed; reuse extract and score stages')
    original=legacy.check(BASE_CFG,base)
    manifest=pd.read_csv(base/'embedding_manifest.csv.gz').rename(columns={'sequence_row':'original_sequence_row'})
    coverage=pd.read_csv(base/'coverage_cells.csv.gz')
    a=ad.read_h5ad(ROOT/BASE_CFG['source']);a=a[manifest.cell_id.drop_duplicates()].copy()
    res=resources();vocab,med=res[2:];vocab_map=dict(zip(vocab.gene_id,vocab.token_id))
    target=vocab_map[BASE_CFG['target']]
    seq=sequences(a,'scratch_v3',res);lookup=dict(zip(a.obs_names,seq));audit=[]
    for cell,rows in manifest.groupby('cell_id',sort=False):
        if rows.group.iloc[0]!='WT':continue
        ii,_=lookup[cell];where=np.flatnonzero(ii==target);rank=int(where[0]+1) if len(where) else 0
        perts=set(rows.perturbed_gene.dropna())-{''}
        missing=sorted(g for g in perts if g not in vocab_map or vocab_map[g] not in ii)
        audit.append(dict(cell_id=cell,scratch_rank=rank,scratch_length=len(ii),scratch_top_half=0<rank<=len(ii)//2,missing_control_genes=';'.join(missing),all_operators_supported=not missing))
    audit=pd.DataFrame(audit).set_index('cell_id');audit.to_csv(common/'scratch_eligibility.csv')
    eligible=set(audit.index[audit.scratch_top_half & audit.all_operators_supported])
    # Exact same complete-case cohort for all three models; keep original coverage separately.
    coverage.to_csv(common/'original_GeneCompass_coverage.csv',index=False)
    for flag in ['top_half_valid','top_half_full']:coverage[flag]=coverage[flag] & coverage.cell_id.isin(eligible)
    coverage.to_csv(common/'coverage_cells.csv.gz',index=False)
    manifest=manifest[manifest.group.eq('KO') | manifest.cell_id.isin(eligible)].copy().reset_index(drop=True)
    manifest.insert(0,'sequence_row',np.arange(len(manifest)))
    if not coverage.top_half_valid.any():raise ValueError('No common top-half WT')
    manifest.to_csv(common/'embedding_manifest.csv.gz',index=False)
    a.obs.to_csv(common/'source_metadata.tsv.gz',sep='\t',index_label='cell_id')
    # Reuse the shared normalization/ranking implementation, fixing original library size
    # so intervention changes the chosen gene only, not every gene via CP10K renormalization.
    order=np.argsort(a.var_names.to_numpy(),kind='stable');genes=a.var_names[order];x=a.X[:,order].tocsr();x.sort_indices()
    columns=vocab.gene_id.map(pd.Series(np.arange(len(genes)),index=genes)).to_numpy();present=~pd.isna(columns);cols=columns[present].astype(int)
    token_col=np.full(len(genes),-1,dtype='int32');token_col[cols]=vocab.token_id.to_numpy()[present]
    scales=np.ones(len(genes));scales[cols]=med.median_positive_cp10k.to_numpy()[present]
    totals=np.asarray(x.sum(1)).ravel();allids=[];allvalues=[];offsets=[0]
    for n,row in manifest.iterrows():
        ii,vv=lookup[row.cell_id]
        if row.operator not in ['before','real_KO',BASE_CFG['symbol']+'_1']:
            r=a.obs_names.get_loc(row.cell_id);changed=x[r].astype(float).copy()
            column=genes.get_loc(row.perturbed_gene);positions=np.flatnonzero(changed.indices==column)
            if len(positions)!=1:raise ValueError('Perturbed gene missing in counts')
            fraction=0. if row.operator.startswith('random') else float(row.operator.split('_')[-1])
            changed.data[positions]*=fraction;changed.eliminate_zeros();changed.data*=10000/totals[r]
            ids,values,length=encode_counts(changed,token_col,scales,8192,'median');ii=ids[0,:length[0]];vv=values[0,:length[0]]
            if fraction==0 and vocab_map[row.perturbed_gene] in ii:raise ValueError('Deleted gene retained')
        allids.extend(ii);allvalues.extend(vv);offsets.append(len(allids))
    for name,val in [('gene_ids',np.asarray(allids,dtype='int32')),('values',np.asarray(allvalues,dtype='float32')),('offsets',np.asarray(offsets,dtype='int64'))]:np.save(common/f'scratch_{name}.npy',val)
    write_json(common/'preparation.json',dict(source=original,base_directory=str(base),manifest_sha256=file_hash(common/'embedding_manifest.csv.gz'),common_WT_primary=int(coverage.top_half_valid.sum()),common_WT_full=int(coverage.top_half_full.sum()),KO=int(manifest.group.eq('KO').sum()),scratch_normalization='Native Train-only CP10K median; intervention retains original library denominator; re-rank all nonzero',selection='GC top-half AND scratch native top-half; all random control genes must be present in scratch',smoke=smoke))
    print(json.loads((common/'preparation.json').read_text()))

@torch.inference_mode()
def extract(name,smoke):
    base,out=paths(smoke);common=out/'common';fixed=json.loads((common/'preparation.json').read_text())
    legacy.check(BASE_CFG,base)
    manifest=pd.read_csv(common/'embedding_manifest.csv.gz')
    if file_hash(common/'embedding_manifest.csv.gz')!=fixed['manifest_sha256']:raise ValueError('Manifest changed')
    model=load(name);source=common if name=='scratch_v3' else base;prefix='scratch_' if name=='scratch_v3' else ''
    ids,values,offsets=[np.load(source/f'{prefix}{key}.npy',mmap_mode='r') for key in ['gene_ids','values','offsets']]
    source_rows=manifest.sequence_row.to_numpy() if name=='scratch_v3' else manifest.original_sequence_row.to_numpy()
    lengths=np.diff(offsets)[source_rows];order=np.argsort(lengths,kind='stable')
    pooled={'mean':np.empty((len(manifest),model.config.hidden_size),dtype='float32')}
    if name!='scratch_v3':pooled['CLS']=np.empty_like(pooled['mean'])
    for start in range(0,len(order),4):
        rows=order[start:start+4];seq=[(ids[offsets[r]:offsets[r+1]],values[offsets[r]:offsets[r+1]]) for r in source_rows[rows]]
        b=batch(seq);h=encode(model,b,name)
        if name!='scratch_v3':pooled['CLS'][rows]=h[:,0].cpu().numpy();h=h[:,1:]
        pooled['mean'][rows]=((h*b['attention_mask'][...,None]).sum(1)/b['attention_mask'].sum(1)[:,None]).cpu().numpy()
        if start%512==0:print(name,start,len(order),flush=True)
    for pooling,h in pooled.items():
        if not np.isfinite(h).all() or (np.linalg.norm(h,axis=1)==0).any():raise ValueError('Invalid embeddings')
        folder=out/name/pooling;folder.mkdir(parents=True,exist_ok=True)
        cfg={**BASE_CFG,'model':name,'output':str(folder.relative_to(ROOT))}
        prep={**fixed['source'],'config':cfg,'comparison':fixed}
        for filename in ['embedding_manifest.csv.gz','coverage_cells.csv.gz','source_metadata.tsv.gz']:shutil.copyfile(common/filename,folder/filename)
        before=manifest[manifest.operator.eq('before')].set_index('cell_id').sequence_row
        sham=manifest[manifest.operator.eq(BASE_CFG['symbol']+'_1')].set_index('cell_id').loc[before.index,'sequence_row']
        error=float(np.abs(h[before.to_numpy()]-h[sham.to_numpy()]).max())
        if error>1e-4:raise ValueError('Sham changed embedding')
        np.save(folder/'cell_embeddings.npy',h)
        write_json(folder/'preparation.json',prep)
        write_json(folder/'inference.json',dict(**provenance(name),preparation=prep,pooling=pooling,sham_max_abs=error))
        for op,label in [('before','WT_before'),(BASE_CFG['symbol']+'_0','WT_insilico_KO'),('real_KO','real_KO')]:
            sub=manifest[manifest.operator.eq(op)];np.save(folder/f'{label}_embeddings.npy',h[sub.sequence_row]);sub.to_csv(folder/f'{label}_cells.csv',index=False)

def score(smoke):
    _,out=paths(smoke);summaries=[];comparisons=[];reference=None
    for name in ['raw','lora','scratch_v3']:
        for pooling in (['mean'] if name=='scratch_v3' else ['CLS','mean']):
            folder=out/name/pooling;cfg=json.loads((folder/'preparation.json').read_text())['config'];legacy.score(cfg,folder,embedding_filename="cell_embeddings.npy")
            # The shared scorer's historical prose is CLS-specific; use the comparison report.
            (folder/'report.md').write_text(f'Model: {name}; actual pooling: {pooling}. See ../../report.md for common design and limitations.\n')
            d=pd.read_csv(folder/'cosine_summary.csv');d.insert(0,'model',name);d.insert(1,'pooling',pooling);summaries.append(d)
            c=pd.read_csv(folder/'KO_vs_random_summary.csv');c['model']=name;c['pooling']=pooling;comparisons.append(c)
            pairs=pd.read_csv(folder/'paired_cosine.csv.gz',usecols=['scope','seed','cell_id','KO_cell_id','operator'])
            if reference is None:reference=pairs
            else:pd.testing.assert_frame_equal(reference,pairs)
    pd.concat(summaries).to_csv(out/'comparison.csv',index=False);pd.concat(comparisons).to_csv(out/'random_control_comparison.csv',index=False)
    (out/'report.md').write_text('# Piezo1共同队列模型对照\n\n所有模型共享WT/真实KO/随机基因、配对seed。mean pooling是跨模型主比较；GC原始与LoRA的CLS为论文式对照。Scratch无CLS，保留其原生全非零输入与训练中位数，故不是完全相同架构/上下文。KO在原生线性归一化表达上缩放并重排，Scratch固定原始CP10K分母。\n\n共同WT需同时满足两种编码的前50%与随机对照可见性，因此数量可能少于旧740细胞；不能把新结果与旧全队列差异归因为模型。正delta仅为表示接近；单文库/基因型混杂，随机配对seed不是生物学重复。\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','extract','score']);p.add_argument('--model',choices=['raw','lora','scratch_v3']);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.stage=='prepare':prepare(args.smoke)
    elif args.stage=='extract':
        if not args.model:p.error('--model required')
        extract(args.model,args.smoke)
    else:score(args.smoke)
