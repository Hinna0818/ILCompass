"""Paper-style quantitative KO, frozen LoRA GeneCompass CLS, random KO references."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared import *
from qc_utils import _raw_counts
import anndata as ad
CONFIG=Path(__file__).with_name('config.json')

def prepare(cfg,out,smoke):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists():raise FileExistsError('Prepared run exists: use extract/score to reuse it')
    a=ad.read_h5ad(ROOT/cfg['source'])
    for column,value in cfg.get('obs_filters',{}).items():
        a=a[a.obs[column].astype(str).eq(str(value))].copy()
    x=_raw_counts(a.layers['counts'])
    if (x!=a.X).nnz or not a.obs_names.is_unique or not a.var_names.is_unique or not a.obs.qc_version.eq('ilc2_balanced_v1').all() or not a.obs.selected_ILC2.all():raise ValueError('Invalid counts/QC')
    labels=a.obs[cfg.get('group_column','condition')]
    wt_label=cfg.get('wt_condition','control');ko_label=cfg.get('ko_condition','Piezo1_cKO')
    if not a.obs.tissue.eq('Lung').all() or set(labels)!={wt_label,ko_label}:raise ValueError('Unexpected population')
    if a.var.loc[cfg['target'],'gene_symbol']!=cfg['symbol']:raise ValueError('Target identity mismatch')
    if set(a.obs[cfg.get('treatment_column','treatment')])!={cfg.get('treatment','IL33')}:raise ValueError('Unexpected treatment metadata')
    if smoke:
        # Only an engineering check; do not treat this expression-enriched subset as an experiment.
        count=np.asarray(x[:,a.var_names.get_loc(cfg['target'])].toarray()).ravel()
        wt=np.flatnonzero(labels.eq(wt_label));ko=np.flatnonzero(labels.eq(ko_label))
        rows=np.r_[wt[np.argsort(-count[wt],kind='stable')[:8]],ko[:8]];a=a[rows].copy();x=x[rows];labels=labels.iloc[rows]
    tokens=resources()[0];median=pickle.load(open(ROOT/cfg['reference'],'rb'))
    keep=a.var.is_core.to_numpy() & a.var_names.isin(tokens);genes=a.var_names[keep].to_numpy();x=x[:,keep].tocsr()
    scale=np.array([median.get(g,1.) for g in genes]);col=int(np.flatnonzero(genes==cfg['target'])[0]);target=tokens[cfg['target']]
    if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid median reference')
    token_col={tokens[g]:j for j,g in enumerate(genes)};tokenize=official_tokenizer();coverage=[];manifest=[];ids=[];values=[];offsets=[0]
    rng=np.random.default_rng(cfg['seed'])
    def append(cell,group,operator,ii,vv,perturbed_gene=''):
        ii,vv=ii[:cfg['max_length']],vv[:cfg['max_length']]
        manifest.append(dict(sequence_row=len(manifest),cell_id=cell,group=group,operator=operator,perturbed_gene=perturbed_gene,length=len(ii)))
        ids.extend(ii);values.extend(vv);offsets.append(len(ids))
    for r in range(len(a)):
        cell=a.obs_names[r];group='WT' if labels.iloc[r]==wt_label else 'KO'
        v=x[r].toarray().ravel()/scale;ii,vv=tokenize(np.log2(1+v),genes,tokens)
        rank=ii.index(target)+1 if target in ii else 0;n=min(len(ii),cfg['max_length'])
        valid=0<rank<=n//2;full=0<rank<=len(ii)//2
        coverage.append(dict(cell_id=cell,group=group,sample=a.obs['sample'].iloc[r],target_counts=int(x[r,col]),rank=rank,full_length=len(ii),input_length=n,detected=rank>0,in_top2048=0<rank<=n,top_half_valid=valid,top_half_full=full))
        if group=='KO':append(cell,group,'real_KO',ii,vv);continue
        if not full:continue
        append(cell,group,'before',ii,vv)
        for fraction in cfg['fractions']:
            changed=v.copy();changed[col]*=fraction
            new_ids,new_values=tokenize(np.log2(1+changed),genes,tokens)
            if fraction==1 and (ii!=new_ids or not np.array_equal(vv,new_values)):raise ValueError('Sham changes inputs')
            if fraction==0 and target in new_ids:raise ValueError('Zeroed target retained')
            append(cell,group,f'{cfg["symbol"]}_{fraction:g}',new_ids,new_values,cfg['target'])
        p=rank-1;choices=np.arange(max(0,p-cfg['random_rank_window']),min(len(ii),p+cfg['random_rank_window']+1));choices=choices[choices!=p]
        for j,pos in enumerate(rng.choice(choices,cfg['random_deletions'],replace=len(choices)<cfg['random_deletions'])):
            changed=v.copy();idx=token_col[ii[pos]];changed[idx]=0
            ni,nv=tokenize(np.log2(1+changed),genes,tokens);append(cell,group,f'random{j}',ni,nv,genes[idx])
    cov=pd.DataFrame(coverage);cov.to_csv(out/'coverage_cells.csv.gz',index=False)
    cov.groupby('group').agg(cells=('cell_id','size'),detected=('detected','sum'),in_top2048=('in_top2048','sum'),top_half_valid=('top_half_valid','sum'),top_half_full=('top_half_full','sum')).to_csv(out/'coverage_summary.csv')
    a.obs.to_csv(out/'source_metadata.tsv.gz',sep='\t',index_label='cell_id')
    pd.DataFrame(manifest).to_csv(out/'embedding_manifest.csv.gz',index=False)
    for name,array in [('gene_ids',np.array(ids,dtype='int32')),('values',np.array(values,dtype='float32')),('offsets',np.array(offsets,dtype='int64'))]:np.save(out/f'{name}.npy',array)
    write_json(out/'preparation.json',dict(config=cfg,smoke=smoke,source_sha256=file_hash(ROOT/cfg['source']),reference_sha256=file_hash(ROOT/cfg['reference']),token_dictionary_sha256=file_hash(ROOT/'external/GeneCompass/scdata/dict/human_mouse_tokens.pickle'),cells=len(a),sequences=len(manifest),treatment=cfg.get('treatment','IL33'),median_matched_genes=sum(g in median for g in genes),measured_core_vocab_genes=len(genes),paper='https://pmc.ncbi.nlm.nih.gov/articles/PMC11615217/',limitations=['Original mouse training median unavailable; mouse-homolog proxy, not exact original preprocessing','Paper top50% denominator not explicit: valid Top2048 primary; full nonzero ranking sensitivity',cfg.get('source_limitation','One WT and one KO library, no known biological replicate matching; library/genotype confounded'),'Pairing seeds are Monte Carlo draws, not biological replicates']))
    print(cov.groupby('group')[['detected','in_top2048','top_half_valid','top_half_full']].sum(),flush=True)

def check(cfg,out):
    saved=json.loads((out/'preparation.json').read_text())
    if saved['config']!=cfg or saved['source_sha256']!=file_hash(ROOT/cfg['source']) or saved['reference_sha256']!=file_hash(ROOT/cfg['reference']):raise ValueError('Prepared input changed')
    return saved

@torch.inference_mode()
def extract(cfg,out):
    saved=check(cfg,out);model=load(cfg['model'])
    if not model.bert.use_cls_token:raise ValueError('CLS required')
    ids,values,offsets=[np.load(out/f'{n}.npy',mmap_mode='r') for n in ['gene_ids','values','offsets']]
    lengths=np.diff(offsets);order=np.argsort(lengths,kind='stable');h=np.empty((len(lengths),model.config.hidden_size),dtype='float32')
    for start in range(0,len(order),cfg['batch_size']):
        rows=order[start:start+cfg['batch_size']]
        seq=[(ids[offsets[r]:offsets[r+1]],values[offsets[r]:offsets[r+1]]) for r in rows]
        b=batch(seq);z=encode(model,b,cfg['model'])[:,0]
        if start==0:
            error=float((encode(model,b,cfg['model'])[:,0]-z).abs().max())
            if error>1e-5:raise ValueError('Nonrepeatable embedding')
        h[rows]=z.cpu().numpy()
        if start%512==0:print('encoded',start,len(order),flush=True)
    if not np.isfinite(h).all() or (np.linalg.norm(h,axis=1)==0).any():raise ValueError('Invalid embedding')
    manifest=pd.read_csv(out/'embedding_manifest.csv.gz')
    before=manifest[manifest.operator.eq('before')].set_index('cell_id').sequence_row
    sham=manifest[manifest.operator.eq(cfg['symbol']+'_1')].set_index('cell_id').sequence_row.loc[before.index]
    sham_error=float(np.abs(h[before.to_numpy()]-h[sham.to_numpy()]).max())
    if sham_error>1e-4:raise ValueError('Sham alignment failed')
    np.save(out/'cls_embeddings.npy',h)
    for op,filename in [('before','WT_before'),(cfg['symbol']+'_0','WT_insilico_KO'),('real_KO','real_KO')]:
        selected=manifest[manifest.operator.eq(op)];np.save(out/f'{filename}_embeddings.npy',h[selected.sequence_row]);selected.to_csv(out/f'{filename}_cells.csv',index=False)
    write_json(out/'inference.json',dict(**provenance(cfg['model']),preparation=saved,representation='last layer CLS, species=1, FP32 eval, no gradients',sham_max_abs=sham_error))

def score(cfg,out,embedding_filename='cls_embeddings.npy'):
    saved=check(cfg,out);inf=json.loads((out/'inference.json').read_text())
    if inf['preparation']!=saved:raise ValueError('Embedding/input mismatch')
    manifest=pd.read_csv(out/'embedding_manifest.csv.gz');coverage=pd.read_csv(out/'coverage_cells.csv.gz').set_index('cell_id')
    h=np.load(out/embedding_filename).astype('float64');h/=np.linalg.norm(h,axis=1,keepdims=True)
    before=manifest[manifest.operator.eq('before')].set_index('cell_id').sequence_row
    ko=manifest[manifest.operator.eq('real_KO')];records=[]
    for scope,flag in [('top_half_valid','top_half_valid'),('top_half_full','top_half_full')]:
        cells=coverage.index[coverage.group.eq('WT') & coverage[flag]]
        if not len(cells):raise ValueError('No eligible WT for '+scope)
        if len(cells)>len(ko) and not cfg.get('pair_with_replacement',False):raise ValueError('Insufficient KO for sampling without replacement')
        for seed in cfg['pairing_seeds']:
            target=ko.iloc[np.random.default_rng(seed).choice(len(ko),len(cells),replace=cfg.get('pair_with_replacement',False))]
            ht=h[target.sequence_row];hc=h[before.loc[cells]];initial=(hc*ht).sum(1)
            for op in manifest.operator.unique():
                if op in ['before','real_KO']:continue
                hp=h[manifest[manifest.operator.eq(op)].set_index('cell_id').loc[cells,'sequence_row']]
                after=(hp*ht).sum(1);selfsim=(hc*hp).sum(1)
                records.extend(dict(scope=scope,seed=seed,cell_id=c,KO_cell_id=k,operator=op,cosine_before=float(b),cosine_after=float(a),delta_sim=float(a-b),cosine_WT_pert=float(s)) for c,k,b,a,s in zip(cells,target.cell_id,initial,after,selfsim))
    d=pd.DataFrame(records);d.to_csv(out/'paired_cosine.csv.gz',index=False)
    summary=d.groupby(['scope','seed','operator']).agg(cells=('cell_id','size'),cosine_before=('cosine_before','mean'),cosine_after=('cosine_after','mean'),delta_sim=('delta_sim','mean'),median_delta_sim=('delta_sim','median'),positive_fraction=('delta_sim',lambda s:float((s>0).mean())),cosine_WT_pert=('cosine_WT_pert','mean')).reset_index()
    summary.to_csv(out/'cosine_by_seed.csv',index=False);summary.groupby(['scope','operator']).mean(numeric_only=True).drop(columns='seed').to_csv(out/'cosine_summary.csv')
    primary=d[d.scope.eq('top_half_valid')].pivot(index=['seed','cell_id'],columns='operator',values='delta_sim')
    random=primary[[f'random{j}' for j in range(cfg['random_deletions'])]].mean(1)
    comparison=pd.DataFrame({'KO_delta_sim':primary[cfg['symbol']+'_0'],'random_delta_sim':random});comparison['excess_gain']=comparison.KO_delta_sim-comparison.random_delta_sim
    comparison.to_csv(out/'KO_vs_random.csv')
    comparison.groupby(level='seed').mean().to_csv(out/'KO_vs_random_summary.csv')
    print(summary.to_string(index=False))
    (out/'report.md').write_text(f'# {cfg["symbol"]} {cfg["model"]} in silico KO\n\n'+('SMOKE ONLY：非正式结果。\n\n' if saved['smoke'] else '')+f'处理条件metadata：{cfg.get("treatment","IL33")}。未知条件不推断为稳态或IL33。末层CLS，目标表达按config中的fractions缩放后完整重排并重新取Top2048，真实KO不按目标基因是否检出筛选。主分析按有效输入前50%，全非零前50%为敏感性分析。\n\nDelta sim = cos(h_pert,h_realKO) - cos(h_WT,h_realKO)。正值表示更接近所配对真实KO，不证明因果预测正确。随机删除排名相近的其他基因是附加对照；sham倍数1应接近零。三个seed不是三个生物学重复，不据细胞数报告生物学显著性。\n\n来源与重复限制见preparation.json；使用LoRA同源代理median，非原始小鼠median的完全复刻。\n\n```csv\n'+summary.to_csv(index=False)+'```\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','extract','score','all']);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--smoke',action='store_true');args=p.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output'];out=out.with_name(out.name+'_smoke') if args.smoke else out
    if args.stage in ['prepare','all']:prepare(cfg,out,args.smoke)
    if args.stage in ['extract','all']:extract(cfg,out)
    if args.stage in ['score','all']:score(cfg,out)
