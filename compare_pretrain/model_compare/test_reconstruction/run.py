"""Common-held-out cells, common targets, 100% joint masking; no fitting."""
import sys,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared import *
import anndata as ad
from torch.nn import functional as F
DEST=OUT/'test_reconstruction'

def cohort():
    old=pd.read_csv(ROOT/'data/prepared/ilc2_posttrain_20261021_update/posttrain_test_cell_map.tsv.gz',sep='\t')
    manifests=[pd.read_csv(ROOT/p,sep='\t') for p in ['scratch/data/prepared/ilc2_scratch_v03/split_manifest.tsv.gz','scratch/data/prepared/ilc2_scratch_v3/split_manifest.tsv.gz']]
    cells=set(old.cell_uid)
    for d in manifests:cells &= set(d.loc[d.split.eq('test'),'cell_id'])
    if not cells:raise ValueError('No common held-out cells; do not silently substitute training cells')
    a=ad.read_h5ad(CORPUS,backed='r');rows=np.flatnonzero(a.obs_names.isin(cells));out=a[rows].to_memory();a.file.close()
    if set(out.obs_names)!=cells:raise ValueError('Missing test cells')
    return out

def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'targets.csv').exists():raise FileExistsError('Fixed manifest already exists; reuse it')
    a=cohort();res=resources();gc=sequences(a,'raw',res);sc=sequences(a,'scratch_v3',res)
    gt={v:k for k,v in res[0].items()};st=dict(zip(res[2].token_id,res[2].gene_id));records=[]
    for cell,(g,s) in enumerate(zip(gc,sc)):
        common=sorted(set(gt[int(t)] for t in g[0])&set(st[int(t)] for t in s[0]))
        seed=int.from_bytes(hashlib.sha256(('20261114|'+a.obs_names[cell]).encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed);chosen=rng.choice(common,max(1,round(.15*len(common))),replace=False) if common else []
        records.extend(dict(cell_id=a.obs_names[cell],gene_id=t) for t in chosen)
    pd.DataFrame(records).to_csv(DEST/'targets.csv',index=False)
    a.obs.to_csv(DEST/'cells.tsv.gz',sep='\t',index_label='cell_id')
    write_json(DEST/'design.json',dict(cells=len(a),targets=len(records),source=str(CORPUS),source_sha256=file_hash(CORPUS),selection='Intersection of old posttrain test, scratch v03 test and v3 test; LoRA reuses v03 split',mask='15% of genes present in BOTH native inputs; selected targets 100% ID+value masked, no random/unchanged leakage',limitations=['Not native 80/10/10 validation: do not compare absolute scores to training logs','Original GeneCompass pretraining overlap unknown','Native input contexts and ID vocabularies differ','Native MSE is not comparable across preprocessing; use logCP10K MSE for cross-family comparison']))
    print('Prepared',len(a),'cells',len(records),'targets')

@torch.inference_mode()
def evaluate(name,limit):
    design=json.loads((DEST/'design.json').read_text())
    if file_hash(CORPUS)!=design['source_sha256']:raise ValueError('Source changed after fixing targets')
    a=cohort();meta=pd.read_csv(DEST/'cells.tsv.gz',sep='\t',index_col=0)
    if list(a.obs_names)!=list(meta.index):raise ValueError('Fixed cell order changed')
    targets=pd.read_csv(DEST/'targets.csv');res=resources();seq=sequences(a,name,res)
    model=load(name);panel=pd.read_csv(ROOT/'results/ILC2_tissue_gene_exposure_v1/functional_gene_sets.csv')
    functional=set(panel.gene_id);vocab=res[2];token_map=dict(zip(vocab.gene_id,vocab.token_id)) if name=='scratch_v3' else res[0]
    med=dict(zip(vocab.gene_id,res[3].median_positive_cp10k)) if name=='scratch_v3' else res[1]
    records=[];total=np.asarray(a.X.sum(1)).ravel();groups=targets.groupby('cell_id').gene_id.apply(list).to_dict()
    for r in range(min(len(a),limit) if limit else len(a)):
        genes=groups.get(a.obs_names[r],[])
        if not genes:continue
        b=batch([seq[r]]);lookup={int(t):j for j,t in enumerate(seq[r][0])}
        genes=sorted(genes,key=lambda g:lookup[token_map[g]]);pos=[lookup[token_map[g]] for g in genes]
        truth=b['values'][0,pos].clone();labels=b['input_ids'][0,pos].clone()
        b['input_ids'][0,pos]=1 if name=='scratch_v3' else res[0]['<mask>']
        b['values'][0,pos]=-1.
        h=encode(model,b,name)
        if name=='scratch_v3':
            mask=torch.zeros_like(b['attention_mask']);mask[0,pos]=True
            logits,pred=model.predict_targets(h,mask);labels=labels-2
        else:
            z=h[0,torch.as_tensor(pos,device='cuda')+1];logits=model.cls(z);pred=model.cls4value(z).squeeze(-1)
        ce=F.cross_entropy(logits.float(),labels,reduction='none').cpu().numpy();hit=(logits.argmax(-1)==labels).cpu().numpy()
        predictions=pred.float().cpu().numpy();truth=truth.cpu().numpy()
        for j,g in enumerate(genes):
            scale=med.get(g,1.)*(1 if name=='scratch_v3' else 10000/total[r])
            predicted_cp=max(0.,np.expm1(float(predictions[j])*np.log(2))*scale)
            actual_cp=float(a.X[r,a.var_names.get_loc(g)])*10000/total[r]
            recovered_cp=np.expm1(float(truth[j])*np.log(2))*scale
            if not np.isclose(recovered_cp,actual_cp,rtol=1e-4,atol=1e-6):raise ValueError('Native expression inverse transform mismatch')
            records.append(dict(cell_id=a.obs_names[r],gene_id=g,functional=g in functional,id_ce=float(ce[j]),id_correct=int(hit[j]),native_squared_error=float((predictions[j]-truth[j])**2),predicted_logCP10K=float(np.log1p(predicted_cp)),target_logCP10K=float(np.log1p(actual_cp))))
        if (r+1)%100==0:print(name,r+1,len(a),flush=True)
    d=pd.DataFrame(records)
    if not np.isfinite(d.select_dtypes('number')).all().all():raise ValueError('Nonfinite predictions')
    d['logCP10K_squared_error']=(d.predicted_logCP10K-d.target_logCP10K)**2
    folder=DEST/(name+('_smoke' if limit else ''));folder.mkdir(exist_ok=True)
    d.to_csv(folder/'predictions.csv.gz',index=False)
    summary=[]
    for scope,sub in [('all',d),('functional',d[d.functional])]:
        summary.append(dict(model=name,scope=scope,targets=len(sub),cells=sub.cell_id.nunique(),id_CE=sub.id_ce.mean(),id_top1=sub.id_correct.mean(),native_MSE=sub.native_squared_error.mean(),logCP10K_MSE=sub.logCP10K_squared_error.mean()))
    pd.DataFrame(summary).to_csv(folder/'metrics.csv',index=False)
    per_gene=d.groupby('gene_id').agg(targets=('id_ce','size'),id_CE=('id_ce','mean'),id_top1=('id_correct','mean'),native_MSE=('native_squared_error','mean'),logCP10K_MSE=('logCP10K_squared_error','mean'))
    correlations={}
    for gene,sub in d.groupby('gene_id'):
        correlations[gene]=float(sub.predicted_logCP10K.corr(sub.target_logCP10K)) if len(sub)>=3 and sub.predicted_logCP10K.std()>0 and sub.target_logCP10K.std()>0 else np.nan
    per_gene['logCP10K_Pearson']=pd.Series(correlations)
    per_gene.to_csv(folder/'per_gene.csv')
    write_json(folder/'provenance.json',dict(**provenance(name),design=design,smoke_cells=limit,expression_conversion='Inverse native log2 scaling to CP10K with source full-count library total; clip negative predicted CP10K to zero; log1p'))
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--model',choices=MODELS);p.add_argument('--smoke-cells',type=int,default=0);args=p.parse_args()
    if args.prepare:prepare()
    elif args.model:evaluate(args.model,args.smoke_cells)
    else:p.error('Use --prepare or --model')
