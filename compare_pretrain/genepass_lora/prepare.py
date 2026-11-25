"""Write official-ordered Top2048 inputs without modifying the source corpus."""
from common import *
import argparse,pickle
import anndata as ad
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from nmur1_genecompass_original import official_tokenizer

def prepare(cfg):
    out=ROOT/cfg['data_directory'];out.mkdir(parents=True,exist_ok=True)
    sig=signature(cfg)
    if (out/'complete.json').exists():
        old=json.loads((out/'complete.json').read_text())
        if old['signature']!=sig or old['max_length']!=cfg['max_length']:raise ValueError('Prepared inputs changed; use a new data_directory')
        print('Verified existing prepared inputs');return
    a=ad.read_h5ad(ROOT/cfg['source'],backed='r')
    split=pd.read_csv(ROOT/cfg['split_manifest'],sep='\t').set_index('cell_id')
    if not a.obs_names.is_unique or not split.index.is_unique or set(a.obs_names)!=set(split.index):raise ValueError('Split/corpus mismatch')
    split=split.loc[a.obs_names].copy()
    if not set(split.split)=={'train','validation','test'} or set(a.obs.study)&{'GSE102299','GSE136154','GSE189285'}:raise ValueError('Split or study boundary failure')
    for key in ['study','tissue','library_id']:
        if not np.array_equal(split[key].astype(str),a.obs[key].astype(str)):raise ValueError('Metadata mismatch')
    tokens=pickle.load(open(ROOT/cfg['token_dictionary'],'rb'));median=pickle.load(open(ROOT/cfg['median_reference'],'rb'))
    keep=a.var.present_in_all_studies.to_numpy()&a.var_names.isin(tokens)
    genes=a.var_names[keep].to_numpy();scale=np.array([median.get(g,1.) for g in genes])
    if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid median')
    ids=open_memmap(out/'input_ids.npy',mode='w+',dtype='int32',shape=(len(a),cfg['max_length']))
    values=open_memmap(out/'values.npy',mode='w+',dtype='float32',shape=ids.shape)
    lengths=np.empty(len(a),dtype='int16');tokenize=official_tokenizer()
    for left in range(0,len(a),256):
        raw=a.X[left:left+256].tocsr()
        if (raw.data<0).any() or not np.isfinite(raw.data).all() or not np.equal(raw.data,np.floor(raw.data)).all():raise ValueError('Counts must be finite nonnegative integers')
        x=raw[:,keep]
        for j in range(x.shape[0]):
            ii,vv=tokenize(np.log2(1+x[j].toarray().ravel()/scale),genes,tokens)
            n=min(len(ii),cfg['max_length']);row=left+j
            if n==0:raise ValueError('Empty input cell')
            ids[row,:]=0;values[row,:]=0;ids[row,:n]=ii[:n];values[row,:n]=vv[:n];lengths[row]=n
        if left%10240==0:print('Tokenized',left,'/',len(a),flush=True)
    ids.flush();values.flush();np.save(out/'lengths.npy',lengths)
    split['input_length']=lengths;split.to_csv(out/'cells.tsv.gz',sep='\t',index_label='cell_id')
    gene_meta=pd.DataFrame({'gene_id':genes,'token_id':[tokens[g] for g in genes],'median':scale,'median_available':[g in median for g in genes]});gene_meta.to_csv(out/'input_genes.tsv',sep='\t',index=False)
    panel=pd.read_csv(ROOT/cfg['functional_panel']).drop_duplicates('gene_id')
    panel['in_input_gene_space']=panel.gene_id.isin(genes);panel['token_id']=panel.gene_id.map(tokens)
    panel.to_csv(out/'functional_panel.csv',index=False)
    dump(out/'complete.json',dict(signature=sig,max_length=cfg['max_length'],cells=len(a),genes=len(genes),split_counts=split.split.value_counts().to_dict(),median_missing=int(sum(g not in median for g in genes)),tokenization='official tokenize_cell: log2(1+counts/frozen proxy median), np.argsort default ties, Top2048; no CP10K',code_sha256=file_hash(Path(__file__))))
    a.file.close();print('Prepared',out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=DEFAULT);args=p.parse_args();prepare(config(args.config))
