"""Original GeneCompass PBS features with the same median/FP32/pooling as ILC2."""
import json,pickle
import numpy as np
import pandas as pd
import torch
from models import ROOT,FrozenILC2Encoder
from posttrain_data import encode,file_hash
from genecompass_coverage import map_genes
from prepare import write_json

CHECKPOINT=ROOT/'model/pretrained/GeneCompass_base'
CACHE=ROOT/'src/downstream/drugpred/downstream_data/IL33_expanded_v1/prepared_original_PBS_v1'


def original_h(a,key):
    CACHE.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    signature=dict(checkpoint=file_hash(CHECKPOINT/'pytorch_model.bin'),tokens=file_hash(ROOT/cfg['token_dictionary']),medians=file_hash(ROOT/cfg['gene_median_reference']),encoding='median',max_gene_tokens=2048,precision='float32',pooling='masked_gene_mean')
    folder=CACHE/key;folder.mkdir(exist_ok=True)
    # Counts hash covers the exact selected cells and feature order, independent of file container metadata.
    import hashlib
    digest=hashlib.sha256()
    for part in [a.X.data,a.X.indices,a.X.indptr]:digest.update(part.tobytes())
    digest.update('\n'.join(a.obs_names).encode());digest.update('\n'.join(a.var_names).encode())
    sig=dict(signature,counts_sha256=digest.hexdigest())
    if (folder/'signature.json').exists() and json.loads((folder/'signature.json').read_text())!=sig:raise ValueError('Original cache changed')
    write_json(folder/'signature.json',sig)
    if (folder/'h.npy').exists():return np.load(folder/'h.npy')
    old=ROOT/'src/downstream/drugpred/downstream_data/GSE136154/prepared_original_E0_v1'
    oldsig=json.loads((old/'signature.json').read_text());reuse=np.full((a.n_obs,768),np.nan,dtype='float32')
    if any(oldsig[k]!=v for k,v in signature.items()):raise ValueError('Original old checkpoint/encoding differs')
    for split in ['train','validation']:
        source=old.parent/split/f'GSE136154_{split}.h5ad'
        if file_hash(source)!=oldsig['inputs'][split]:raise ValueError('Old source changed')
        meta=pd.read_csv(old/f'{split}_metadata.tsv',sep='\t',index_col=0);loc=meta.index.get_indexer(a.obs_names);mask=loc>=0
        if mask.any():reuse[mask]=np.load(old/f'{split}_h.npy',mmap_mode='r')[loc[mask]]
    new=np.flatnonzero(~np.isfinite(reuse).all(1));partial=folder/'h.partial.npy'
    if partial.exists():h=np.lib.format.open_memmap(partial,mode='r+')
    else:
        h=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=reuse.shape);h[:]=reuse;h.flush()
    new=np.flatnonzero(~np.isfinite(h).all(1))
    if len(new):
        if not torch.cuda.is_available():raise RuntimeError('GPU access needed for original encoder')
        tok=pickle.loads((ROOT/cfg['token_dictionary']).read_bytes());med=pickle.loads((ROOT/cfg['gene_median_reference']).read_bytes())
        cols,median,_=map_genes(a.var_names,tok,med);ids,values,lengths=encode(a.X[new],cols,median,2048,'median')
        torch.backends.cuda.matmul.allow_tf32=False;model=FrozenILC2Encoder(CHECKPOINT).cuda().eval()
        for start in range(0,len(new),8):
            end=min(start+8,len(new));length=int(lengths[start:end].max());ti=torch.tensor(ids[start:end,:length],device='cuda',dtype=torch.long);tv=torch.tensor(values[start:end,:length],device='cuda')
            mask=torch.arange(length,device='cuda')[None,:]<torch.tensor(lengths[start:end],device='cuda')[:,None]
            result=model(ti,tv,mask).cpu().numpy()
            if not np.isfinite(result).all():raise ValueError('Nonfinite original embedding')
            h[new[start:end]]=result
            if end%200==0 or end==len(new):h.flush();print('ORIGINAL ENCODE',key,end,'/',len(new),flush=True)
        del model;torch.cuda.empty_cache()
    h.flush();del h;partial.replace(folder/'h.npy');a.obs.to_csv(folder/'metadata.tsv.gz',sep='\t')
    return np.load(folder/'h.npy')
