"""Fixed model adapters; reuse existing tokenizers and pretrained heads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import sys,json,pickle,importlib.util,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for path in ('src','external/GeneCompass','src/downstream/geneko','compare_pretrain/genepass_lora','scratch/src'):
    sys.path.insert(0,str(ROOT/path))
import numpy as np
import pandas as pd
import torch
from genecompass_compat import load_genecompass
from nmur1_genecompass_original import official_tokenizer
from il33_data import encode_control_rows
from model_v3 import V3Transformer,V3Config
from posttrain_data import file_hash

MODELS={
 'raw':'model/pretrained/GeneCompass_base',
 'lora':'results/ILC2_GeneCompass_LoRA_v1/r8_seed0/best.pt',
 'official':'model/posttrained/base_last1_20261022_official_0.8_0.2',
 'tissue':'model/posttrained/base_last1_20261028_tissue_norm_045_010_045',
 'scratch_v3':'scratch/results/ilc2_scratch_v3/E3_8L384_MLM_seed0/best.pt'}
CORPUS=ROOT/'scratch/data/pretrained_corpus/ILC2_steady_state_with_BM_v1.h5ad'
NMUR1=ROOT/'data/processed/GSE102299_qc/GSE102299_Nmur1_KO_ILC2.h5ad'
OUT=ROOT/'results/ILC2_pretrain_model_compare_v1'

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def resources():
    tokens=pickle.load(open(ROOT/'external/GeneCompass/scdata/dict/human_mouse_tokens.pickle','rb'))
    median=pickle.load(open(ROOT/'data/references/mouse_gene_median.pickle','rb'))
    prepared=ROOT/'scratch/data/prepared/ilc2_scratch_v3'
    vocab=pd.read_csv(prepared/'gene_vocab.tsv',sep='\t')
    med=pd.read_csv(prepared/'gene_medians_train.tsv',sep='\t')
    if not np.array_equal(vocab.gene_id,med.gene_id):raise ValueError('Median/vocabulary mismatch')
    return tokens,median,vocab,med

def sequences(a,kind,res):
    tokens,median,vocab,med=res
    if kind=='scratch_v3':
        seq=[]
        for start in range(0,len(a),64):
            ii,vv,nn=encode_control_rows(a.X[start:start+64],a.var_names,vocab,med,8192)
            seq.extend((ii[j,:n].astype('int64'),vv[j,:n].astype('float32')) for j,n in enumerate(nn))
        return seq
    keep=a.var_names.isin(tokens)
    if 'present_in_all_studies' in a.var:keep &= a.var.present_in_all_studies.to_numpy()
    elif 'is_core' in a.var:keep &= a.var.is_core.to_numpy()
    genes=a.var_names[keep].to_numpy();x=a.X[:,keep].tocsr()
    scale=np.array([median.get(g,1.) for g in genes]);tokenize=official_tokenizer();seq=[]
    for row in range(len(a)):
        ii,vv=tokenize(np.log2(1+x[row].toarray().ravel()/scale),genes,tokens)
        seq.append((np.asarray(ii[:2048],dtype='int64'),np.asarray(vv[:2048],dtype='float32')))
    return seq

def load(name):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    path=ROOT/MODELS[name]
    if name=='scratch_v3':
        state=torch.load(path,map_location='cpu',weights_only=False)
        from train_v3 import frozen_inputs
        prepared,source_provenance,_=frozen_inputs(state['input_config'])
        if prepared!=ROOT/'scratch/data/prepared/ilc2_scratch_v3' or source_provenance!=state['provenance']:raise ValueError('Scratch preprocessing provenance mismatch')
        m=V3Transformer(V3Config(**state['model_config']));m.load_state_dict(state['model'])
    elif name=='lora':
        spec=importlib.util.spec_from_file_location('lora_adapter_model',ROOT/'compare_pretrain/genepass_lora/model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=json.loads((ROOT/'compare_pretrain/genepass_lora/config.json').read_text());cfg['gradient_checkpointing']=False
        state=torch.load(path,map_location='cpu',weights_only=False);m=module.build(cfg,state['model'])
    else:
        m,report=load_genecompass(path)
        if report['missing'] or report['unexpected']:raise ValueError(report)
    return m.cuda().eval().requires_grad_(False)

def batch(seq):
    lengths=np.array([len(x[0]) for x in seq]);width=int(lengths.max())
    if lengths.min()==0:raise ValueError('Empty input')
    ii=np.zeros((len(seq),width),dtype='int64');vv=np.zeros(ii.shape,dtype='float32')
    for j,(ids,values) in enumerate(seq):ii[j,:len(ids)]=ids;vv[j,:len(ids)]=values
    return {k:torch.as_tensor(v,device='cuda') for k,v in dict(input_ids=ii,values=vv,attention_mask=np.arange(width)[None,:]<lengths[:,None],position_ids=np.broadcast_to(np.arange(width),ii.shape).copy()).items()}

def encode(model,b,name):
    if name=='scratch_v3':return model(**b)['gene_embeddings']
    return model.bert(input_ids=b['input_ids'],values=b['values'],attention_mask=b['attention_mask'],species=torch.ones((len(b['input_ids']),1),dtype=torch.long,device='cuda'),emb_warmup_alpha=1.,return_dict=True).last_hidden_state

def provenance(name):
    p=ROOT/MODELS[name];p=p/'pytorch_model.bin' if p.is_dir() else p
    return dict(model=name,checkpoint=str(p),sha256=file_hash(p),precision='FP32',encoder_updates=0)
