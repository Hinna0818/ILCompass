"""LoRA-only IL33 supplement: reuse locked historical inputs, trainers and metrics."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shared as adapter
ROOT=adapter.ROOT
# Historical prepare.py imports must resolve in the drugpred directory.
sys.path.insert(0,str(ROOT/'src/downstream/drugpred'))
import train_il33_mixed_delta as task
import train_il33_bidirectional_tissue as tissue
import train_il33_lung_full as cd
import il33_liver_fewshot_v3 as few
import numpy as np
import pandas as pd
import torch
from posttrain_data import file_hash
NAME='GeneCompass_LoRA'
BASE=ROOT/'results/ILC2_LoRA_IL33_v1'
DATA=ROOT/'data/processed/ILC2_LoRA_IL33_v1'
PARENTS={
 'mixed':ROOT/'data/processed/IL33_mixed_cell_delta_v1',
 'crossstudy':ROOT/'data/processed/IL33_external_delta_v1',
 'lung_liver':ROOT/'scratch/data/downstream/GSE189285_IL33_lung_to_liver_scratch_v3_seed0'}
CD_CFG=json.loads((ROOT/'configs/drugpred_il33_lung_tissue.json').read_text())
COMMON=ROOT/'data/processed/IL33_response_compare_v2'
LIVER=ROOT/'data/processed/IL33_lung_to_liver_classification_v1'

def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.resolve()!=source.resolve():raise ValueError(f'Unexpected asset: {target}')
    else:target.symlink_to(source.resolve())

def configuration(name):
    old=json.loads((PARENTS[name]/'prepare_manifest.json').read_text())
    cfg=dict(old['config'],models={NAME:adapter.MODELS['lora']},version='LoRA_'+name,
        data_directory=str((DATA/name).relative_to(ROOT)),
        result_directory=str((BASE/name).relative_to(ROOT)),
        model_directory=str((BASE/name/'checkpoints').relative_to(ROOT)))
    if 'checkpoint' in cfg:cfg['checkpoint']=adapter.MODELS['lora']
    return old,cfg

def cd_config():
    return dict(CD_CFG,checkpoint=adapter.MODELS['lora'],
        cache_directory=str((DATA/'C_D').relative_to(ROOT)),result_directory=str((BASE/'C_D').relative_to(ROOT)))

def few_config():
    cfg=json.loads((ROOT/'scratch/configs/il33_liver_fewshot_v3_seed0.json').read_text())
    cfg.update(source_data=str((DATA/'lung_liver').relative_to(ROOT)),source_results=str((BASE/'lung_liver').relative_to(ROOT)),
        data_directory=str((DATA/'fewshot').relative_to(ROOT)),result_directory=str((BASE/'fewshot').relative_to(ROOT)))
    return cfg

def audit():
    BASE.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in PARENTS:
        old,cfg=configuration(name);task.prepared(old['config'])
        for role in old['cell_map_sha256']:
            m=pd.read_csv(PARENTS[name]/role/'metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
            rows.append(dict(task=name,role=role,cells=len(m),PBS=int(m.condition.eq('PBS').sum()),IL33=int(m.condition.eq('IL33').sum()),seeds=str(cfg['seeds'])))
        adapter.write_json(BASE/(name+'_config.json'),cfg)
    adapter.write_json(BASE/'C_D_config.json',cd_config());adapter.write_json(BASE/'fewshot_config.json',few_config())
    pd.DataFrame(rows).to_csv(BASE/'historical_cell_counts.csv',index=False)
    # These are exact historical caches, not newly chosen normalization or ranks.
    cache_meta=pd.read_csv(COMMON/'PBS_metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
    cmeta=pd.read_csv(ROOT/CD_CFG['cache_directory']/'PBS_metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
    with np.load(COMMON/'PBS_inputs.npz') as common,np.load(ROOT/CD_CFG['cache_directory']/'PBS_inputs.npz') as c:
        loc=cache_meta.index.get_indexer(cmeta.index)
        if (loc<0).any():raise ValueError('C/D cells absent from shared cache')
        for key in ['ids','vals','length']:np.testing.assert_array_equal(common[key][loc],c[key])
    lm=pd.read_csv(LIVER/'liver_cells.tsv.gz',sep='\t',index_col=0,low_memory=False)
    all_ids=cache_meta.index.append(lm.index[lm.condition.eq('PBS')])
    if not all_ids.is_unique:raise ValueError('Duplicated feature cell IDs')
    for name,parent in PARENTS.items():
        for role in json.loads((parent/'prepare_manifest.json').read_text())['cell_map_sha256']:
            m=pd.read_csv(parent/role/'metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
            if not m.index[m.condition.eq('PBS')].isin(all_ids).all():raise ValueError('Task PBS missing from cache: '+name+'/'+role)
    corpus=pd.read_csv(ROOT/'scratch/data/prepared/ilc2_scratch_v03/split_manifest.tsv.gz',sep='\t')
    if set(all_ids)&set(corpus.cell_id) or set(cache_meta.study)&set(corpus.study):raise ValueError('Downstream overlaps LoRA training corpus')
    protected={}
    for directory in ['results/IL33_mixed_cell_delta_v1','results/IL33_external_delta_v1','results/GSE136154_IL33_encoder_comparison_v1','scratch/results/GSE189285_IL33_lung_to_liver_scratch_v3_seed0','scratch/results/GSE189285_IL33_liver_fewshot_v3_seed0']:
        for p in (ROOT/directory).iterdir():
            if p.is_file() and (p.suffix in ['.csv','.tsv'] or p.name=='test_protocol.json'):protected[str(p.relative_to(ROOT))]=file_hash(p)
    if (BASE/'audit.json').exists():
        for path,digest in json.loads((BASE/'audit.json').read_text())['source_metrics_sha256'].items():
            if file_hash(ROOT/path)!=digest:raise ValueError('Historical metric changed since first audit')
    adapter.write_json(BASE/'audit.json',dict(passed=True,source_metrics_sha256=protected,
        checkpoint=adapter.MODELS['lora'],historical_CD_inputs_equal_shared_cache=True,
        pretraining_cell_overlap=0,baseline_training_or_inference=False))
    print(pd.DataFrame(rows).to_string(index=False));print('Historical cache alignment and pretraining boundary passed',flush=True)

@torch.inference_mode()
def encode():
    audit();folder=DATA/'features';folder.mkdir(parents=True,exist_ok=True)
    tokcfg=json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    sig=json.loads((COMMON/'signature.json').read_text())
    for key,path in [('tokens',tokcfg['token_dictionary']),('medians',tokcfg['gene_median_reference'])]:
        if file_hash(ROOT/path)!=sig[key]:raise ValueError('Historical tokenizer reference changed')
    provenance=adapter.provenance('lora')
    meta=pd.read_csv(COMMON/'PBS_metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
    lm=pd.read_csv(LIVER/'liver_cells.tsv.gz',sep='\t',index_col=0,low_memory=False);lm=lm[lm.condition.eq('PBS')]
    expected={'checkpoint':provenance,'input_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [COMMON/'PBS_inputs.npz',COMMON/'PBS_metadata.tsv.gz',LIVER/'liver_inputs.npz',LIVER/'liver_cells.tsv.gz']},'pooling':'masked_gene_mean excluding CLS and PAD; FP32; no masking; historical stable-sorted counts/median Top2048'}
    signature=folder/'signature.json'
    if signature.exists() and json.loads(signature.read_text())!=expected:raise ValueError('Feature cache provenance changed')
    adapter.write_json(signature,expected)
    model=None
    for key,path,m in [('lung',COMMON/'PBS_inputs.npz',meta),('liver',LIVER/'liver_inputs.npz',lm)]:
        final=folder/(key+'_h.npy')
        if final.exists():
            h=np.load(final,mmap_mode='r')
            if h.shape!=(len(m),768) or not np.isfinite(h).all():raise ValueError('Invalid completed features')
            continue
        if model is None:
            model=adapter.load('lora');torch.backends.cudnn.allow_tf32=False
            if not model.bert.use_cls_token:raise ValueError('Historical pooling expects internal CLS')
        # NpzFile member access decompresses the entire array each time.
        # Materialize once so per-cell slicing does not repeatedly decode ~318 MB.
        with np.load(path) as z:
            input_ids,input_values,input_lengths=[z[k] for k in ['ids','vals','length']]
        source_rows=np.arange(len(m)) if key=='lung' else pd.read_csv(LIVER/'liver_cells.tsv.gz',sep='\t',index_col=0,low_memory=False).index.get_indexer(m.index)
        partial=folder/(key+'_partial.npy');exists=partial.exists()
        h=np.lib.format.open_memmap(partial,mode='r+' if exists else 'w+',dtype='float32',shape=(len(m),768))
        if not exists:h[:]=np.nan
        missing=np.flatnonzero(~np.isfinite(h).all(1))
        for start in range(0,len(missing),4):
            rows=missing[start:start+4];rr=source_rows[rows]
            b=adapter.batch([(input_ids[r,:input_lengths[r]],input_values[r,:input_lengths[r]]) for r in rr])
            hidden=adapter.encode(model,b,'lora')[:,1:]
            h[rows]=((hidden*b['attention_mask'][...,None]).sum(1)/b['attention_mask'].sum(1)[:,None]).cpu().numpy()
            if start%400==0:h.flush();print('LoRA encode',key,start,len(missing),flush=True)
        if not np.isfinite(h).all():raise ValueError('Incomplete features')
        h.flush();del h;partial.replace(final)
    pd.concat([meta,lm],sort=False).to_csv(folder/'cells.tsv.gz',sep='\t',index_label='cell_id')
    adapter.write_json(folder/'complete.json',dict(signature=expected,features={k:file_hash(folder/(k+'_h.npy')) for k in ['lung','liver']}))

def feature_rows(cell_ids):
    folder=DATA/'features';complete=json.loads((folder/'complete.json').read_text())
    if complete['signature']['checkpoint']!=adapter.provenance('lora'):raise ValueError('LoRA checkpoint changed')
    for key,digest in complete['features'].items():
        if file_hash(folder/(key+'_h.npy'))!=digest:raise ValueError('Frozen feature file changed')
    meta=pd.read_csv(folder/'cells.tsv.gz',sep='\t',index_col=0,low_memory=False)
    h=np.concatenate([np.load(folder/'lung_h.npy'),np.load(folder/'liver_h.npy')])
    loc=meta.index.get_indexer(cell_ids)
    if not meta.index.is_unique or (loc<0).any():raise ValueError('PBS feature cells missing')
    return h[loc]

def prepare(name):
    old,cfg=configuration(name);data,out,_=task.paths(cfg);out.mkdir(parents=True,exist_ok=True)
    if (data/'prepare_manifest.json').exists():task.prepared(cfg);return cfg
    parent=PARENTS[name]
    for filename in ['gene_panel.tsv','functional_gene_sets.tsv','cell_manifest.tsv.gz']:
        link(parent/filename,data/filename)
    hashes={}
    for role in old['cell_map_sha256']:
        for filename in ['metadata.tsv.gz','y.npy']+(['statistics.npz'] if role=='train' else []):link(parent/role/filename,data/role/filename)
        m=pd.read_csv(data/role/'metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
        np.save(data/role/(NAME+'_h.npy'),feature_rows(m.index[m.condition.eq('PBS')]))
        hashes[role]=file_hash(data/role/(NAME+'_h.npy'))
    manifest=dict(old,config=cfg,encoder_dimensions={NAME:768},checkpoint_sha256={NAME:file_hash(ROOT/adapter.MODELS['lora'])},aligned_feature_sha256=hashes,
        extension_of=dict(parent=str(parent),manifest_sha256=file_hash(parent/'prepare_manifest.json')))
    for key in ['scratch_provenance','checkpoint_epoch','encoder_final_norm_calls','encoding_statistics','protected_historical_sha256']:
        manifest.pop(key,None)
    manifest['representation']='LoRA FP32 frozen historical-input valid gene mean excluding CLS/PAD'
    adapter.write_json(data/'config.json',cfg);adapter.write_json(data/'prepare_manifest.json',manifest)
    return cfg

def score_predictions(pred,truth,ctrl,panel,info,folder):
    from il33_lung_metrics import differential,evaluate,correlation,r2
    from il33_benchmark import effect_rank_metrics
    eligible=panel.DEG_eligible.to_numpy() if 'DEG_eligible' in panel else np.ones(len(panel),dtype=bool)
    _,true_order=differential(truth,ctrl,eligible);_,order=differential(pred,ctrl,eligible)
    scores,moments=evaluate(pred,truth,ctrl,true_order,order,top_n=(20,50,100,200))
    scores.update(effect_rank_metrics(moments['observed_delta'],moments['predicted_delta']))
    folder.mkdir(parents=True,exist_ok=True)
    pd.concat([panel[['gene_id','gene_symbol']],pd.DataFrame(moments)],axis=1).to_csv(folder/'gene_predictions.tsv.gz',sep='\t',index=False)
    tops=[]
    for n in [20,50,100,200]:
        idx=true_order[:n];row=dict(**info,n=n)
        for target in ['mean','delta']:
            obs,pp=moments['observed_'+target][idx],moments['predicted_'+target][idx]
            row.update({target+'_RMSE':float(np.sqrt(np.mean((obs-pp)**2))),target+'_R2':r2(obs,pp),target+'_Pearson':correlation(obs,pp)})
        tops.append(row)
    pd.DataFrame(tops).to_csv(folder/'topn_metrics.csv',index=False)
    return dict(**info,**scores)

def train_score(name):
    cfg=prepare(name);device=torch.device('cuda');data,out,_,manifest,fingerprint=task.prepared(cfg)
    for seed in cfg['seeds']:
        if name=='lung_liver':tissue.train_fixed(cfg,NAME,seed,device,resume=True)
        else:task.train_one(cfg,NAME,seed,device,resume=True)
    panel=pd.read_csv(data/'gene_panel.tsv',sep='\t');dataset=task.load_split(data,manifest,'test',NAME)
    if name!='lung_liver':lock=task.checkpoint_lock(cfg,fingerprint)
    else:
        lock=json.loads((task.run_paths(cfg,NAME,0)[0]/'complete.json').read_text())
        if lock['final_sha256']!=file_hash(task.run_paths(cfg,NAME,0)[0]/'final.pt'):raise ValueError('Final head changed')
    adapter.write_json(out/'test_protocol.json',lock)
    stats=dict(np.load(data/'train/statistics.npz'));rows=[];extra=[]
    groups=cfg['backgrounds'] if name=='mixed' else cfg['target_backgrounds'] if name=='lung_liver' else cfg['external_backgrounds']
    for seed in cfg['seeds']:
        path=task.run_paths(cfg,NAME,seed)[0]/('final.pt' if name=='lung_liver' else 'best.pt')
        state=torch.load(path,map_location='cpu',weights_only=False);model=task.restore_head(state,cfg,len(panel),device)
        folder=out/'test'/NAME/f'seed{seed}'
        result,_=task.score_model(model,dataset,cfg,panel,stats,device,full_de=True,save_to=folder,groups=groups,training_reference='background' if name=='mixed' else 'training_average')
        rows.extend(dict(method=NAME,seed=seed,**r) for r in result)
        m,y,h=dataset;pm=m[m.condition.eq('PBS')];pred=np.asarray(y[m.condition.eq('PBS').to_numpy()])+np.load(folder/'effective_delta.npy')
        for group in groups:
            control=np.asarray(y[(m.condition.eq('PBS')&m.response_group.eq(group)).to_numpy()]);truth=np.asarray(y[(m.condition.eq('IL33')&m.response_group.eq(group)).to_numpy()])
            extra.append(score_predictions(pred[pm.response_group.eq(group)],truth,control,panel,dict(method=NAME,seed=seed,group=group),folder/group))
        # Historical cross-study supplement also tested source validation PBS inputs.
        if name=='crossstudy':
            vm,vy,vh=task.load_split(data,manifest,'validation',NAME);vpm=vm[vm.condition.eq('PBS')]
            control=np.asarray(y[m.condition.eq('PBS').to_numpy()]);truth=np.asarray(y[m.condition.eq('IL33').to_numpy()])
            for study in ['GSE136154','GSE102299']:
                mask=vpm.study.eq(study).to_numpy();x=np.asarray(vy[(vm.condition.eq('PBS')&vm.study.eq(study)).to_numpy()])
                prediction=x+task.predict_delta(model,vh[mask],x,cfg['prediction_batch_size'],device)
                extra.append(score_predictions(prediction,truth,control,panel,dict(method=NAME,seed=seed,group=study+'_source_PBS'),folder/(study+'_source_PBS')))
        del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False);pd.DataFrame(extra).to_csv(out/'extended_metrics.csv',index=False)

def c_d():
    cfg=cd_config();data=ROOT/cfg['cache_directory'];out=ROOT/cfg['result_directory'];out.mkdir(parents=True,exist_ok=True)
    parent=ROOT/CD_CFG['cache_directory']
    for filename in ['genes.tsv','train_y.npy','validation_y.npy','train_metadata.tsv.gz','validation_metadata.tsv.gz','PBS_metadata.tsv.gz']:link(parent/filename,data/filename)
    meta=pd.read_csv(data/'PBS_metadata.tsv.gz',sep='\t',index_col=0,low_memory=False);np.save(data/'PBS_h.npy',feature_rows(meta.index))
    adapter.write_json(data/'signature.json',dict(checkpoint=adapter.provenance('lora'),historical_signature=json.loads((parent/'signature.json').read_text()),features_sha256=file_hash(data/'PBS_h.npy')))
    cd.CFG=cfg;cd.CACHE=data;cd.OUT=out;cd.fit()
    groups=cd.load_data();g=groups['validation'];panel=pd.read_csv(data/'genes.tsv',sep='\t');rows=[]
    for seed in cfg['seeds']:
        pred=np.load(out/f'seed{seed}/D_predicted_log1p.npy',mmap_mode='r')
        rows.append(score_predictions(pred,g['target'],g['pbs'],panel,dict(method=NAME,seed=seed),out/f'seed{seed}'))
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False);adapter.write_json(out/'config.json',cfg)

def fewshot():
    few.ENCODER=NAME;cfg=few_config();data=ROOT/cfg['data_directory'];out=ROOT/cfg['result_directory'];data.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    original=ROOT/'scratch/data/downstream/GSE189285_IL33_liver_fewshot_v3_seed0'
    source=ROOT/cfg['source_data'];meta=pd.read_csv(source/'test/metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
    split=pd.read_csv(original/'split_manifest.tsv.gz',sep='\t',index_col=0,low_memory=False);np.testing.assert_array_equal(split.index,meta.index)
    pbs=meta.condition.eq('PBS').to_numpy()
    if not (split.condition.to_numpy()==meta.condition.to_numpy()).all():raise ValueError('Fewshot condition mismatch')
    link(original/'split_manifest.tsv.gz',data/'split_manifest.tsv.gz')
    if not (data/'manifest.json').exists():
        y=np.load(source/'test/y.npy',mmap_mode='r');h=np.load(source/f'test/{NAME}_h.npy');pbs=meta.condition.eq('PBS').to_numpy();loc=np.full(len(meta),-1);loc[pbs]=np.arange(pbs.sum());counts=[]
        for fraction in cfg['fractions']:
            key=few.label(fraction);selected=split[key].to_numpy();pr=np.flatnonzero(selected&pbs);tr=np.flatnonzero(selected&~pbs)
            if (split.loc[selected,'role']=='test').any():raise ValueError('Fewshot overlaps test')
            target=y[tr].mean(0,dtype=np.float64);x=np.array(y[pr]);np.savez_compressed(data/(key+'.npz'),control=x,h=h[loc[pr]],target_mean=target,delta=target-x.mean(0,dtype=np.float64));counts.append(dict(fraction=fraction,PBS=len(pr),IL33=len(tr)))
        protected={str(p.relative_to(ROOT)):file_hash(p) for p in [BASE/'lung_liver/checkpoints'/NAME/'seed0/final.pt',source/'prepare_manifest.json']}
        adapter.write_json(data/'manifest.json',dict(config=cfg,protected_sha256=protected,split_sha256=file_hash(data/'split_manifest.tsv.gz'),adaptation_sha256={few.label(f):file_hash(data/(few.label(f)+'.npz')) for f in cfg['fractions']},test_counts=split[split.role.eq('test')].condition.value_counts().to_dict(),adaptation_counts=counts,encoder_updates=0))
        pd.DataFrame(counts).to_csv(out/'adaptation_cell_counts.csv',index=False)
    few.train(cfg)  # Exact previous optimizer, loss, fixed steps and lung-only standardization.
    _,_,manifest=few.ready(cfg);complete=json.loads((out/'training_complete.json').read_text())
    for path,digest in complete['checkpoints'].items():
        if file_hash(out/path)!=digest:raise ValueError('Adaptation head changed')
    panel=pd.read_csv(source/'gene_panel.tsv',sep='\t');parent=json.loads((source/'config.json').read_text());test=split.role.eq('test').to_numpy()
    y=np.load(source/'test/y.npy',mmap_mode='r');h=np.load(source/f'test/{NAME}_h.npy')[test[pbs]]
    x=np.array(y[test&pbs]);truth=np.array(y[test&~pbs]);rows=[]
    variants=[('zero_shot',0,0,BASE/'lung_liver/checkpoints'/NAME/'seed0/final.pt')]+[(f'{few.label(f)}_step{s}',f,s,out/'checkpoints'/f'{few.label(f)}_step{s}.pt') for f in cfg['fractions'] for s in cfg['save_steps']]
    for label,fraction,step,path in variants:
        state=torch.load(path,map_location='cpu',weights_only=False);model=task.restore_head(state,parent,len(panel),torch.device('cpu'));pred=x+task.predict_delta(model,h,x,128,torch.device('cpu'))
        rows.append(score_predictions(pred,truth,x,panel,dict(method=label,fraction=fraction,step=step),out/'test'/label))
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False)

def compare():
    sources={
      'mixed':[('results/IL33_mixed_cell_delta_v1/test_background_metrics.csv','historical'),('results/IL33_mixed_cell_delta_official_v1/test_background_metrics.csv','historical_official')],
      'crossstudy':[('results/IL33_external_delta_v1/test_background_metrics.csv','historical')],
      'lung_liver':[('results/GSE189285_IL33_bidirectional_delta_v1/lung_to_liver/test_metrics.csv','historical'),('scratch/results/GSE189285_IL33_lung_to_liver_scratch_v3_seed0/comparison_seed0.csv','historical_v3')],
      'C_D':[('results/GSE136154_IL33_encoder_comparison_v1/metrics.tsv','historical_raw'),('results/GSE136154_IL33_tissue_encoder_v1/metrics.tsv','historical_tissue')],
      'fewshot':[('scratch/results/GSE189285_IL33_liver_fewshot_v3_seed0/comparison.csv','historical_v3_and_baselines')]}
    for name,files in sources.items():
        new=pd.read_csv(BASE/name/'test_metrics.csv');new['source']='LoRA_new';parts=[new]
        for path,label in files:
            d=pd.read_csv(ROOT/path,sep='\t' if path.endswith('.tsv') else ',');d['source']=label;d['source_path']=path;parts.append(d)
        pd.concat(parts,ignore_index=True,sort=False).to_csv(BASE/name/'comparison_with_history.csv',index=False)
    verify_history()

def verify_history():
    audit_record=json.loads((BASE/'audit.json').read_text())
    for path,digest in audit_record['source_metrics_sha256'].items():
        if file_hash(ROOT/path)!=digest:raise ValueError('Historical scores changed')
    adapter.write_json(BASE/'verification.json',dict(historical_metrics_unchanged=True,only_LoRA_encoder_inferred=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','encode','mixed','C_D','lung_liver','fewshot','crossstudy','compare','all']);args=p.parse_args();torch.set_num_threads(4)
    if args.stage=='audit':audit()
    elif args.stage=='encode':encode()
    elif args.stage=='all':
        encode();c_d()
        for name in ['mixed','lung_liver']:train_score(name)
        fewshot();train_score('crossstudy');compare()
    elif args.stage=='compare':compare()
    elif args.stage=='C_D':c_d()
    elif args.stage=='fewshot':fewshot()
    else:train_score(args.stage)
