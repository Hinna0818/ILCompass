"""Matched simple-head controls: population mean targets only; no test access."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import fcntl
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from train import ROOT,load_cache
from prepare import write_json


class GroupHead(nn.Module):
    def __init__(self,h0,effects,hidden_size,use_ilc2):
        super().__init__()
        self.use_ilc2=use_ilc2
        self.register_buffer('h_center',h0.mean(0))
        self.register_buffer('h_scale',h0.std(0,unbiased=False).clamp_min(1e-4))
        self.register_buffer('effects',effects)
        # Same initialized weights/shape in both controls; only ILC2 input is masked.
        self.net=nn.Sequential(nn.Linear(h0.shape[1]+2,hidden_size),nn.GELU(),nn.Linear(hidden_size,effects.shape[1]))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,h_mean,y_mean,condition):
        if condition.shape[-1]!=2 or (condition.sum(-1)>1).any():
            raise ValueError('Only PBS or single perturbations are allowed')
        hs=(h_mean-self.h_center)/self.h_scale
        if not self.use_ilc2:hs=torch.zeros_like(hs)
        condition=condition.expand(*hs.shape[:-1],2)
        residual=self.net(torch.cat([hs,condition],-1))*condition.sum(-1,keepdim=True)
        # Clipping applies to final population mean, not per-cell predicted expression.
        return (y_mean+condition@self.effects+residual).clamp_min(0)


def metrics(pred,truth,control):
    dp=pred-control;dt=truth-control
    cp=dp-dp.mean();ct=dt-dt.mean();den=cp.norm()*ct.norm()
    return {'delta_rmse':float((pred-truth).square().mean().sqrt()),
            'delta_pearson':float((cp*ct).sum()/den) if den>0 else None,
            'zero_mean_gene_fraction':float((pred==0).float().mean())}


@torch.no_grad()
def evaluate(model,data):
    h,y,pools=data;hm=h[pools['PBS']].mean(0);ym=y[pools['PBS']].mean(0)
    model.eval();r={};preds={}
    for name,c in [('IL33',[1.,0.]),('CGRP',[0.,1.])]:
        pred=model(hm,ym,torch.tensor(c));truth=y[pools[name]].mean(0)
        if not torch.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid group mean')
        r[name]=metrics(pred,truth,ym);preds[name]=pred
    r['score']=(r['IL33']['delta_rmse']+r['CGRP']['delta_rmse'])/2
    model.train();return r,preds


def run_one(cfg,root,train,val,genes,provenance,seed,mode):
    out=root/f'{mode}_seed{seed}';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():raise ValueError(f'Existing result {out}')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    h,y,pools=train;h0=h[pools['PBS']];y0=y[pools['PBS']]
    effects=torch.stack([y[pools[k]].mean(0)-y0.mean(0) for k in ['IL33','CGRP']])
    model=GroupHead(h0,effects,cfg['hidden_size'],mode=='ilc2')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    baseline,_=evaluate(model,val);best=baseline['score'];best_step=0;bad=0
    # Critical boundaries: initial model equals constrained baseline; PBS is identity.
    vh,vy,vp=val;vm=vy[vp['PBS']].mean(0);vhm=vh[vp['PBS']].mean(0)
    for i,name in enumerate(['IL33','CGRP']):
        pred=model(vhm,vm,torch.eye(2)[i]);torch.testing.assert_close(pred,(vm+effects[i]).clamp_min(0))
    torch.testing.assert_close(model(vhm,vm,torch.zeros(2)),vm)
    def save(name,step):
        tmp=out/f'{name}.partial.pt'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                    'mode':mode,'seed':seed,'best_step':best_step,'best_score':best,
                    'provenance':provenance,'torch_rng':torch.get_rng_state(),
                    'numpy_rng':rng.bit_generator.state},tmp);tmp.replace(out/f'{name}.pt')
    save('best',0);history=[];start=time.monotonic()
    print(f'START {mode} seed={seed} baseline={best:.8f}',flush=True)
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,cfg['max_steps']+1):
            task=int(rng.integers(2));name=['IL33','CGRP'][task]
            source=rng.choice(pools['PBS'],cfg['group_size'],replace=True)
            target=rng.choice(pools[name],cfg['group_size'],replace=True)
            c=torch.eye(2)[task]
            prediction=model(h[source].mean(0),y[source].mean(0),c)
            loss=(prediction-y[target].mean(0)).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite group loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%cfg['log_every']==0:
                log.write(json.dumps({'phase':'train','step':step,'condition':name,'group_mse':float(loss.detach()),'grad':float(grad)})+'\n')
            if step%cfg['validation_every']==0:
                scores,_=evaluate(model,val)
                if scores['score']<best:best=scores['score'];best_step=step;bad=0;save('best',step)
                else:bad+=1
                history.append({'step':step,**scores});log.write(json.dumps({'phase':'validation','step':step,**scores})+'\n')
                print(f'VALIDATION {mode} seed={seed} step={step} macro={scores["score"]:.8f} best={best:.8f} at {best_step} patience={bad}',flush=True)
                if bad>=cfg['patience']:break
    save('latest',step)
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model'])
    scores,predictions=evaluate(model,val);train_scores,_=evaluate(model,train)
    ablation=None
    if mode=='ilc2':
        model.use_ilc2=False;ablation,_=evaluate(model,val);model.use_ilc2=True
    result={'mode':mode,'seed':seed,'stopped_step':step,'best_step':best_step,'validation':scores,
            'baseline':baseline,'train':train_scores,'history':history,
            'train_mean_h_replacement_ablation':ablation,
            'parameters':sum(p.numel() for p in model.parameters()),
            'elapsed_seconds':time.monotonic()-start,'reason':'early_stopping' if bad>=cfg['patience'] else 'max_steps',
            'relative_improvement_over_constrained_baseline':1-scores['score']/baseline['score']}
    write_json(out/'result.json',result)
    frame=pd.DataFrame({'gene_id':genes})
    for name in ['IL33','CGRP']:
        frame[f'{name}_predicted_mean_log1p']=predictions[name].numpy()
        frame[f'{name}_observed_D_mean_log1p']=val[1][val[2][name]].mean(0).numpy()
    frame.to_csv(out/'D_group_predictions.tsv.gz',sep='\t',index=False)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/drugpred_group_head.json'))
    args=parser.parse_args();cfg=json.loads(open(args.config).read());out=ROOT/cfg['result_directory'];out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads'])
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if list(out.glob('*_seed*/best.pt')):raise ValueError('Refuse to overwrite existing experiment')
        source=ROOT/cfg['source_run'];original=json.loads((source/'config.json').read_text());cache=ROOT/original['cache_directory']
        sig=json.loads((cache/'signature.json').read_text())
        if sig!=json.loads((cache/'complete.json').read_text())['signature'] or sig['split_version']!='GSE136154_E0_v3_Ctrain_Dval':raise ValueError('Invalid provenance')
        cell_sets=[]
        for role,batch in [('train','C'),('validation','D')]:
            meta=pd.read_csv(cache/f'{role}_metadata.tsv',sep='\t')
            if not meta.author_batch.eq(batch).all() or not meta.cell_uid.is_unique:raise ValueError('Invalid roles')
            cell_sets.append(set(meta.cell_uid))
        if cell_sets[0]&cell_sets[1]:raise ValueError('Development overlap')
        train=load_cache(cache,'train','cpu');val=load_cache(cache,'validation','cpu')
        genes=(cache/'gene_ids.txt').read_text().splitlines()
        provenance={'config':cfg,'cache_signature':sig,'gene_ids':genes,
                    'nonnegativity':'clamp final group mean at zero during training and validation for all models',
                    'architecture':'770->64->G, GELU, zero-initialized final layer; condition-only masks 768 h features',
                    'sampling':'same numpy seed and group sequence for paired models; independent PBS/target groups',
                    'validation':'full D PBS population mean as input, full D target mean as score; no minibatch selection',
                    'limitations':['Independent biological backgrounds not multiplied by resampling','D is reused development validation, not independent test',
                    'Equal nominal parameter counts; condition-only input feature weights receive no data gradients']}
        write_json(out/'config.json',provenance)
        write_json(out/'status.json',{'phase':'running'})
        results=[]
        for seed in cfg['seeds']:
            for mode in ['condition_only','ilc2']:
                results.append(run_one(cfg,out,train,val,genes,provenance,seed,mode))
                write_json(out/'status.json',{'phase':'running','completed_runs':len(results)})
        table=[]
        for r in results:
            table.append({'model':r['mode'],'seed':r['seed'],'best_step':r['best_step'],'stopped_step':r['stopped_step'],
                          'IL33_rmse':r['validation']['IL33']['delta_rmse'],'CGRP_rmse':r['validation']['CGRP']['delta_rmse'],
                          'macro_rmse':r['validation']['score'],
                          'IL33_pearson':r['validation']['IL33']['delta_pearson'],'CGRP_pearson':r['validation']['CGRP']['delta_pearson']})
        frame=pd.DataFrame(table);frame.to_csv(out/'comparison.tsv',sep='\t',index=False)
        summary={}
        for mode in ['condition_only','ilc2']:
            selected=frame[frame.model.eq(mode)]
            summary[mode]={k:{'mean':float(selected[k].mean()),'sd':float(selected[k].std(ddof=1))} for k in ['IL33_rmse','CGRP_rmse','macro_rmse']}
        write_json(out/'summary.json',{'baseline':results[0]['baseline'],'models':summary,
                                      'runs':table,'combination_accessed':False})
        write_json(out/'status.json',{'phase':'completed','runs':len(results)})
        print('COMPLETED '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
