"""Single-GPU frozen-base LoRA MLM training; resume only at optimizer boundaries."""
from common import *
import argparse,math,random,csv
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from model import build,masked_outputs
from data import Inputs
from posttrain_training import autocast,save_training_state,restore_training_state,step_optimizer

def to_device(b):return {k:v.cuda() for k,v in b.items()}

@torch.no_grad()
def evaluate(model,data,cfg,rows=None):
    model.eval();rows=data.rows('validation') if rows is None else rows;records=[];functional=[]
    for start in range(0,len(rows),cfg['micro_batch_size']):
        indices=rows[start:start+cfg['micro_batch_size']]
        b=to_device(data.batch(indices,cfg['eval_mask_seed'],validation=True))
        with autocast(torch.device('cuda'),cfg):o=masked_outputs(model,b)
        truth=o['targets'];n=len(truth);iderr=torch.nn.functional.cross_entropy(o['logits'],truth,reduction='none')
        err=(o['pred']-o['truth']).square();hit=o['logits'].argmax(-1).eq(truth)
        records.append([float(iderr.sum()),float(err.sum()),float(hit.sum()),n])
        selected=torch.isin(truth,torch.as_tensor(data.functional,device='cuda'))
        if selected.any():
            functional.extend(zip(truth[selected].cpu().tolist(),iderr[selected].cpu().tolist(),err[selected].cpu().tolist(),hit[selected].cpu().tolist()))
    sums=np.asarray(records).sum(0);result=dict(id=sums[0]/sums[3],expression=sums[1]/sums[3],top1=sums[2]/sums[3],masked_positions=int(sums[3]),cells=len(rows))
    result['loss']=.2*result['id']+.8*result['expression']
    f=pd.DataFrame(functional,columns=['token_id','id_ce','expression_mse','top1'])
    table=f.groupby('token_id').agg(id_ce=('id_ce','mean'),expression_mse=('expression_mse','mean'),top1=('top1','mean'),masked_positions=('top1','size')).reset_index() if len(f) else f
    if len(f):result['functional']={'id':float(f.id_ce.mean()),'expression':float(f.expression_mse.mean()),'top1':float(f.top1.mean()),'masked_positions':len(f)}
    if not np.isfinite(result['loss']):raise ValueError('Nonfinite validation loss')
    return result,table

def train(cfg,resume):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if cfg['precision']=='bfloat16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable; choose float32 explicitly')
    if cfg['precision'] not in ['bfloat16','float32']:raise ValueError('Only BF16/FP32 supported for this experiment')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Inputs(cfg);out=ROOT/cfg['output_directory'];out.mkdir(parents=True,exist_ok=True)
    # Strict resume checks include all data/training logic, not just the adapter shapes.
    provenance=dict(config=cfg,data=data.report,code={name:file_hash(Path(__file__).with_name(name)) for name in ['common.py','data.py','model.py','train.py']})
    record=out/'run_config.json'
    if record.exists():
        if not resume:raise ValueError('Run exists; use --resume or a new output_directory')
        if json.loads(record.read_text())!=provenance:raise ValueError('Resume config/source/code changed')
    elif resume:raise ValueError('No existing run to resume')
    else:dump(record,provenance)
    model=build(cfg).cuda();params=[p for p in model.parameters() if p.requires_grad]
    dump(out/'trainable_parameters.json',{n:p.numel() for n,p in model.named_parameters() if p.requires_grad})
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['epsilon'],weight_decay=cfg['weight_decay'])
    trainrows=data.rows('train');micro=cfg['micro_batch_size'];acc=cfg['gradient_accumulation'];n_batches=math.ceil(len(trainrows)/micro)
    total_steps=cfg['epochs']*math.ceil(n_batches/acc);warmup=max(1,math.ceil(total_steps*cfg['warmup_fraction']))
    def factor(step):return step/warmup if step<warmup else .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor);scaler=torch.cuda.amp.GradScaler(enabled=False)
    progress=dict(epoch=0,next_batch=0,step=0,best=float('inf'),sums=[0.,0.,0.,0.])
    if resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);progress=restore_training_state(state,model,optimizer,scheduler,scaler)
    else:
        print('Evaluating raw model on fixed Validation masks...',flush=True)
        val,table=evaluate(model,data,cfg);dump(out/'raw_validation.json',val);table.to_csv(out/'raw_functional_validation.csv',index=False)
        print('RAW_VALIDATION',val,flush=True)
        save_training_state(out/'last.pt',model,optimizer,scheduler,scaler,progress)
    for epoch in range(progress['epoch'],cfg['epochs']):
        model.train();order=np.random.default_rng(cfg['seed']+epoch).permutation(trainrows)
        start=progress['next_batch'];sums=np.array(progress['sums'],float)
        for first in range(start,n_batches,acc):
            group=[data.batch(order[j*micro:(j+1)*micro],cfg['seed']+epoch*100000+j) for j in range(first,min(first+acc,n_batches))]
            denominator=sum(int(b['labels'].ne(-100).sum()) for b in group)
            optimizer.zero_grad(set_to_none=True)
            for cpu in group:
                b=to_device(cpu);n=int(b['labels'].ne(-100).sum())
                with autocast(torch.device('cuda'),cfg):o=masked_outputs(model,b);loss=o['loss']*(n/denominator)
                
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();sums+=np.array([o['loss'].item()*n,o['id'].item()*n,o['expression'].item()*n,n])
                del o,loss,b
            status=step_optimizer(optimizer,scaler,params,cfg['max_grad_norm']);scheduler.step()
            progress.update(epoch=epoch,next_batch=min(first+acc,n_batches),step=progress['step']+1,sums=sums.tolist())
            if progress['step']%cfg['log_every_steps']==0:
                print(f"epoch={epoch+1} step={progress['step']} batch={progress['next_batch']}/{n_batches} loss={sums[0]/sums[3]:.6f} id={sums[1]/sums[3]:.6f} expression={sums[2]/sums[3]:.6f} lr={scheduler.get_last_lr()[0]:.5g} grad_norm={status['grad_norm']:.5g}",flush=True)
            if progress['step']%cfg['save_every_steps']==0:save_training_state(out/'last.pt',model,optimizer,scheduler,scaler,progress)
        val,table=evaluate(model,data,cfg);dump(out/f'validation_epoch{epoch+1:02d}.json',val);table.to_csv(out/f'functional_validation_epoch{epoch+1:02d}.csv',index=False)
        row=dict(epoch=epoch+1,step=progress['step'],train_loss=sums[0]/sums[3],train_id=sums[1]/sums[3],train_expression=sums[2]/sums[3],val_loss=val['loss'],val_id=val['id'],val_expression=val['expression'],val_top1=val['top1'])
        path=out/'metrics.csv';history=pd.read_csv(path) if path.exists() else pd.DataFrame();history=history[history.epoch<epoch+1] if len(history) else history
        pd.concat([history,pd.DataFrame([row])]).to_csv(out/'metrics.partial.csv',index=False);(out/'metrics.partial.csv').replace(path)
        improved=val['loss']<progress['best'];progress.update(epoch=epoch+1,next_batch=0,sums=[0.,0.,0.,0.],best=min(progress['best'],val['loss']))
        if improved:save_training_state(out/'best.pt',model,optimizer,scheduler,scaler,progress)
        save_training_state(out/'last.pt',model,optimizer,scheduler,scaler,progress)
        print('EPOCH',row,'best',progress['best'],flush=True)
    dump(out/'complete.json',dict(epochs=cfg['epochs'],optimizer_steps=progress['step'],best_validation_loss=progress['best'],test_evaluated=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=DEFAULT);p.add_argument('--resume',action='store_true');a=p.parse_args();cfg=config(a.config)
    torch.set_num_threads(cfg['cpu_threads'])
    with threadpool_limits(cfg['cpu_threads']):train(cfg,a.resume)
