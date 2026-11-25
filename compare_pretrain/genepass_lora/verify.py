"""GPU acceptance only: updates are discarded, never saved as a training checkpoint."""
from common import *
import argparse,hashlib,random,tempfile,time
import numpy as np
import torch
from model import build,masked_outputs,LoRALinear
from data import Inputs
from posttrain_training import autocast,save_training_state,restore_training_state,step_optimizer

def digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main(cfg):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Inputs(cfg);model=build(cfg).cuda();model.eval()
    train=data.rows('train');rows=train[np.argsort(data.lengths[train])[-cfg['micro_batch_size']:]]
    b=data.batch(rows,cfg['eval_mask_seed'],validation=True)
    again=data.batch(rows,cfg['eval_mask_seed'],validation=True)
    assert all(torch.equal(b[k],again[k]) for k in b)
    one=data.batch(rows[:1],cfg['eval_mask_seed'],validation=True)
    assert torch.equal(one['labels'],b['labels'][:1,:one['labels'].shape[1]])
    b={k:v.cuda() for k,v in b.items()}
    short={k:(v[:1,:128] if k!='species' else v[:1]) for k,v in b.items()}
    with torch.no_grad():
        kwargs={k:short[k] for k in ['input_ids','values','attention_mask','species']}
        adapted=model.bert(**kwargs,emb_warmup_alpha=1.,return_dict=True).last_hidden_state
        modules=[]
        for layer in model.bert.encoder.layer:
            for name in ['query','value']:
                m=getattr(layer.attention.self,name);modules.append((layer.attention.self,name,m));setattr(layer.attention.self,name,m.base)
        raw=model.bert(**kwargs,emb_warmup_alpha=1.,return_dict=True).last_hidden_state
        for parent,name,m in modules:setattr(parent,name,m)
        initial=float((adapted-raw).abs().max());assert initial<1e-6
        o=masked_outputs(model,short);mask=short['labels'].ne(-100)
        all_logits=model.cls(raw[:,1:]).float();all_values=model.cls4value(raw[:,1:]).squeeze(-1).float()
        ce=torch.nn.functional.cross_entropy(all_logits.reshape(-1,model.config.vocab_size),short['labels'].reshape(-1))
        mse=torch.nn.functional.mse_loss(all_values[mask],short['labels_values'][mask])
        ce_error=float((ce-o['id']).abs());mse_error=float((mse-o['expression']).abs())
        assert ce_error<1e-5 and mse_error<1e-5
    del adapted,raw,all_logits,all_values,o
    frozen_before=digest(model);params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==294912 and len(params)==48
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    model.train();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    def update():
        optimizer.zero_grad(set_to_none=True)
        with autocast(torch.device('cuda'),cfg):o=masked_outputs(model,b)
        loss=float(o['loss']);o['loss'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        status=step_optimizer(optimizer,scaler,params,cfg['max_grad_norm']);scheduler.step();return loss,status
    loss1,status=update()
    assert any(p.detach().abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.B'))
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'resume.pt';save_training_state(path,model,optimizer,scheduler,scaler,{'step':1})
        loss2,_=update();expected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        state=torch.load(path,map_location='cpu',weights_only=False);restore_training_state(state,model,optimizer,scheduler,scaler)
        replay,_=update();resume_error=max(float((p.detach().cpu()-expected[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad)
        assert abs(loss2-replay)<1e-6 and resume_error<1e-7
    update_seconds=time.monotonic()-start
    assert digest(model)==frozen_before
    from train import evaluate
    validation, functional=evaluate(model,data,cfg,rows=data.rows('validation')[:8])
    assert validation['cells']==8 and np.isfinite(validation['loss'])
    output=ROOT/'results/ILC2_GeneCompass_LoRA_v1/acceptance'
    dump(output/'report.json',dict(passed=True,trainable_parameters=sum(p.numel() for p in params),frozen_unchanged=True,zero_adapter_alignment=initial,masked_head_CE_error=ce_error,masked_head_MSE_error=mse_error,fixed_mask_reproduced=True,padding_not_target=True,resume_parameter_error=resume_error,resume_loss_error=abs(loss2-replay),micro_batch=len(rows),max_input_length=b['input_ids'].shape[1],precision=cfg['precision'],peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,three_optimizer_updates_seconds=update_seconds,first_loss=loss1,grad_norm=status['grad_norm'],diagnostic_weights_discarded=True,formal_training_started=False,validation_loop_smoke_cells=validation['cells']))
    print((output/'report.json').read_text(),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=DEFAULT);args=p.parse_args();cfg=config(args.config);torch.set_num_threads(cfg['cpu_threads']);main(cfg)
