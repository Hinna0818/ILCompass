"""Short real-Train GPU acceptance, including backward/AdamW and checkpoint replay.

Weights are diagnostic and discarded. This is not formal pretraining or a speed
claim for an entire epoch. Never scores Test or changes frozen preparation files.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from input_pipeline import ROOT, build_loader
from model import ILC2Transformer, ModelConfig, MODEL_INPUTS, mlm_objective
from train import (seed_all, make_optimizer, make_scheduler, autocast_context, move_batch,
                   save_checkpoint, load_training_checkpoint, frozen_inputs)


def update(model, optimizer, batches, cfg, device):
    optimizer.zero_grad(set_to_none=True)
    cells=sum(int(b['loss_cell_mask'].sum()) for b in batches)
    if not cells:raise ValueError("Acceptance batches have no targets")
    total=0.0
    for batch in batches:
        with autocast_context(device,cfg['precision']):
            result=mlm_objective(model,move_batch(batch,device))
            loss=(cfg['id_loss_weight']*result['id_sum']+cfg['value_loss_weight']*result['value_sum'])/cells
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite acceptance loss")
        loss.backward()
        total+=float(loss.detach())
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total,grad


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'scratch/configs/training_e1b_v03.json')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    input_cfg=json.loads((ROOT/cfg['input_config']).read_text())
    out=(ROOT/cfg['training_acceptance']).parent
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU acceptance requires CUDA BF16")
    device=torch.device('cuda')
    seed_all(cfg['seed'])
    prepared,provenance,assets=frozen_inputs(input_cfg)
    vocab=pd.read_csv(prepared/'gene_vocab.tsv',sep='\t')
    model=ILC2Transformer(ModelConfig(n_genes=len(vocab),**cfg['model'])).to(device).train()
    optimizer=make_optimizer(model,cfg)
    scheduler=make_scheduler(optimizer,cfg,100)
    loader=build_loader('train',input_cfg)
    data=loader.dataset
    order=np.argsort(data.lengths,kind='stable')
    size=input_cfg['batch_size']
    cases={}
    for label,quantile in [('p10',.1),('p50',.5),('p95',.95),('longest',1.)]:
        end=max(size,min(len(order),int(round(quantile*len(order)))))
        indices=order[end-size:end]
        cases[label]=loader.collate_fn([data[(int(i),0)] for i in indices])
    measurements=[]
    for label,batch in cases.items():
        update(model,optimizer,[batch],cfg,device)  # Includes optimizer-state allocation/warmup.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(2):loss,grad=update(model,optimizer,[batch],cfg,device)
        torch.cuda.synchronize()
        seconds=(time.perf_counter()-start)/2
        row=dict(case=label,cells=len(batch['cell_ids']),lengths=batch['lengths'].tolist(),
                 tokens=int(batch['lengths'].sum()),targets=int(batch['target_counts'].sum()),
                 seconds_per_update=seconds,cells_per_second=len(batch['cell_ids'])/seconds,
                 tokens_per_second=int(batch['lengths'].sum())/seconds,
                 peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                 peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,loss=loss,gradient_norm=grad)
        measurements.append(row)
        print(json.dumps(row),flush=True)
    # Profile an actual long-batch backward to identify the dispatched SDPA backend.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        update(model,optimizer,[cases['longest']],cfg,device)
    backend_ops=sorted({e.key for e in profile.key_averages() if 'scaled_dot_product' in e.key})

    # Real multiworker loading and one complete accumulation window (weights discarded).
    iterator=iter(loader)
    batches=[next(iterator) for _ in range(cfg['accumulation_steps'])]
    torch.cuda.synchronize();start=time.perf_counter()
    accumulated_loss,accumulated_grad=update(model,optimizer,batches,cfg,device)
    torch.cuda.synchronize()
    accumulation_seconds=time.perf_counter()-start

    # Replay the next stochastic update, including optimizer, scheduler and GPU RNG.
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        checkpoint=Path(temporary)/'diagnostic.pt'
        state=dict(epoch=0,global_step=1,sampler=loader.batch_sampler.state_dict(len(batches)))
        save_checkpoint(checkpoint,model,optimizer,scheduler,state,cfg,input_cfg,provenance,assets)
        next_batch=next(iterator)
        update(model,optimizer,[next_batch],cfg,device);scheduler.step()
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        expected_lr=scheduler.get_last_lr()
        restored=load_training_checkpoint(checkpoint,model,optimizer,scheduler,cfg,input_cfg,provenance)
        resumed_loader=build_loader('train',input_cfg)
        resumed_loader.batch_sampler.load_state_dict(restored['sampler'])
        replay=next(iter(resumed_loader))
        for key in (*MODEL_INPUTS,'labels','value_labels','target_mask'):
            torch.testing.assert_close(next_batch[key],replay[key],rtol=0,atol=0)
        update(model,optimizer,[replay],cfg,device);scheduler.step()
        max_replay_error=0.0
        for key,value in model.state_dict().items():
            actual=value.detach().cpu()
            torch.testing.assert_close(actual,expected[key],rtol=1e-4,atol=2e-6)
            max_replay_error=max(max_replay_error,float((actual-expected[key]).abs().max()))
        assert scheduler.get_last_lr()==expected_lr

    # Full encoder padding test on real unmasked input, and frozen IDF extraction.
    model.eval()
    sample=data[(int(order[len(order)//2]),0)]
    ids=torch.tensor(np.array(sample['gene_ids'],copy=True),dtype=torch.long,device=device)[None]
    values=torch.tensor(np.array(sample['values'],copy=True),device=device)[None]
    inputs=dict(input_ids=ids,values=values,attention_mask=torch.ones_like(ids,dtype=torch.bool),
                position_ids=torch.arange(ids.shape[1],device=device)[None])
    with torch.no_grad():
        baseline=model(**inputs)
        padded={k:torch.nn.functional.pad(v,(0,17),value=False if k=='attention_mask' else 0) for k,v in inputs.items()}
        actual=model(**padded)
        torch.testing.assert_close(baseline['gene_embeddings'],actual['gene_embeddings'][:,:ids.shape[1]],rtol=1e-4,atol=2e-5)
        torch.testing.assert_close(baseline['cell_embedding'],actual['cell_embedding'],rtol=1e-4,atol=2e-5)
        idf=torch.tensor(vocab.idf_weight.to_numpy(),dtype=torch.float32,device=device)
        pooled=model.pool_idf(baseline['gene_embeddings'],ids,inputs['attention_mask'],idf)
        assert torch.isfinite(pooled).all()
    checks=dict(passed=True,formal_pretraining_started=False,parameters=sum(p.numel() for p in model.parameters()),
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                model_config=asdict(model.config),input_config=input_cfg,provenance=provenance,
                precision=cfg['precision'],accumulation_steps=cfg['accumulation_steps'],
                full_real_longest_batch_backward_optimizer=True,real_max_length=int(data.lengths.max()),
                actual_sdpa_operations=backend_ops,measurements=measurements,
                real_multiworker_accumulation=dict(cells=sum(len(b['cell_ids']) for b in batches),
                    seconds=accumulation_seconds,loss=accumulated_loss,gradient_norm=accumulated_grad),
                checkpoint_next_batch_identical=True,checkpoint_next_update_max_absolute_error=max_replay_error,
                checkpoint_optimizer_scheduler_rng_replayed=True,full_encoder_PAD_invariance=True,
                idf_extraction_finite=True,
                limitations=['Short diagnostic updates; weights discarded, not formal pretraining',
                             'Warm batch timings exclude checkpoint/validation and most input I/O; not an epoch-speed estimate',
                             'CPU tests verify gradient accumulation without dropout; dropout masks differ across batch layouts',
                             'No Test model scoring or downstream validation performed'])
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    pd.DataFrame(measurements).to_csv(out/'gpu_benchmark.csv',index=False)
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
