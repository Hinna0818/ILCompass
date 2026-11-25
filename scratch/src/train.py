"""Train E1b on frozen tokens. No automatic final-Test evaluation.

From scratch/: python src/train.py --config configs/training_e1b_v03.json
Checkpoints contain trusted local Python/NumPy state; do not load untrusted files.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch

from input_pipeline import ROOT, build_loader
from model import ILC2Transformer, ModelConfig, MODEL_INPUTS, mlm_objective

sys.path.append(str(ROOT/"src"))
from posttrain_data import file_hash


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def autocast_context(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()


def move_batch(batch, device):
    needed = (*MODEL_INPUTS, "labels", "value_labels", "target_mask", "corruption_type")
    return {k: batch[k].to(device, non_blocking=True) for k in needed}


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg["learning_rate"], betas=cfg["adam_betas"], eps=cfg["adam_eps"])


def make_scheduler(optimizer, cfg, total_steps):
    warmup = max(1, round(total_steps * cfg["warmup_fraction"]))
    floor = cfg["min_learning_rate"] / cfg["learning_rate"]

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class Metrics:
    """Cell-macro objective plus token-micro, branch, gene and source diagnostics."""
    def __init__(self, n_genes):
        self.cells = self.valid_cells = self.tokens = self.padded_tokens = self.batches = 0
        self.id_sum = self.value_sum = 0.0
        self.branches = np.zeros((4, 4), dtype=np.float64)  # count, CE, SSE, correct
        self.genes = np.zeros((n_genes, 4), dtype=np.float64)
        self.groups = {}

    def update(self, output, cpu_batch, frame):
        self.cells += len(cpu_batch["cell_ids"])
        self.valid_cells += int(output["valid_cells"])
        self.tokens += int(cpu_batch["lengths"].sum())
        self.padded_tokens += cpu_batch["input_ids"].numel()
        self.batches += 1
        self.id_sum += float(output["id_sum"].detach())
        self.value_sum += float(output["value_sum"].detach())
        labels = output["labels"].cpu().numpy()
        kinds = output["branches"].cpu().numpy()
        values = np.column_stack([np.ones(len(labels)), output["id_losses"].cpu().numpy(),
                                  output["value_losses"].cpu().numpy(), output["correct"].float().cpu().numpy()])
        np.add.at(self.genes, labels, values)
        np.add.at(self.branches, kinds, values)
        per_cell = np.column_stack([(output["target_counts"] > 0).cpu().numpy(),
                                    output["id_per_cell"].detach().cpu().numpy(),
                                    output["value_per_cell"].detach().cpu().numpy()])
        rows = frame.iloc[cpu_batch["sample_indices"].numpy()]
        for dimension in ("study", "tissue"):
            for name in rows[dimension].unique():
                key = f"{dimension}:{name}"
                self.groups.setdefault(key, np.zeros(3))
                self.groups[key] += per_cell[rows[dimension].eq(name).to_numpy()].sum(0)

    def summary(self, cfg):
        n = self.valid_cells
        targets, ce, sse, correct = self.genes.sum(0)
        return dict(cells=self.cells, valid_cells=n, empty_target_cells=self.cells-n,
                    tokens=self.tokens, targets=int(targets), batches=self.batches,
                    padding_fraction=1-self.tokens/max(1,self.padded_tokens),
                    id_macro=self.id_sum/n if n else None, value_macro=self.value_sum/n if n else None,
                    loss=(cfg["id_loss_weight"]*self.id_sum+cfg["value_loss_weight"]*self.value_sum)/n if n else None,
                    id_micro=ce/targets if targets else None, value_micro=sse/targets if targets else None,
                    id_accuracy=correct/targets if targets else None)

    def save_details(self, prefix, vocab, cfg):
        for name, array, metadata in (
            ("genes", self.genes, vocab[["gene_id", "gene_symbol", "training_positive_count", "is_functional67"]].copy()),
            ("branches", self.branches[1:], pd.DataFrame({"branch": ["MASK", "random", "unchanged"]})),
        ):
            count = array[:, 0]
            metadata["targets"] = count.astype(np.int64)
            for j, label in enumerate(("id_ce", "value_mse", "id_accuracy"), 1):
                metadata[label] = np.divide(array[:, j], count, out=np.full(len(count), np.nan), where=count>0)
            metadata.to_csv(str(prefix)+f"_{name}.tsv", sep="\t", index=False)
        groups = []
        for key, (n, ce, mse) in sorted(self.groups.items()):
            groups.append(dict(group=key, valid_cells=int(n), id_macro=ce/n if n else None,
                               value_macro=mse/n if n else None,
                               loss=(cfg["id_loss_weight"]*ce+cfg["value_loss_weight"]*mse)/n if n else None))
        pd.DataFrame(groups).to_csv(str(prefix)+"_groups.tsv", sep="\t", index=False)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    loader.batch_sampler.set_epoch(0)
    metrics = Metrics(model.config.n_genes)
    for cpu_batch in loader:
        with autocast_context(device, cfg["precision"]):
            output = mlm_objective(model, move_batch(cpu_batch, device))
        metrics.update(output, cpu_batch, loader.dataset.frame)
    if metrics.valid_cells == 0 or not math.isfinite(metrics.summary(cfg)["loss"]):
        raise ValueError("Evaluation has no targets or non-finite loss")
    return metrics


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, state, cfg, input_cfg, provenance, assets):
    # Called only after an optimizer window: no half-accumulated gradients to lose.
    atomic_save(dict(model=model.state_dict(), model_config=asdict(model.config), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), state=state, rng=rng_state(), config=cfg,
                     input_config=input_cfg, provenance=provenance, frozen_assets=assets), path)


def load_training_checkpoint(path, model, optimizer, scheduler, cfg, input_cfg, provenance):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != cfg or checkpoint["input_config"] != input_cfg or checkpoint["provenance"] != provenance:
        raise ValueError("Resume requires identical training config, code and frozen data artifacts")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng(checkpoint["rng"])
    return checkpoint["state"]


def frozen_inputs(input_cfg):
    prepared = (ROOT / input_cfg["tokenized"]).parent
    names = ["config.json", "gene_vocab.tsv", "gene_medians_train.tsv", "train_gene_detection.tsv",
             "split_manifest.tsv.gz", "sample_condition_manifest.tsv", "condition_manifest.tsv.gz",
             "source_manifest.tsv", "tokenized_shards_v1/config.json", "tokenized_shards_v1/cell_manifest.tsv.gz",
             "tokenized_shards_v1/shard_manifest.tsv", "fixed_eval_masks_v1/config.json"]
    files = [prepared / name for name in names]
    files += [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("input_pipeline.py"),
              Path(__file__).with_name("token_data.py")]
    provenance = {str(path.relative_to(ROOT)): file_hash(path) for path in files}
    assets = {name: (prepared/name).read_bytes() for name in names[:9]}
    return prepared, provenance, assets


def train(model, train_loader, val_loader, vocab, cfg, input_cfg, device, provenance, assets, resume=None):
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    if not resume and ((out/"last.pt").exists() or (out/"run.json").exists()):
        raise FileExistsError("Run already exists; use --resume or a new config/output")
    optimizer = make_optimizer(model, cfg)
    total_steps = math.ceil(train_loader.batch_sampler.total_batches/cfg["accumulation_steps"])*cfg["epochs"]
    scheduler = make_scheduler(optimizer, cfg, total_steps)
    state = dict(epoch=0, global_step=0, best=float("inf"), stale=0, sampler=None,
                 train_metrics=None, finished=False, elapsed_seconds=0.0)
    if resume:
        state = load_training_checkpoint(resume, model, optimizer, scheduler, cfg, input_cfg, provenance)
    if state["finished"]:
        print("Run already completed (epoch budget or early stopping); no updates performed.", flush=True)
        return
    run = dict(config=cfg, input_config=input_cfg, model_config=asdict(model.config), provenance=provenance,
               parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
               device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
               total_planned_optimizer_steps=total_steps, split_scope="internal cell holdout; no automatic Test scoring")
    (out/"run.json").write_text(json.dumps(run, indent=2)+"\n")
    print(json.dumps(run, indent=2), flush=True)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"Signal {signum}: saving at the next completed accumulation window.", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    for epoch in range(state["epoch"], cfg["epochs"]):
        model.train()
        sampler = train_loader.batch_sampler
        if state["sampler"] is not None:
            sampler.load_state_dict(state["sampler"])
        else:
            sampler.set_epoch(epoch)
        metrics = Metrics(model.config.n_genes)
        if state["train_metrics"] is not None:
            # Plain arrays/dicts keep checkpoints loadable outside this CLI module.
            metrics.__dict__.update(state["train_metrics"])
        iterator = iter(train_loader)
        consumed = 0
        elapsed_base = state["elapsed_seconds"]
        start = time.perf_counter()
        while True:
            window = list(itertools.islice(iterator, cfg["accumulation_steps"]))
            if not window:
                break
            n_cells = sum(int(b["loss_cell_mask"].sum()) for b in window)
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch in window:
                with autocast_context(device, cfg["precision"]):
                    result = mlm_objective(model, move_batch(cpu_batch, device))
                    loss = (cfg["id_loss_weight"]*result["id_sum"]+cfg["value_loss_weight"]*result["value_sum"])/max(1,n_cells)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: epoch={epoch+1}, source_rows={cpu_batch['source_rows'].tolist()}")
                if n_cells:
                    loss.backward()
                metrics.update(result, cpu_batch, train_loader.dataset.frame)
            gradient_norm = 0.0
            used_lr = optimizer.param_groups[0]["lr"]
            if n_cells:
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True))
                optimizer.step()
                scheduler.step()
                state["global_step"] += 1
            optimizer.zero_grad(set_to_none=True)
            consumed += len(window)
            state.update(epoch=epoch, sampler=sampler.state_dict(consumed), train_metrics=metrics.__dict__,
                         elapsed_seconds=elapsed_base+time.perf_counter()-start)
            if state["global_step"] % cfg["log_every_steps"] == 0 or stop["requested"]:
                summary = metrics.summary(cfg)
                print(f"epoch={epoch+1} step={state['global_step']} batch={state['sampler']['start_batch']}/{sampler.total_batches} "
                      f"loss={summary['loss']} id={summary['id_macro']} value={summary['value_macro']} "
                      f"lr={used_lr:.6g} grad_norm={gradient_norm:.5g} pad={summary['padding_fraction']:.3%} "
                      f"max_length={max(int(b['lengths'].max()) for b in window)}", flush=True)
            if state["global_step"] % cfg["checkpoint_every_steps"] == 0 or stop["requested"]:
                save_checkpoint(out/"last.pt",model,optimizer,scheduler,state,cfg,input_cfg,provenance,assets)
            if stop["requested"]:
                print("Stopped safely; resume from last.pt.", flush=True)
                return
        # Save before validation too, so an interrupted evaluation can simply be repeated.
        save_checkpoint(out/"last.pt",model,optimizer,scheduler,state,cfg,input_cfg,provenance,assets)
        val_metrics = evaluate(model,val_loader,device,cfg)
        train_summary, val_summary = metrics.summary(cfg), val_metrics.summary(cfg)
        improved = val_summary["loss"] < state["best"]
        state["best"] = min(state["best"],val_summary["loss"])
        state["stale"] = 0 if improved else state["stale"]+1
        prefix = out/f"epoch_{epoch+1:03d}"
        metrics.save_details(str(prefix)+"_train",vocab,cfg)
        val_metrics.save_details(str(prefix)+"_validation",vocab,cfg)
        record=dict(epoch=epoch+1,step=state["global_step"],train=train_summary,validation=val_summary,
                    best=state["best"],stale=state["stale"],elapsed_training_seconds=state["elapsed_seconds"])
        Path(str(prefix)+".json").write_text(json.dumps(record,indent=2)+"\n")
        rows=[]
        for path in sorted(out.glob("epoch_*.json")):
            entry=json.loads(path.read_text())
            for split in ("train","validation"):
                rows.append(dict(epoch=entry["epoch"],step=entry["step"],split=split,**entry[split]))
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
        state.update(epoch=epoch+1,sampler=None,train_metrics=None,
                     finished=epoch+1>=cfg["epochs"] or state["stale"]>=cfg["early_stopping_patience"])
        save_checkpoint(out/"last.pt",model,optimizer,scheduler,state,cfg,input_cfg,provenance,assets)
        if improved:
            save_checkpoint(out/"best.pt",model,optimizer,scheduler,state,cfg,input_cfg,provenance,assets)
        print(f"epoch={epoch+1} train={train_summary['loss']:.7g} val={val_summary['loss']:.7g} "
              f"best={state['best']:.7g} stale={state['stale']}",flush=True)
        if state["finished"] or stop["requested"]:
            break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"scratch/configs/training_e1b_v03.json")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--evaluate",type=Path,help="Explicit checkpoint-only scoring; no optimization")
    parser.add_argument("--split",choices=["validation","test"],default="validation")
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args()
    if args.resume and args.evaluate:
        parser.error("--resume and --evaluate are mutually exclusive")
    cfg=json.loads(args.config.read_text())
    input_cfg=json.loads((ROOT/cfg["input_config"]).read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    device=torch.device(args.device)
    if device.type=="cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("A BF16-capable CUDA GPU is required; CPU is only for small verification runs")
    if cfg["precision"] not in ("bf16","fp32") or cfg["accumulation_steps"]<1:
        raise ValueError("Invalid precision or gradient accumulation")
    prepared,provenance,assets=frozen_inputs(input_cfg)
    vocab=pd.read_csv(prepared/"gene_vocab.tsv",sep="\t")
    seed_all(cfg["seed"])
    model=ILC2Transformer(ModelConfig(n_genes=len(vocab),**cfg["model"])).to(device)
    if args.evaluate:
        checkpoint=torch.load(args.evaluate,map_location="cpu",weights_only=False)
        if checkpoint["provenance"]!=provenance or checkpoint["model_config"]!=asdict(model.config) or checkpoint["config"]!=cfg or checkpoint["input_config"]!=input_cfg:
            raise ValueError("Evaluation checkpoint/config/frozen data mismatch")
        model.load_state_dict(checkpoint["model"])
        metrics=evaluate(model,build_loader(args.split,input_cfg),device,cfg)
        prefix=args.evaluate.parent/f"{args.evaluate.stem}_{args.split}"
        metrics.save_details(prefix,vocab,cfg)
        Path(str(prefix)+".json").write_text(json.dumps(metrics.summary(cfg),indent=2)+"\n")
        print(json.dumps(metrics.summary(cfg),indent=2),flush=True)
    else:
        gate=json.loads((ROOT/cfg["training_acceptance"]).read_text())
        if not gate["passed"] or gate["provenance"]!=provenance or gate["model_config"]!=asdict(model.config) or gate["input_config"]!=input_cfg:
            raise ValueError("Run validate_training.py again: full training acceptance is missing or stale")
        if gate["precision"]!=cfg["precision"] or gate["accumulation_steps"]!=cfg["accumulation_steps"]:
            raise ValueError("Precision/accumulation changed after GPU acceptance")
        train(model,build_loader("train",input_cfg),build_loader("validation",input_cfg),vocab,cfg,input_cfg,
              device,provenance,assets,resume=args.resume)


if __name__=="__main__":
    main()
