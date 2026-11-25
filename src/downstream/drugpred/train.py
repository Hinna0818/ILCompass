"""E0 DeSCOPE-like VAE training; development-only preparation and prior validation."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ilc2_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ilc2_mpl")

import argparse
import fcntl
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from models import ROOT, ILC2PerturbationVAE
from prepare import prepare, write_json

CONDITIONS = {"PBS": [0., 0.], "IL33": [1., 0.], "CGRP": [0., 1.]}


def load_cache(cache, split, device):
    obs = pd.read_csv(cache / f"{split}_metadata.tsv", sep="\t")
    h = torch.from_numpy(np.load(cache / f"{split}_h.npy")).to(device)
    y = torch.from_numpy(np.load(cache / f"{split}_y.npy")).to(device)
    if not obs.split.eq(split).all() or len(obs) != len(h) or len(h) != len(y):
        raise ValueError("Cache rows/roles do not agree.")
    if not obs.condition.isin(CONDITIONS).all():
        raise ValueError("Only PBS and single perturbations may enter training/validation.")
    pools = {key: np.flatnonzero(obs.condition.eq(key)) for key in CONDITIONS}
    return h, y, pools


def delta_metrics(pred, truth, control):
    delta_p, delta_t = pred.mean(0) - control, truth.mean(0) - control
    p, t = delta_p - delta_p.mean(), delta_t - delta_t.mean()
    denominator = np.linalg.norm(p) * np.linalg.norm(t)
    return {"delta_rmse": float(np.sqrt(np.mean((delta_p - delta_t) ** 2))),
            "delta_pearson": float(p @ t / denominator) if denominator > 0 else None,
            "gene_variance_rmse": float(np.sqrt(np.mean((pred.var(0) - truth.var(0)) ** 2)))}


@torch.no_grad()
def evaluate(model, data, cfg, projections, quantiles):
    h, y, pools = data
    source_h = h[pools["PBS"]]
    control = y[pools["PBS"]].mean(0).cpu().numpy()
    result = {}
    model.eval()
    # Evaluation sampling must neither advance nor depend on the training RNG.
    with torch.random.fork_rng(devices=[h.device.index] if h.is_cuda else []):
        for condition in CONDITIONS:
            truth = y[pools[condition]].cpu().numpy()
            target_projection = np.quantile(truth @ projections, quantiles, axis=0)
            repeats = []
            for seed in cfg["evaluation_seeds"]:
                torch.manual_seed(seed)
                chunks = []
                for left in range(0, len(source_h), cfg["evaluation_batch_size"]):
                    x = source_h[left:left + cfg["evaluation_batch_size"]]
                    c = torch.tensor(CONDITIONS[condition], device=h.device).expand(len(x), -1)
                    chunks.append(model.predict(x, c).cpu().numpy())
                prediction = np.concatenate(chunks)
                metrics = delta_metrics(prediction, truth, control)
                # Fixed quantile approximation handles unequal cell counts without matching cells.
                metrics["sliced_wasserstein_1"] = float(np.abs(
                    np.quantile(prediction @ projections, quantiles, axis=0) - target_projection).mean())
                metrics["sampling_seed"] = seed
                repeats.append(metrics)
            result[condition] = {"repeats": repeats}
            for metric in ["delta_rmse", "delta_pearson", "gene_variance_rmse", "sliced_wasserstein_1"]:
                vals = [r[metric] for r in repeats if r[metric] is not None]
                result[condition][metric] = float(np.mean(vals)) if vals else None
            result[condition]["delta_rmse_mc_sd"] = float(np.std([r["delta_rmse"] for r in repeats]))
            # Posterior reconstruction is diagnostic only; it never selects a checkpoint.
            target_idx = pools[condition][:cfg["evaluation_batch_size"]]
            source_idx = np.resize(pools["PBS"], len(target_idx))
            if condition == "PBS":
                source_idx = target_idx
            c = torch.tensor(CONDITIONS[condition], device=h.device).expand(len(target_idx), -1)
            posterior = model(h[source_idx], h[target_idx], c, y[target_idx])
            result[condition]["posterior_mse_probe"] = float(posterior["mse"])
            result[condition]["kl_probe"] = float(posterior["kl"])
    result["score"] = (result["IL33"]["delta_rmse"] + result["CGRP"]["delta_rmse"]) / 2
    model.train()
    return result


def head_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}


def save_checkpoint(path, model, optimizer, rng, step, best, bad, cfg, signature):
    partial = path.with_suffix(".partial.pt")
    torch.save({"head_state": head_state(model), "optimizer": optimizer.state_dict(),
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "step": step, "best_score": best,
                "bad_validations": bad, "config": cfg, "cache_signature": signature}, partial)
    partial.replace(path)


def fit(cfg, resume, result_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run; do not silently start long CPU encoding.")
    device = torch.device("cuda:0")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    print(f"START official E0 cVAE seed={cfg['seed']} GPU={torch.cuda.get_device_name(device)}; "
          "C=train, D=validation; combination files excluded", flush=True)
    snapshot = result_dir / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise ValueError("Existing run configuration differs.")
    if (result_dir / "latest.pt").exists() and not resume:
        raise ValueError("Existing training checkpoint; explicitly use --resume.")
    write_json(snapshot, cfg)
    write_json(result_dir / "status.json", {"phase": "preparing", "pid": os.getpid()})
    cache = prepare(cfg, device)
    signature = json.loads((cache / "signature.json").read_text())
    gene_ids = (cache / "gene_ids.txt").read_text().splitlines()
    model = ILC2PerturbationVAE(gene_ids, ROOT / cfg["checkpoint"], beta=cfg["beta"]).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train = load_cache(cache, "train", device)
    validation = load_cache(cache, "validation", device)
    h, y, pools = train
    projection_rng = np.random.default_rng(0)
    projections = projection_rng.normal(size=(len(gene_ids), cfg["sliced_wasserstein_projections"]))
    projections = (projections / np.linalg.norm(projections, axis=0)).astype(np.float32)
    quantiles = (np.arange(cfg["sliced_wasserstein_quantiles"]) + .5) / cfg["sliced_wasserstein_quantiles"]
    np.save(result_dir / "evaluation_projections.npy", projections)
    start, best, bad = 0, math.inf, 0
    if resume and (result_dir / "latest.pt").exists():
        saved = torch.load(result_dir / "latest.pt", map_location=device, weights_only=False)
        if saved["config"] != cfg or saved["cache_signature"] != signature:
            raise ValueError("Resume config/data provenance differs.")
        report = model.load_state_dict(saved["head_state"], strict=False)
        if report.unexpected_keys or any(not k.startswith("encoder.") for k in report.missing_keys):
            raise ValueError(f"Invalid head checkpoint: {report}")
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        start, best, bad = saved["step"], saved["best_score"], saved["bad_validations"]
        print(f"RESUME step={start} best={best:.6f} patience={bad}/{cfg['patience']}", flush=True)
    else:
        control_train = y[pools["PBS"]].mean(0).cpu().numpy()
        control_val = validation[1][validation[2]["PBS"]].mean(0).cpu().numpy()
        baselines = {}
        for condition in ["IL33", "CGRP"]:
            delta = y[pools[condition]].mean(0).cpu().numpy() - control_train
            true_delta = validation[1][validation[2][condition]].mean(0).cpu().numpy() - control_val
            baselines[condition] = {"no_change_delta_rmse": float(np.sqrt(np.mean(true_delta**2))),
                                   "train_mean_effect_delta_rmse": float(np.sqrt(np.mean((delta - true_delta)**2)))}
        write_json(result_dir / "validation_baselines.json", baselines)
        initial = evaluate(model, validation, cfg, projections, quantiles)
        write_json(result_dir / "initial_validation.json", initial)
        print(f"INITIAL prior-only validation score={initial['score']:.6f}", flush=True)
    print(f"TRAIN: G={len(gene_ids)}, trainable={sum(p.numel() for p in parameters):,}, "
          f"batch={cfg['batch_size']} beta={cfg['beta']} max_steps={cfg['max_steps']}", flush=True)
    model.train()
    t0 = time.monotonic()
    step = start
    with (result_dir / "metrics.jsonl").open("a", buffering=1) as metrics_file:
        while step < cfg["max_steps"] and bad < cfg["patience"]:
            step += 1
            condition = "PBS" if rng.random() < cfg["pbs_step_probability"] else rng.choice(["IL33", "CGRP"])
            target = rng.choice(pools[condition], cfg["batch_size"], replace=True)
            source = target if condition == "PBS" else rng.choice(pools["PBS"], cfg["batch_size"], replace=True)
            c = torch.tensor(CONDITIONS[condition], device=device).expand(cfg["batch_size"], -1)
            optimizer.zero_grad(set_to_none=True)
            out = model(h[source], h[target], c, y[target])
            if not torch.isfinite(out["loss"]):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            out["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step == start + 1 or step % cfg["log_every"] == 0:
                record = {"phase": "train", "step": step, "condition": condition,
                          **{key: float(out[key].detach()) for key in ["loss", "mse", "kl", "weighted_kl"]},
                          "kl_per_dimension": float(out["kl"].detach()) / 64, "grad_norm": float(grad),
                          "elapsed_seconds": time.monotonic() - t0}
                metrics_file.write(json.dumps(record) + "\n")
                print(f"TRAIN step={step}/{cfg['max_steps']} {condition} loss={record['loss']:.5f} "
                      f"mse={record['mse']:.5f} kl={record['kl']:.5f} "
                      f"weighted_kl={record['weighted_kl']:.5f} grad={float(grad):.3f}", flush=True)
                write_json(result_dir / "status.json", {"phase": "training", "step": step, "pid": os.getpid()})
            if step % cfg["validation_every"] == 0 or step == cfg["max_steps"]:
                scores = evaluate(model, validation, cfg, projections, quantiles)
                score = scores["score"]
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite validation score")
                improved = score < best
                if improved:
                    best, bad = score, 0
                else:
                    bad += 1
                metrics_file.write(json.dumps({"phase": "validation", "step": step, **scores}) + "\n")
                if improved:
                    save_checkpoint(result_dir / "best.pt", model, optimizer, rng, step, best, bad, cfg, signature)
                    write_json(result_dir / "best_validation.json", {"step": step, **scores})
                save_checkpoint(result_dir / "latest.pt", model, optimizer, rng, step, best, bad, cfg, signature)
                print(f"VALIDATION step={step} prior_delta_rmse={score:.6f} "
                      f"IL33={scores['IL33']['delta_rmse']:.6f} CGRP={scores['CGRP']['delta_rmse']:.6f} "
                      f"best={best:.6f} patience={bad}/{cfg['patience']}", flush=True)
    write_json(result_dir / "status.json", {"phase": "completed", "step": step, "best_score": best,
                                           "reason": "early_stopping" if bad >= cfg["patience"] else "max_steps"})
    print(f"COMPLETED step={step} best_validation={best:.6f}; no combination evaluation performed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/drugpred.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(open(args.config).read())
    result_dir = ROOT / cfg["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fit(cfg, args.resume, result_dir)
        except Exception as exc:
            write_json(result_dir / "status.json", {"phase": "failed", "error": repr(exc), "pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
