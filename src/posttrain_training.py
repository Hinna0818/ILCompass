"""训练/独立测试共用的固定掩码、loss 和分组织评价。"""
from contextlib import nullcontext
import json
import pickle
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk

from posttrain_data import file_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/GeneCompass"))
from genecompass.data_collator import DataCollatorForLanguageModelingModified
from genecompass.pretrainer import GenecompassPreCollator


def load_split(data, split):
    report = json.loads((data / "prepare_report.json").read_text())
    if file_hash(data / f"{split}_cell_map.tsv.gz") != report["cell_map_sha256"][split]:
        raise ValueError(f"{split}: 锁定的 cell map 发生变化")
    ds = load_from_disk(str(data / split))
    meta = pd.read_csv(data / f"{split}_cell_map.tsv.gz", sep="\t", keep_default_na=False)
    if len(ds) != len(meta) or not meta.cell_uid.is_unique or not meta.split.eq(split).all():
        raise ValueError(f"{split}: 编码行与 cell map 不一致")
    if (meta.groupby("library_id").tissue.nunique() != 1).any():
        raise ValueError("同一文库映射到多个组织，需要先审核 metadata")
    return ds, meta


def check_prepared(cfg):
    data = ROOT / cfg["prepared_dataset_path"]
    report = json.loads((data / "prepare_report.json").read_text())
    if report["config"]["dataset_version"] != cfg["dataset_version"]:
        raise ValueError("训练配置与数据版本不一致")
    token_hash = file_hash(ROOT / cfg["token_dictionary"])
    if token_hash != report["token_dictionary_sha256"]:
        raise ValueError("词表与编码阶段不一致")
    return data, report


def make_collator(cfg):
    tokens = pickle.load(open(ROOT / cfg["token_dictionary"], "rb"))
    collator = DataCollatorForLanguageModelingModified(
        tokenizer=GenecompassPreCollator(token_dictionary=tokens), mlm=True,
        mlm_probability=cfg["mlm_probability"])
    if cfg.get("tissue_auxiliary"):
        from tissue_loss_model.multitask import TissueCollator
        collator = TissueCollator(collator)
    return collator


def batches(ds, idx, collator, batch_size, seed, device, shuffle=False, start_batch=0):
    idx = np.asarray(idx)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    for number, start in enumerate(range(start_batch * batch_size, len(idx), batch_size), start_batch):
        rows = ds[idx[start:start + batch_size].tolist()]
        keys = ("input_ids", "values", "species") + (("tissue_labels",) if "tissue_labels" in rows else ())
        ex = [{k: rows[k][i] for k in keys}
              for i in range(len(rows["input_ids"]))]
        # 隔离 collator 的 CPU RNG：验证不会重设训练 dropout/CUDA 的 RNG。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + number)
            b = collator(ex)
        if not (b["labels"] != -100).any():
            raise ValueError("当前 batch 没有预测位置")
        yield {k: v.to(device) for k, v in b.items() if v is not None}


def save_training_state(path, model, optimizer, scheduler, scaler, progress):
    """仅保存可训练权重；冻结权重由同一 Base 提供。原子替换保留上一个有效断点。"""
    state = {
        "model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "progress": progress,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    partial = path.with_suffix(".partial.pt")
    torch.save(state, partial)
    partial.replace(path)


def restore_training_state(state, model, optimizer, scheduler, scaler):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state["model"]) != expected:
        raise ValueError("断点可训练参数与当前模型不一致")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in expected:
                p.copy_(state["model"][name])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["progress"]


def autocast(device, cfg):
    if device.type == "cuda" and cfg["precision"] != "float32":
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg["precision"]]
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def loss_coefficients(cfg, scales):
    if cfg["loss_mode"] == "normalized":
        keys = ("id", "expression", "tissue") if cfg.get("tissue_auxiliary") else ("id", "expression")
        return {k: cfg["loss_weights"][k] / scales[k] for k in keys}
    if cfg["loss_mode"] == "official":
        return {"id": 0.2, "expression": 0.8}
    raise ValueError(f"未知 loss_mode: {cfg['loss_mode']}")


def task_losses(out, cfg):
    losses = {"id": out.id_loss.float(), "expression": out.value_loss.float()}
    if cfg.get("tissue_auxiliary"):
        losses["tissue"] = out.tissue_loss.float()
    return losses


def training_loss(out, cfg, scales):
    coeff = loss_coefficients(cfg, scales)
    return sum(coeff[k] * value for k, value in task_losses(out, cfg).items())


def initial_loss_scales(report, cfg):
    scales = {k: report["overall"][k] for k in ("id", "expression")}
    if cfg.get("tissue_auxiliary"):
        scales["tissue"] = report["tissue_classification"]["unmasked"]["macro_ce"]
    return scales


def step_optimizer(optimizer, scaler, params, max_norm):
    """先让 GradScaler 检出溢出；只裁剪有限梯度，溢出组不更新参数。"""
    scale_before = scaler.get_scale()
    scaler.unscale_(optimizer)
    finite = torch.stack([torch.isfinite(p.grad).all() for p in params if p.grad is not None]).all().item()
    if not finite and not scaler.is_enabled():
        raise FloatingPointError("未启用 FP16 GradScaler 时出现非有限梯度；停止以保护模型参数")
    grad_norm = None
    if finite:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True).item()
    # unscale_ 已记录 inf/NaN；溢出时 step 自动跳过，update 降低下一组的 scale。
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scale_after = scaler.get_scale()
    if not np.isfinite(scale_after) or scale_after <= 0:
        raise FloatingPointError("AMP scale 已无效；请检查数值或显式选择其他 precision")
    return {"stepped": bool(finite), "grad_norm": grad_norm,
            "amp_scale_before": scale_before, "amp_scale_after": scale_after}


def metric_score(metrics, scales):
    # 所有配方共用此选模指标；各组织先汇总预测位置，再给组织相同权重。
    return 0.5 * metrics["id"] / scales["id"] + 0.5 * metrics["expression"] / scales["expression"]


def aggregate(rows, scales):
    n = sum(x["masked_positions"] for x in rows)
    if n == 0:
        raise ValueError("评价集合没有预测位置")
    result = {k: sum(x[k] * x["masked_positions"] for x in rows) / n
              for k in ("id", "expression", "top1")}
    result["masked_positions"] = n
    result["cells"] = sum(x["cells"] for x in rows)
    result["official_loss"] = 0.2 * result["id"] + 0.8 * result["expression"]
    if scales is not None:
        result["normalized_score"] = metric_score(result, scales)
    return result


@torch.no_grad()
def evaluate(model, ds, meta, collator, cfg, device, scales=None, seed=None):
    model.eval()
    by_library = {}
    tissue_targets, tissue_predictions, tissue_errors = [], [], []
    for number, (library, group) in enumerate(meta.groupby("library_id", sort=True)):
        pieces = []
        eval_seed = (cfg["eval_mask_seed"] if seed is None else seed) + number * 100000
        for b in batches(ds, group.index.to_numpy(), collator, cfg["eval_batch_size"], eval_seed, device):
            with autocast(device, cfg):
                out = model(**b)
            if cfg.get("tissue_auxiliary"):
                tissue_targets.extend(b["tissue_labels"].cpu().tolist())
                tissue_predictions.extend(out.tissue_logits.argmax(-1).cpu().tolist())
                tissue_errors.extend(torch.nn.functional.cross_entropy(
                    out.tissue_logits.float(), b["tissue_labels"], reduction="none").cpu().tolist())
            mask = b["labels"] != -100
            n = mask.sum().item()
            pieces.append({"id": out.id_loss.item(), "expression": out.value_loss.item(),
                           "top1": (out.logits.argmax(-1)[mask] == b["labels"][mask]).float().mean().item(),
                           "masked_positions": n, "cells": len(b["input_ids"])})
        by_library[library] = aggregate(pieces, scales) | {"tissue": group.tissue.iloc[0], "study": group.study.iloc[0]}
    by_tissue = {t: aggregate([v for v in by_library.values() if v["tissue"] == t], scales)
                 for t in sorted(meta.tissue.unique())}
    by_study = {s: aggregate([v for v in by_library.values() if v["study"] == s], scales)
                for s in sorted(meta.study.unique())}
    report = {"overall": aggregate(list(by_library.values()), scales), "by_tissue": by_tissue,
              "by_study": by_study, "by_library": by_library}
    if scales is not None:
        report["selection_score"] = float(np.mean([v["normalized_score"] for v in by_tissue.values()]))
    if cfg.get("tissue_auxiliary"):
        from tissue_loss_model.multitask import classification_metrics, evaluate_unmasked
        report["tissue_classification"] = {
            "masked": classification_metrics(tissue_targets, tissue_predictions, tissue_errors, model.info["labels"]),
            "unmasked": evaluate_unmasked(model, ds, meta, cfg, device)}
        if scales is not None:
            report["mlm_selection_score"] = report["selection_score"]
            coeff = loss_coefficients(cfg, scales)
            report["selection_score"] = float(sum(coeff[k] * np.mean([v[k] for v in by_tissue.values()])
                                                   for k in ("id", "expression"))
                                              + coeff["tissue"] * report["tissue_classification"]["unmasked"]["macro_ce"])
        if not all(np.isfinite(v["macro_ce"]) for v in report["tissue_classification"].values()):
            raise ValueError("Tissue validation produced nonfinite loss")
    if not all(np.isfinite(v[k]) for v in by_library.values() for k in ("id", "expression", "top1")):
        raise ValueError("评价出现非有限数值")
    return report


def calibration_indices(meta, per_library, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(g.index, min(len(g), per_library), replace=False)
                                   for _, g in meta.groupby("library_id", sort=True)]))


def freeze_for_posttrain(model, prefixes):
    for p in model.parameters():
        p.requires_grad = False
    matched = {prefix: [] for prefix in prefixes}
    for name, p in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix):
                p.requires_grad = True
                matched[prefix].append(p)
    if any(not p for p in matched.values()):
        raise ValueError("可训练参数前缀未匹配")
    # ID decoder 与输入词嵌入 tied；不能因训练预测头而解冻整个基因坐标。
    if model.bert.embeddings.word_embeddings.weight.requires_grad:
        raise ValueError("输入 gene embedding 必须冻结")
    return [p for p in model.parameters() if p.requires_grad]


def gradient_diagnostics(model, out, cfg, scales):
    shared = [p for p in model.bert.encoder.parameters() if p.requires_grad]
    result = {}
    coeff = loss_coefficients(cfg, scales)
    for key, loss in task_losses(out, cfg).items():
        gradients = torch.autograd.grad(coeff[key] * loss.float(), shared, retain_graph=True)
        result[key] = float(torch.sqrt(sum(g.float().square().sum() for g in gradients)).item())
    result["expression_over_id"] = result["expression"] / max(result["id"], 1e-12)
    if cfg.get("tissue_auxiliary"):
        result["tissue_over_id"] = result["tissue"] / max(result["id"], 1e-12)
    return result
