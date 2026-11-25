"""ILC2 posttrain：固定尺度双任务 loss、组织等权选模；训练不读取 test。

训练：python src/3finetune_FM.py
无参数更新检查：python src/3finetune_FM.py --check-only --device cpu
资源试跑（用户手动）：python src/3finetune_FM.py --max-updates 20
独立测试另运行 src/evaluate_posttrain.py，不自动用测试集选模型。
"""
import argparse
import json
import math
import random
import tempfile
import time

import numpy as np
import torch

from posttrain_data import file_hash
from posttrain_training import (
    ROOT, autocast, batches, calibration_indices, check_prepared, evaluate,
    freeze_for_posttrain, gradient_diagnostics, load_split, loss_coefficients, make_collator, step_optimizer, training_loss,
    save_training_state, restore_training_state, initial_loss_scales, task_losses,
)
from genecompass_compat import load_genecompass


def write_json(path, obj):
    partial = path.with_suffix(".partial.json")
    partial.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def save_model(model, directory):
    partial = directory / "pytorch_model.partial.bin"
    backbone = getattr(model, "backbone", model)
    torch.save({n: p.detach().cpu() for n, p in backbone.state_dict().items()}, partial)
    partial.replace(directory / "pytorch_model.bin")
    model.config.save_pretrained(directory)
    if hasattr(model, "save_auxiliary"):
        model.save_auxiliary(directory)


def load_training_model(directory, cfg, device, info=None):
    model, load = load_genecompass(directory, device=device)
    if load["missing"] or load["unexpected"]:
        raise ValueError(f"checkpoint 参数不匹配: {load}")
    if cfg.get("tissue_auxiliary"):
        from tissue_loss_model.multitask import attach_tissue_head
        model = attach_tissue_head(model, directory, info, device)
    return model


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def check_only(model, train, meta, collator, cfg, device, result_dir):
    """真实模型完整长度小批次前后向；不调用 optimizer.step，不看 val/test。"""
    params = freeze_for_posttrain(model, cfg["trainable_prefixes"])
    selected = meta.iloc[:cfg.get("preflight_batch_size", 1)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = evaluate(model, train, selected, collator, cfg, device)
    metric = report["overall"]
    scales = initial_loss_scales(report, cfg)
    if min(scales.values()) <= 0:
        raise ValueError("无法由非正基线 loss 构建尺度")
    b = next(batches(train, selected.index, collator, len(selected), cfg["random_seed"], device))
    model.train()
    with autocast(device, cfg):
        out = model(**b)
        loss = training_loss(out, cfg, scales)
    gradients = gradient_diagnostics(model, out, cfg, scales)
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
        raise ValueError("可训练参数缺失梯度或存在非有限梯度")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError("冻结参数产生了梯度")
    result = {"passed": True, "device": str(device), "optimizer_steps": 0,
              "scope": f"{len(selected)} real train cells, full padded sequence; CPU check does not certify GPU memory",
              "sequence_length": b["input_ids"].shape[1], "loss": loss.item(),
              "loss_mode": cfg["loss_mode"], "loss_coefficients": loss_coefficients(cfg, scales),
              "raw_metrics": metric, "shared_gradient_norm": gradients,
              "trainable_params": sum(p.numel() for p in params),
              "test_read": False, "training_started": False}
    if cfg.get("tissue_auxiliary"):
        result["tissue_classification"] = report["tissue_classification"]
    model.zero_grad(set_to_none=True)
    # 同正式训练的完整 checkpoint 保存/重载路径，但只用临时目录，不留下候选模型。
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="ilc2_update_checkpoint_", dir="/tmp") as tmp:
        save_model(model, Path(tmp))
        restored = load_training_model(Path(tmp), cfg, device, getattr(model, "info", None))
        restored_report = evaluate(restored, train, selected, collator, cfg, device)
        restored_metric = restored_report["overall"]
        if any(abs(metric[k] - restored_metric[k]) > 1e-6 for k in ("id", "expression", "top1")):
            raise ValueError("保存/重载后固定掩码指标改变")
        result["checkpoint_save_reload_passed"] = True
        if cfg.get("tissue_auxiliary"):
            for view in ("masked", "unmasked"):
                if abs(report["tissue_classification"][view]["macro_ce"] - restored_report["tissue_classification"][view]["macro_ce"]) > 1e-6:
                    raise ValueError("保存/重载改变了组织分类输出")
            result["tissue_head_save_reload_passed"] = True
    if device.type == "cuda":
        result["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    write_json(result_dir / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(default_config="configs/3finetune_FM.json"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--run-name", help="新实验名；已有正式结果不覆盖")
    parser.add_argument("--loss-mode", choices=["normalized", "official"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-updates", type=int, help="资源试跑，不保存正式模型，也不读取 test")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true", help="从本实验 latest_training_state.pt 完整恢复")
    recovery.add_argument("--resume-from", help="从旧实验最佳权重继续；旧优化器缺失时明确记录重建")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.loss_mode:
        cfg["loss_mode"] = args.loss_mode
    if cfg.get("tissue_auxiliary"):
        if cfg["loss_mode"] != "normalized" or set(cfg["loss_weights"]) != {"id", "expression", "tissue"}:
            raise ValueError("组织辅助实验使用三项固定尺度归一化loss")
        if min(cfg["loss_weights"].values()) <= 0 or not math.isclose(sum(cfg["loss_weights"].values()), 1.):
            raise ValueError("三项loss权重必须为正且和为1")
        if args.resume_from:
            parser.error("组织辅助实验仅支持 --resume 完整恢复")
    if (args.resume or args.resume_from) and (args.check_only or args.max_updates):
        parser.error("恢复训练不能与检查/资源试跑混用")
    if args.resume_from == cfg["run_name"]:
        parser.error("--resume-from 需要使用新的 --run-name，保留原实验")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates 必须为正数")
    if args.device == "cpu" and not args.check_only:
        parser.error("CPU 仅用于 --check-only；正式训练请在有 CUDA 的环境运行")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请检查 GPU/驱动。仅检查可用 --check-only --device cpu")
    device = torch.device(args.device)
    torch.set_num_threads(cfg["cpu_threads"])
    seed = cfg["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        if cfg["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持配置的 bfloat16")
    data, prep = check_prepared(cfg)
    train, train_meta = load_split(data, "posttrain_train")
    info = None
    if cfg.get("tissue_auxiliary"):
        from tissue_loss_model.multitask import add_labels, tissue_info
        info = tissue_info(train_meta)
        train = add_labels(train, train_meta, info)
    collator = make_collator(cfg)
    result_dir = ROOT / cfg["result_root"] / (cfg["run_name"] + ("_trial" if args.max_updates else ""))
    out = ROOT / cfg["output_root"] / cfg["run_name"]
    state_path = out / "latest_training_state.pt"
    metadata = None
    if args.resume or args.resume_from:
        source_name = cfg["run_name"] if args.resume else args.resume_from
        source_out = ROOT / cfg["output_root"] / source_name
        metadata = json.loads((source_out / "run_metadata.json").read_text())
        if {k: v for k, v in cfg.items() if k != "run_name"} != {k: v for k, v in metadata["config"].items() if k != "run_name"}:
            raise ValueError("恢复训练配置发生变化；不能混用 loss、数据或训练计划")
        if metadata["cell_map_sha256"] != prep["cell_map_sha256"]:
            raise ValueError("恢复训练的来源划分发生变化")
        if args.resume and (result_dir / "training_summary.json").exists():
            print("训练已经完成，无需重新启动。", flush=True)
            return
        if args.resume and not state_path.exists():
            raise FileNotFoundError("完整断点缺失；旧实验请用 --resume-from 并指定新实验名")
    checkpoint = source_out if metadata is not None else ROOT / cfg["checkpoint_path"]
    model = load_training_model(checkpoint, cfg, device, info)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        check_only(model, train, train_meta, collator, cfg, device, result_dir)
        return
    if not args.resume and not args.max_updates and ((out.exists() and any(out.iterdir())) or (result_dir / "train_log.jsonl").exists()):
        raise FileExistsError("正式模型/日志已存在；请用 --run-name 指定新实验名")
    val, val_meta = load_split(data, "posttrain_val")
    if cfg.get("tissue_auxiliary"):
        val = add_labels(val, val_meta, info)
    if set(train_meta.cell_uid) & set(val_meta.cell_uid):
        raise ValueError("train/val 细胞重叠")
    if set(train_meta.tissue) != set(val_meta.tissue):
        raise ValueError("train/val 组织覆盖不一致")
    if metadata is None:
        # 仅原始 Base + 训练细胞估计尺度；恢复时复用已锁定尺度，不用已训练模型重估。
        cal_idx = calibration_indices(train_meta, cfg["calibration_cells_per_library"], cfg["calibration_seed"])
        if cfg.get("tissue_auxiliary"):
            print("Tissue labels/weights (train only):", json.dumps(info), flush=True)
            print(f"Calibrating fixed scales on {len(cal_idx)} train cells; train={len(train)}, val={len(val)}, test not loaded", flush=True)
        calibration = evaluate(model, train, train_meta.iloc[cal_idx], collator, cfg, device, seed=cfg["calibration_seed"])
        scales = initial_loss_scales(calibration, cfg)
        if cfg.get("tissue_auxiliary"):
            print("Fixed train-only loss scales:", json.dumps(scales), flush=True)
            print("Evaluating initial validation baseline...", flush=True)
        baseline = evaluate(model, val, val_meta, collator, cfg, device, scales)
    else:
        scales, baseline = metadata["loss_scales"], metadata["baseline_val"]
    if not all(math.isfinite(v) and v > 0 for v in scales.values()):
        raise ValueError("训练基线 loss 尺度必须有限且为正")
    params = freeze_for_posttrain(model, cfg["trainable_prefixes"])
    n_batches = math.ceil(len(train) / cfg["batch_size"])
    updates_per_epoch = math.ceil(n_batches / cfg["grad_accum_steps"])
    planned = cfg["max_epochs"] * updates_per_epoch
    total = min(args.max_updates, planned) if args.max_updates else planned
    warmup = max(1, round(total * cfg["warmup_fraction"]))
    groups = params
    if cfg.get("tissue_auxiliary"):
        head_ids = {id(p) for p in model.tissue_head.parameters()}
        groups = [{"params": [p for p in params if id(p) not in head_ids], "lr": cfg["learning_rate"]},
                  {"params": list(model.tissue_head.parameters()), "lr": cfg["tissue_head_learning_rate"]}]
    opt = torch.optim.AdamW(groups, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: lr_factor(u, total, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "float16")
    if metadata is None:
        metadata = {"config": cfg, "loss_scales": scales, "calibration": calibration,
                "calibration_cell_ids": train_meta.iloc[cal_idx].cell_uid.tolist(),
                "baseline_val": baseline, "prepare_report_sha256": file_hash(data / "prepare_report.json"),
                "base_checkpoint_sha256": file_hash(ROOT / cfg["checkpoint_path"] / "pytorch_model.bin"),
                "cell_map_sha256": prep["cell_map_sha256"], "planned_updates": total,
                "updates_per_epoch": updates_per_epoch, "warmup_updates": warmup,
                "selection_metric": "mean_tissue(0.5*ID/train_baseline_ID + 0.5*MSE/train_baseline_MSE)",
                "trainable_params": sum(p.numel() for p in params), "test_read": False,
                "evaluation_scope": "within-library held-out cells; not independent biological samples"}
        if cfg.get("tissue_auxiliary"):
            metadata["tissue_info"] = info
            metadata["selection_metric"] = "config.loss_weights-weighted fixed-scale tissue-macro ID/MSE and unmasked tissue CE; lower is better"
            metadata["tissue_training_view"] = "same 15% MLM-corrupted view; training-frequency class weights; labels never encoder inputs"
    best_score, best_update, bad, update = baseline["selection_score"], 0, 0, 0
    skipped_steps, last_eval_update, last_gradient_update = 0, 0, None
    start_epoch, start_micro, elapsed_before = 0, 0, 0.
    recent, gradient_log = [], []
    if args.resume_from:
        records = [json.loads(line) for line in (ROOT / cfg["result_root"] / args.resume_from / "train_log.jsonl").read_text().splitlines()]
        best_record = None
        for record in records:
            if "val" in record and record["val"]["selection_score"] < best_score - cfg["early_stop_min_delta"]:
                best_record = record
                best_score, best_update = record["val"]["selection_score"], record["update"]
        # 本次旧日志只在 epoch 1 内；不能把跨 epoch 的跳步总数猜成具体文库位置。
        if best_record is None or best_record.get("epoch") != 1:
            raise ValueError("旧权重恢复仅支持有明确首 epoch 验证记录的实验")
        update = best_update
        skipped_steps = best_record["amp_skipped_steps"]
        start_micro = (update + skipped_steps) * cfg["grad_accum_steps"]
        if start_micro >= n_batches:
            raise ValueError("旧断点位置超过首 epoch，需人工核实")
        matched = evaluate(model, val, val_meta, collator, cfg, device, scales)
        if abs(matched["selection_score"] - best_score) > 1e-6:
            raise ValueError("旧模型权重与推断的验证断点不匹配")
        last_eval_update = update
        rate = cfg["learning_rate"] * lr_factor(update, total, warmup)
        sched.load_state_dict(sched.state_dict() | {"last_epoch": update, "_step_count": update + 1, "_last_lr": [rate]})
        for group in opt.param_groups:
            group["lr"] = rate
        if scaler.is_enabled():
            scaler.load_state_dict(scaler.state_dict() | {"scale": best_record["amp_scale"]})
        metadata = metadata | {"config": cfg, "recovery": {
            "source_run": args.resume_from, "source_update": update,
            "optimizer_restored": False, "rng_restored": False,
            "note": "Legacy weights only: Adam moments and dropout RNG restarted; original loss scales, split and LR schedule retained. Updates after the saved best checkpoint are lost."}}
        print("Legacy recovery:", json.dumps(metadata["recovery"], ensure_ascii=False), flush=True)
    elif args.resume:
        # 只加载本项目自己写入的可信断点，其中含 Python/NumPy RNG 状态。
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        progress = restore_training_state(state, model, opt, sched, scaler)
        update, best_update, best_score, bad = (progress[k] for k in ("update", "best_update", "best_score", "bad"))
        start_epoch, start_micro = progress["epoch"], progress["next_micro"]
        skipped_steps, last_eval_update = progress["skipped_steps"], progress["last_eval_update"]
        last_gradient_update = progress["last_gradient_update"]
        recent, gradient_log = progress["recent"], progress["gradient_log"]
        elapsed_before = progress["elapsed_seconds"]
        del state
        print(f"Full recovery: update={update}, epoch={start_epoch+1}, next_micro={start_micro}", flush=True)
    if not args.resume:
        write_json(result_dir / "run_metadata.json", metadata)
        if not args.max_updates:
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "run_metadata.json", metadata)
            save_model(model, out)
    t0 = time.monotonic()

    def checkpoint(epoch, next_micro):
        if not args.max_updates:
            save_training_state(state_path, model, opt, sched, scaler, {
                "update": update, "epoch": epoch, "next_micro": next_micro,
                "best_update": best_update, "best_score": best_score, "bad": bad,
                "skipped_steps": skipped_steps, "last_eval_update": last_eval_update,
                "last_gradient_update": last_gradient_update, "recent": recent,
                "gradient_log": gradient_log, "elapsed_seconds": elapsed_before + time.monotonic() - t0})

    checkpoint(start_epoch, start_micro)
    print(json.dumps({"loss_mode": cfg["loss_mode"], "loss_coefficients": loss_coefficients(cfg, scales),
                      "loss_scales": scales, "baseline_val_score": best_score,
                      "planned_updates": total, "train_cells": len(train), "val_cells": len(val)}, indent=2), flush=True)
    with (result_dir / "train_log.jsonl").open("a" if args.resume else "w") as log:
        if args.resume:
            log.write(json.dumps({"event": "full_recovery", "update": update,
                                  "epoch": start_epoch + 1, "next_micro": start_micro}) + "\n")
            log.flush()
        if not args.resume:
            log.write(json.dumps({"update": 0, "val": baseline}) + "\n")
            if args.resume_from:
                log.write(json.dumps({"event": "legacy_recovery", "update": update, "val": matched, "optimizer_restored": False}) + "\n")
            log.flush()
        stop = False
        if args.resume and (bad >= cfg["early_stop_patience"] or last_eval_update == update == total):
            stop = True
        for epoch in range(start_epoch, cfg["max_epochs"]):
            if stop:
                break
            model.train()
            opt.zero_grad(set_to_none=True)
            micro_in_step = 0
            offset = start_micro if epoch == start_epoch else 0
            for micro, b in enumerate(batches(train, np.arange(len(train)), collator, cfg["batch_size"], seed + epoch * 100000, device, shuffle=True, start_batch=offset), offset):
                group_start = (micro // cfg["grad_accum_steps"]) * cfg["grad_accum_steps"]
                group_size = min(cfg["grad_accum_steps"], n_batches - group_start)
                with autocast(device, cfg):
                    output = model(**b)
                    loss = training_loss(output, cfg, scales)
                if not torch.isfinite(loss):
                    raise ValueError(f"update {update}: 非有限 loss")
                if micro_in_step == 0 and update != last_gradient_update and (update == 0 or update % cfg["eval_every_updates"] == 0):
                    grad = gradient_diagnostics(model, output, cfg, scales) | {"update": update}
                    gradient_log.append(grad)
                    last_gradient_update = update
                    print("shared gradients", grad, flush=True)
                scaler.scale(loss / group_size).backward()
                raw = {k: v.item() for k, v in task_losses(output, cfg).items()}
                if cfg.get("tissue_auxiliary"):
                    raw.update({"weighted_"+k: loss_coefficients(cfg, scales)[k]*v for k, v in list(raw.items())})
                recent.append(raw | {"loss": loss.item()})
                micro_in_step += 1
                if micro_in_step != group_size:
                    continue
                step = step_optimizer(opt, scaler, params, cfg["grad_clip"])
                micro_in_step = 0
                if not step["stepped"]:
                    skipped_steps += 1
                    del recent[-group_size:]
                    event = step | {"event": "amp_overflow_skip", "update": update,
                                    "epoch": epoch + 1, "micro_batch": micro + 1}
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                    log.flush()
                    print("AMP overflow: skipped optimizer step", event, flush=True)
                    checkpoint(epoch, micro + 1)
                    continue
                grad_norm = step["grad_norm"]
                sched.step()
                update += 1
                if update % cfg["log_every_updates"] == 0:
                    print(f"update {update}/{total} epoch {epoch+1} loss {np.mean([x['loss'] for x in recent]):.4f} "
                          f"grad {grad_norm:.3f} lr {sched.get_last_lr()[0]:.3g} "
                          f"amp_scale {scaler.get_scale():.0f}", flush=True)
                    if cfg.get("tissue_auxiliary"):
                        components = {k: float(np.mean([x[k] for x in recent])) for k in recent[0]}
                        log.write(json.dumps({"event": "train_components", "update": update,
                                              "losses": components, "learning_rates": sched.get_last_lr()}) + "\n")
                        log.flush()
                        print("loss components", json.dumps(components), flush=True)
                if update % cfg["eval_every_updates"] == 0 or update >= total:
                    val_metrics = evaluate(model, val, val_meta, collator, cfg, device, scales)
                    last_eval_update = update
                    score = val_metrics["selection_score"]
                    record = {"update": update, "epoch": epoch + 1, "val": val_metrics,
                              "train_recent": {k: float(np.mean([x[k] for x in recent])) for k in recent[0]},
                              "grad_norm": grad_norm, "lr": sched.get_last_lr()[0],
                              "amp_scale": scaler.get_scale(), "amp_skipped_steps": skipped_steps}
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    recent.clear()
                    print(f"validation update {update}: tissue-macro score={score:.6f}", flush=True)
                    if cfg.get("tissue_auxiliary"):
                        tissue_val = val_metrics["tissue_classification"]["unmasked"]
                        print(f"tissue validation: accuracy={tissue_val['accuracy']:.4f} macro_F1={tissue_val['macro_F1']:.4f} macro_CE={tissue_val['macro_ce']:.4f}", flush=True)
                    if score < best_score - cfg["early_stop_min_delta"]:
                        best_score, best_update, bad = score, update, 0
                        if not args.max_updates:
                            save_model(model, out)
                    else:
                        bad += 1
                    model.train()
                    checkpoint(epoch, micro + 1)
                    if bad >= cfg["early_stop_patience"] or update >= total:
                        stop = True
                        break
                elif update % cfg["log_every_updates"] == 0:
                    checkpoint(epoch, micro + 1)
            if stop:
                break
            checkpoint(epoch + 1, 0)
        if update == 0:
            raise RuntimeError("本次没有成功执行任何参数更新，请查看 AMP 溢出日志并检查 precision")
        # AMP 跳过会使成功更新数少于计划值，最后不足一个评估间隔也必须参与选模。
        if last_eval_update != update:
            val_metrics = evaluate(model, val, val_meta, collator, cfg, device, scales)
            log.write(json.dumps({"event": "final_validation", "update": update, "val": val_metrics,
                                  "amp_scale": scaler.get_scale(), "amp_skipped_steps": skipped_steps}, allow_nan=False) + "\n")
            if val_metrics["selection_score"] < best_score - cfg["early_stop_min_delta"]:
                best_score, best_update = val_metrics["selection_score"], update
                if not args.max_updates:
                    save_model(model, out)
    summary = {"updates": update, "best_update": best_update, "best_val_score": best_score,
               "selected_unmodified_base": best_update == 0, "trial_only": bool(args.max_updates),
               "elapsed_seconds": elapsed_before + time.monotonic() - t0, "test_read": False,
               "shared_gradient_diagnostics": gradient_log, "amp_skipped_steps": skipped_steps,
               "final_amp_scale": scaler.get_scale()}
    if "recovery" in metadata:
        summary["recovery"] = metadata["recovery"]
    if not args.max_updates:
        write_json(out / "training_summary.json", summary)
    write_json(result_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
