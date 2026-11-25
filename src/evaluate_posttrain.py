"""训练结束并锁定方案后，单独评价 5% test；同细胞同掩码比较原始/适配模型。"""
import argparse
import gc
import json

import torch

from posttrain_data import file_hash
from posttrain_training import ROOT, check_prepared, evaluate, load_split, make_collator
from genecompass_compat import load_genecompass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="已选定 posttrain 模型目录")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    model_dir = ROOT / args.model_dir
    if not (model_dir / "training_summary.json").exists():
        raise ValueError("模型尚无训练完成记录，不能做最终测试")
    run = json.loads((model_dir / "run_metadata.json").read_text())
    cfg, scales = run["config"], run["loss_scales"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.set_num_threads(cfg["cpu_threads"])
    data, prep = check_prepared(cfg)
    if file_hash(data / "prepare_report.json") != run["prepare_report_sha256"]:
        raise ValueError("训练后数据准备报告发生变化")
    if file_hash(data / "posttrain_test_cell_map.tsv.gz") != run["cell_map_sha256"]["posttrain_test"]:
        raise ValueError("锁定的测试细胞清单发生变化")
    if file_hash(ROOT / cfg["checkpoint_path"] / "pytorch_model.bin") != run["base_checkpoint_sha256"]:
        raise ValueError("原始 Base 权重发生变化")
    result_dir = ROOT / cfg["result_root"] / cfg["run_name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "test_comparison.json"
    if output.exists():
        raise FileExistsError("已有最终测试结果；请先查看，避免反复据 test 调参")
    ds, meta = load_split(data, "posttrain_test")
    collator = make_collator(cfg)
    comparison = {"evaluation_scope": "within-library held-out cells; not biological replicate holdout",
                  "cells": len(ds), "loss_scales_from_train": scales,
                  "eval_mask_seed": cfg["eval_mask_seed"], "eval_batch_size": cfg["eval_batch_size"]}
    for label, directory in (("genecompass_base", ROOT / cfg["checkpoint_path"]), ("posttrain", model_dir)):
        model, load = load_genecompass(directory, device=device)
        if load["missing"] or load["unexpected"]:
            raise ValueError(f"{label}: checkpoint 加载不完整")
        comparison[label] = evaluate(model, ds, meta, collator, cfg, device, scales)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: comparison[k]["overall"] for k in ("genecompass_base", "posttrain")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
