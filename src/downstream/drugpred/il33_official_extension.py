"""Add the official 0.8-expression/0.2-ID encoder to the locked mixed-cell task.

Reuse the original trainer, targets, splits, panel and hyperparameters. Only the
new frozen features and newly fitted official heads are stored in this version.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch

import train_il33_mixed_delta as task

ROOT = task.ROOT
CONFIG = "configs/drugpred_il33_official_extension.json"
ENCODER = "ILC2_official"


def configuration(path=CONFIG):
    extension = json.loads((ROOT / path).read_text())
    base = json.loads((ROOT / extension["base_config"]).read_text())
    cfg = dict(base, **{k: extension[k] for k in ("version", "data_directory", "result_directory", "model_directory")})
    cfg["models"] = {ENCODER: extension["checkpoint"]}
    return base, cfg


def align_features(cache_metadata, split_metadata, features):
    """Feature rows follow PBS IDs, never the old train/development roles."""
    pbs = split_metadata.index[split_metadata.condition.eq("PBS")]
    loc = cache_metadata.index.get_indexer(pbs)
    if not cache_metadata.index.is_unique or (loc < 0).any():
        raise ValueError("Official feature cache lacks unique source PBS cells")
    cached = cache_metadata.loc[pbs, ["study", "response_group", "library_id", "condition"]].astype(str)
    wanted = split_metadata.loc[pbs, cached.columns].astype(str)
    if not cached.equals(wanted):
        raise ValueError("Official cached cell provenance differs")
    h = np.asarray(features[loc])
    if h.shape != (len(pbs), 768) or h.dtype != np.float32 or not np.isfinite(h).all():
        raise ValueError("Invalid official FP32 encoder features")
    return h


def prepare(base, cfg):
    source, old_out, _, original, original_hash = task.prepared(base)
    data, out, _ = task.paths(cfg)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if (data / "prepare_manifest.json").exists():
        task.prepared(cfg)
        print("Official supplement already prepared", flush=True)
        return
    cache = ROOT / cfg["encoder_cache"]
    signature = json.loads((cache / "signature.json").read_text())
    checkpoint_hash = task.file_hash(ROOT / cfg["models"][ENCODER] / "pytorch_model.bin")
    if checkpoint_hash != signature["checkpoints"][ENCODER]:
        raise ValueError("Official checkpoint differs from the cached encoder")
    if (signature["encoding"] != "median" or signature["max_gene_tokens"] != 2048
            or signature["precision"] != "float32" or signature["pooling"] != "masked_gene_mean"):
        raise ValueError("Official input/pooling differs from the fixed comparison")
    # The original run already re-encoded every PBS cell against this shared cache.
    for name, sha in original["feature_cache_sha256"].items():
        if task.file_hash(cache / name / "PBS_h.npy") != sha:
            raise ValueError("The verified common encoder cache changed")
    original_protocol = task.checkpoint_lock(base, original_hash)
    if json.loads((old_out / "test_protocol.json").read_text()) != original_protocol:
        raise ValueError("Original six locked heads changed")
    for name in ("gene_panel.tsv", "functional_gene_sets.tsv", "cell_manifest.tsv.gz", "full_counts.h5ad"):
        target = data / name
        if not target.exists():
            target.symlink_to(source / name)
    cache_metadata = pd.read_csv(cache / "PBS_metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
    feature_path = cache / ENCODER / "PBS_h.npy"
    features = np.load(feature_path, mmap_mode="r")
    feature_hashes = {}
    for role in task.ROLES:
        folder = data / role
        folder.mkdir(exist_ok=True)
        names = ["metadata.tsv.gz", "y.npy"]
        if role == "train":
            names.append("statistics.npz")
        for name in names:
            target = folder / name
            if not target.exists():
                target.symlink_to(source / role / name)
        m = pd.read_csv(folder / "metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
        h = align_features(cache_metadata, m, features)
        np.save(folder / f"{ENCODER}_h.npy", h)
        feature_hashes[role] = task.file_hash(folder / f"{ENCODER}_h.npy")
        print(f"Official {role}: PBS={len(h)}, panel={original['output_genes']}", flush=True)
    manifest = dict(original, config=cfg, checkpoint_sha256={ENCODER: checkpoint_hash},
        feature_cache_sha256={ENCODER: task.file_hash(feature_path)},
        aligned_feature_sha256=feature_hashes,
        extension_of=dict(manifest_sha256=original_hash, data_directory=base["data_directory"],
            original_test_protocol_sha256=task.file_hash(old_out / "test_protocol.json"),
            original_test_metrics_sha256=task.file_hash(old_out / "test_background_metrics.csv")),
        test_scope="Supplement on an already opened test; no retuning of the original recipe")
    task.atomic_json(data / "config.json", cfg)
    task.atomic_json(data / "prepare_manifest.json", manifest)
    task.atomic_json(out / "prepare_report.json", manifest)


def report(base, cfg):
    _, old_out, _, _, original_hash = task.prepared(base)
    _, out, _, manifest, _ = task.prepared(cfg)
    if original_hash != manifest["extension_of"]["manifest_sha256"]:
        raise ValueError("Original task changed")
    original_path = old_out / "test_background_metrics.csv"
    if task.file_hash(original_path) != manifest["extension_of"]["original_test_metrics_sha256"]:
        raise ValueError("Original scores changed")
    original = pd.read_csv(original_path)
    extra = pd.read_csv(out / "test_background_metrics.csv")
    # Recomputed baselines must agree before combining scores from separate runs.
    baseline = extra[extra.seed.eq(-1)].sort_values(["method", "group"])
    reference = original[original.seed.eq(-1)].sort_values(["method", "group"])
    if not baseline[["method", "group"]].reset_index(drop=True).equals(reference[["method", "group"]].reset_index(drop=True)):
        raise ValueError("Different backgrounds/baselines")
    np.testing.assert_allclose(baseline.delta_MSE, reference.delta_MSE, rtol=1e-10, atol=1e-14)
    combined = pd.concat([original, extra[extra.method.eq(ENCODER)]], ignore_index=True)
    panel = pd.read_csv(task.paths(cfg)[0] / "gene_panel.tsv", sep="\t")
    truth = {}
    for row in combined.itertuples():
        folder = (out if row.method == ENCODER else old_out) / "test" / row.method
        if row.seed != -1:
            folder = folder / f"seed{row.seed}"
        moments = pd.read_csv(folder / f"{row.group}_gene_predictions.tsv.gz", sep="\t",
                              usecols=["gene_id", "predicted_delta", "observed_delta"])
        if not moments.gene_id.equals(panel.gene_id):
            raise ValueError("Different output gene order")
        observed = moments.observed_delta.to_numpy()
        if row.group in truth:
            np.testing.assert_allclose(observed, truth[row.group], rtol=0, atol=1e-14)
        else:
            truth[row.group] = observed
        mse = float(np.mean((moments.predicted_delta-observed)**2))
        np.testing.assert_allclose(mse, row.delta_MSE, rtol=1e-10, atol=1e-14)
    task.atomic_json(out / "comparison_verification.json", dict(method_background_rows=len(combined),
        output_genes=len(panel), same_gene_order=True, same_observed_delta=True, recomputed_delta_MSE=True,
        same_baselines=True, original_models_retrained=False))
    combined.to_csv(out / "comparison_background_metrics.csv", index=False)
    rows = []
    for (method, seed), frame in combined.groupby(["method", "seed"], sort=False):
        for scope, part in [("all", frame), *list(frame.groupby("study", sort=False))]:
            row = dict(method=method, seed=int(seed), scope=scope,
                       **task.aggregate_metrics(part.to_dict("records")))
            row.update(functional_RMSE=float(np.sqrt((part.functional__delta_RMSE**2).mean())),
                train_DE100_RMSE=float(np.sqrt((part.train_DE100__delta_RMSE**2).mean())),
                DEG_overlap20=float(part.DEG_overlap_20.mean()), DEG_overlap100=float(part.DEG_overlap_100.mean()),
                variance_R2=float(part.R2_var_all.mean()))
            rows.append(row)
    macro = pd.DataFrame(rows)
    macro.to_csv(out / "comparison_macro_metrics.csv", index=False)
    columns = [c for c in macro if c not in ("method", "seed", "scope")]
    macro.groupby(["method", "scope"])[columns].agg(["mean", "std"]).to_csv(out / "comparison_seed_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["GeneCompass", ENCODER, "ILC2_tissue"]
    scores = macro[macro.scope.eq("all") & macro.method.isin(names)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["sqrt_macro_delta_MSE", "functional_RMSE"], ["All panel genes", "67 functional candidates"]):
        table = scores.groupby("method")[col].agg(["mean", "std"]).reindex(names)
        ax.bar(names, table["mean"], yerr=table["std"], capsize=4)
        ax.set(title=label, ylabel="sqrt(macro delta MSE)")
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out / "three_encoder_comparison.png", dpi=180)
    plt.close(fig)
    runs = []
    for run_cfg in (base, cfg):
        for encoder in run_cfg["models"]:
            for seed in run_cfg["seeds"]:
                folder, _ = task.run_paths(run_cfg, encoder, seed)
                c = json.loads((folder / "complete.json").read_text())
                runs.append(dict(model=encoder, seed=seed, epochs=c["epochs"], best_epoch=c["best_epoch"],
                                 best_validation_MSE=c["best_validation"]["macro_delta_MSE"], best_sha256=c["best_sha256"]))
    pd.DataFrame(runs).to_csv(out / "comparison_training_runs.csv", index=False)
    summary = macro[macro.scope.eq("all")].groupby("method")[columns].agg(["mean", "std"])
    def formatted(method, metric):
        mean, std = summary.loc[method, (metric, "mean")], summary.loc[method, (metric, "std")]
        if not np.isfinite(mean):
            return "—"
        return f"{mean:.6f} ± {std:.6f}" if np.isfinite(std) else f"{mean:.6f}"
    lines = ["# IL-33：三种冻结encoder的固定方案比较", "",
        "复用同一32,202细胞、11,611输出基因、四背景等权direct-Δ任务。三模型各seed0/1/2，"
        "相同预测头、训练参数、validation选模和早停；原两模型权重不改，只补充official三组训练。", "",
        "Official表示使用0.8×expression MSE + 0.2×gene-ID CE后训练的encoder；这不是本次下游loss的权重。"
        "本次仅训练下游预测头，不重新后训练encoder。", "",
        "## 最终test结果", "",
        "Δ为log1p(CP10K)均值差；先对四背景Δ-MSE等权平均再开平方。±为三个优化seed的样本SD，不是动物重复CI。", "",
        "| 方法 | sqrt(macro Δ-MSE) ↓ | Δ-Pearson ↑ | 67候选Δ-RMSE ↓ | DEG overlap@100 ↑ |",
        "|---|---:|---:|---:|---:|"]
    for method in ["control_copy", "global_mean_delta", "background_mean_delta", *names]:
        lines.append("| " + method + " | " + " | ".join(formatted(method, c) for c in
            ["sqrt_macro_delta_MSE", "macro_Pearson_delta", "functional_RMSE", "DEG_overlap100"]) + " |")
    official = summary.loc[ENCODER, ("sqrt_macro_delta_MSE", "mean")]
    changes = {m: 100*(official/summary.loc[m, ("sqrt_macro_delta_MSE", "mean")]-1) for m in ["GeneCompass", "ILC2_tissue"]}
    lines += ["", f"Official主误差相对GeneCompass变化 {changes['GeneCompass']:+.3f}%，"
        f"相对Tissue变化 {changes['ILC2_tissue']:+.3f}%（负值表示误差降低）。",
        "不能仅据微小均值差或三个优化seed宣称生物学显著优势。", "",
        "## 分背景Δ-RMSE", "", "| 背景 | GeneCompass | Official | Tissue |", "|---|---:|---:|---:|"]
    by_group = combined[combined.method.isin(names)].groupby(["group", "method"]).delta_RMSE.mean()
    for group in cfg["backgrounds"]:
        lines.append("| " + group + " | " + " | ".join(f"{by_group.loc[group, m]:.6f}" for m in names) + " |")
    lines += ["", "## 分背景方差R²", "", "| 背景 | GeneCompass | Official | Tissue |", "|---|---:|---:|---:|"]
    variances = combined[combined.method.isin(names)].groupby(["group", "method"]).R2_var_all.mean()
    for group in cfg["backgrounds"]:
        lines.append("| " + group + " | " + " | ".join(f"{variances.loc[group, m]:.6f}" for m in names) + " |")
    lines += ["", "方差R²为负表示逐基因方差向量预测较差，不是负方差；均值监督没有直接拟合完整分布。"]
    lines += ["", "## 训练完成情况", "", "| 模型 | seed | 完成epoch | 最佳epoch |", "|---|---:|---:|---:|"]
    for run in runs:
        lines.append(f"| {run['model']} | {run['seed']} | {run['epochs']} | {run['best_epoch']} |")
    device_path = out / "evaluation_device.json"
    evaluated_on = json.loads(device_path.read_text())["device"] if device_path.exists() else "见运行日志"
    lines += ["", "## 评价边界与复核", "",
        "- 原test已在补充official前打开；本次继承全部原参数，未用test挑epoch或seed，属于同一测试集的补充比较，不是新的盲测。",
        "- 同一文库同时出现在train/validation/test；结论只限已见来源内部细胞评价，不是跨研究或独立动物泛化。",
        "- 当前只监督群体平均Δ；即使均值误差较低，也不能证明真实单细胞命运、完整分布或因果调控关系。",
        "- 比较基线都采用相同非负截断。固定mean-Δ会削弱下调幅度，不能把相对该基线的全部收益归因为encoder质量。",
        "- 新旧实验重新计算的三种基线MSE已核对一致，panel/目标/metadata通过链接复用，模型checkpoint和特征缓存签名保存在prepare_report.json。",
        "- comparison_verification.json记录各方法×背景的固定gene顺序、共同观测Δ和逐基因moments独立复算MSE的核验。",
        f"- 本次official最终评分设备：{evaluated_on}；原GeneCompass/Tissue评分为CPU。",
        "- 原始分背景结果：comparison_background_metrics.csv；分研究/macro：comparison_macro_metrics.csv；三seed汇总：comparison_seed_summary.csv。",
        "- 九组训练与权重签名：comparison_training_runs.csv；图：three_encoder_comparison.png。", ""]
    (out / "comparison_report.md").write_text("\n".join(lines))
    print(scores.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "train", "test", "report", "all"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base, cfg = configuration(args.config)
    torch.set_num_threads(cfg["cpu_threads"])
    device = torch.device(args.device)
    if args.command not in ("prepare", "report") and device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; explicitly select --device cpu")
    if args.command in ("prepare", "all"):
        prepare(base, cfg)
    if args.command == "check":
        task.check(cfg, device)
    if args.command in ("train", "all"):
        for seed in cfg["seeds"]:
            task.train_one(cfg, ENCODER, seed, device, args.resume)
    if args.command in ("test", "all"):
        # Match the original models' final CPU inference, including in all mode.
        test_device = torch.device("cpu") if args.command == "all" else device
        task.test(cfg, test_device)
        task.atomic_json(task.paths(cfg)[1] / "evaluation_device.json", dict(device=str(test_device)))
    if args.command in ("test", "report", "all"):
        report(base, cfg)


if __name__ == "__main__":
    main()
