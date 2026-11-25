"""GSE189285 lung<->liver direct delta: fixed epochs, no validation or best epoch.

Both directions are declared together. A tissue's IL33 cells are training data
in one direction and test outcomes in the other; fitting remains source-only.
All final heads are locked before any transfer score is produced.
"""
import argparse
import fcntl
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

import train_il33_mixed_delta as task
import train_il33_external_delta as external

ROOT = task.ROOT
CONFIG = "configs/drugpred_il33_bidirectional_tissue.json"
DIRECTIONS = {"lung_to_liver": ("Lung", "Liver"), "liver_to_lung": ("Liver", "Lung")}
INHERITED = ("models", "seeds", "hidden_size", "batch_size", "prediction_batch_size", "learning_rate",
             "weight_decay", "gradient_clip", "minimum_train_detection", "tokenization_config",
             "functional_gene_sets", "cpu_threads")


def configuration(path=CONFIG):
    settings = json.loads((ROOT / path).read_text())
    _, base = external.configuration(settings.pop("base_config"))
    return dict({k: base[k] for k in INHERITED}, **settings)


def direction_config(cfg, direction):
    source, target = DIRECTIONS[direction]
    return dict(cfg, direction=direction, source_tissue=source, target_tissue=target,
        backgrounds=[f"GSE189285_{source}"], target_backgrounds=[f"GSE189285_{target}"],
        **{key: str(Path(cfg[key]) / direction)
           for key in ("data_directory", "result_directory", "model_directory")})


def load_tissue(cfg, tissue):
    parts, metas = [], []
    for condition in ("PBS", "IL33"):
        path = ROOT / cfg["tissues"][tissue][condition]
        obs, _ = external.annotations(path)
        if not (obs.study.eq("GSE189285") & obs.tissue.eq(tissue) & obs.condition.eq(condition)
                & obs.qc_version.eq("ilc2_balanced_v1") & obs.selected_ILC2.eq(True)).all():
            raise ValueError(f"Unexpected source/QC: {path}")
        a = external.raw_object(path, obs)
        m = obs.copy()
        if "response_group" in m:
            m["original_response_group"] = m.response_group
        m["response_group"] = f"GSE189285_{tissue}"
        m["source_h5ad"], m["source_cell_id"], m["source_row"] = str(path), m.index, np.arange(len(m))
        metas.append(m)
        parts.append(a)
    if not parts[0].var_names.equals(parts[1].var_names):
        raise ValueError("Paired condition gene spaces differ; raw counts must not be silently padded")
    expected_umi = sum(int(a.X.sum(dtype=np.int64)) for a in parts)
    var = parts[0].var[["gene_symbol"]].copy()
    a = ad.concat(parts, join="inner", merge="same")
    a.var = var
    if int(a.X.sum(dtype=np.int64)) != expected_umi or not a.obs_names.is_unique:
        raise ValueError("Counts or cells lost during tissue merge")
    return a, pd.concat(metas)


def cached_features(cfg, tissue, a, meta, tokens, medians):
    cache = ROOT / cfg[f"{tissue.lower()}_feature_cache"]
    liver = tissue == "Liver"
    cm = pd.read_csv(cache / ("liver_cells.tsv.gz" if liver else "PBS_metadata.tsv.gz"), sep="\t", index_col=0, low_memory=False)
    pbs = meta.index[meta.condition.eq("PBS")]
    if not cm.index.is_unique:
        raise ValueError("Duplicated feature cache IDs")
    loc = cm.index.get_indexer(pbs)
    if (loc < 0).any() or not cm.loc[pbs, ["study", "gsm", "condition", "tissue"]].astype(str).equals(
            meta.loc[pbs, ["study", "gsm", "condition", "tissue"]].astype(str)):
        raise ValueError("Cached cell identities disagree")
    col, median, _ = task.map_genes(a.var_names, tokens, medians)
    encoded = task.encode(a.X[meta.condition.eq("PBS").to_numpy()], col, median, 2048, "median")
    with np.load(cache / ("liver_inputs.npz" if liver else "PBS_inputs.npz")) as inputs:
        for name, actual in zip(("ids", "vals", "length"), encoded):
            if not np.array_equal(actual, inputs[name][loc]):
                raise ValueError("Raw PBS no longer reproduces cached Top2048 inputs")
    features, hashes = {}, {}
    for encoder in cfg["models"]:
        path = cache / (f"{encoder}_liver.npy" if liver else f"{encoder}/PBS_h.npy")
        h = np.load(path, mmap_mode="r")
        if h.shape != (len(cm), 768):
            raise ValueError("Wrong feature cache dimensions")
        values = np.asarray(h[loc])
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid frozen features")
        features[encoder], hashes[encoder] = values, task.file_hash(path)
    return features, hashes


def prepare(cfg):
    data, out, _ = task.paths(cfg)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if (data / "config.json").exists() and json.loads((data / "config.json").read_text()) != cfg:
        raise ValueError("Settings changed; use a new version")
    task.atomic_json(data / "config.json", cfg)
    if (data / "prepare_complete.json").exists():
        ready(cfg)
        print("Both directions already prepared", flush=True)
        return
    tok_cfg = json.loads((ROOT / cfg["tokenization_config"]).read_text())
    tokens = pickle.loads((ROOT / tok_cfg["token_dictionary"]).read_bytes())
    medians = pickle.loads((ROOT / tok_cfg["gene_median_reference"]).read_bytes())
    signature = json.loads((ROOT / cfg["lung_feature_cache"] / "signature.json").read_text())
    liver_cfg = json.loads((ROOT / cfg["liver_cache_provenance"]).read_text())
    checkpoints = {name: task.file_hash(ROOT / path / "pytorch_model.bin") for name, path in cfg["models"].items()}
    if (signature["tokens"] != task.file_hash(ROOT / tok_cfg["token_dictionary"])
            or signature["medians"] != task.file_hash(ROOT / tok_cfg["gene_median_reference"])
            or signature["encoding"] != "median" or signature["max_gene_tokens"] != 2048
            or signature["precision"] != "float32" or signature["pooling"] != "masked_gene_mean"
            or checkpoints != signature["checkpoints"] or checkpoints != liver_cfg["checkpoint_hashes"]):
        raise ValueError("Checkpoint/tokenization/pooling cache provenance differs")
    source_hashes = {path: task.file_hash(ROOT / path) for parts in cfg["tissues"].values() for path in parts.values()}
    if any(source_hashes[path] != sha for path, sha in liver_cfg["liver_sources"].items()):
        raise ValueError("Liver source counts changed since feature extraction")
    objects, metadata, features, feature_hashes = {}, {}, {}, {}
    for tissue in cfg["tissues"]:
        a, m = load_tissue(cfg, tissue)
        objects[tissue], metadata[tissue] = a, m
        features[tissue], feature_hashes[tissue] = cached_features(cfg, tissue, a, m, tokens, medians)
        print(f"Verified {tissue}: {len(m)} cells; PBS={int(m.condition.eq('PBS').sum())}", flush=True)
    if set(metadata["Lung"].index) & set(metadata["Liver"].index):
        raise ValueError("Tissue cell IDs overlap")
    common = objects["Lung"].var_names.intersection(objects["Liver"].var_names, sort=False)
    functional = pd.read_csv(ROOT / cfg["functional_gene_sets"])
    shared = dict(config=cfg, source_sha256=source_hashes, checkpoint_sha256=checkpoints,
        feature_cache_sha256=feature_hashes, common_measured_genes=len(common),
        role_scope="direction-specific; each tissue is source in one direction and target in the other",
        validation=False, epoch_selection="fixed final epoch only", pooling="FP32 masked gene mean, no CLS",
        raw_UMI={t: int(a.X.sum(dtype=np.int64)) for t, a in objects.items()})
    task.atomic_json(data / "shared_manifest.json", shared)
    shared_hash = task.file_hash(data / "shared_manifest.json")
    counts = []
    for direction in DIRECTIONS:
        dc = direction_config(cfg, direction)
        dd, oo, _ = task.paths(dc)
        dd.mkdir(parents=True, exist_ok=True)
        oo.mkdir(parents=True, exist_ok=True)
        source, target = dc["source_tissue"], dc["target_tissue"]
        source_meta = external.new_roles(metadata[source], "train")
        target_meta = external.new_roles(metadata[target], "test")
        a = objects[source]
        var = a.var.copy()
        var["present_source"], var["present_target"] = True, var.index.isin(common)
        panel = task.select_panel(a.X, var, source_meta, dc["backgrounds"], ["source", "target"],
                                  functional.gene_id, cfg["minimum_train_detection"])
        panel["in_encoder_vocab"] = panel.gene_id.isin(tokens)
        panel.to_csv(dd / "gene_panel.tsv", sep="\t", index=False)
        functional.to_csv(dd / "functional_gene_sets.tsv", sep="\t", index=False)
        pd.concat([source_meta, target_meta]).to_csv(dd / "cell_manifest.tsv.gz", sep="\t", index_label="cell_id")
        hashes = {}
        for role, tissue, m in (("train", source, source_meta), ("test", target, target_meta)):
            folder = dd / role
            folder.mkdir(exist_ok=True)
            m.to_csv(folder / "metadata.tsv.gz", sep="\t", index_label="cell_id")
            hashes[role] = task.file_hash(folder / "metadata.tsv.gz")
            for encoder in cfg["models"]:
                np.save(folder / f"{encoder}_h.npy", features[tissue][encoder])
            if role == "train":
                y = task.normalized(objects[tissue], panel.gene_id)
                if not np.isfinite(y).all():
                    raise ValueError("Missing source output genes")
                np.save(folder / "y.npy", y)
                stats = task.training_statistics(y, m, panel, dc["backgrounds"], oo / "training_DEG")
                np.savez_compressed(folder / "statistics.npz", **stats)
                del y
                raw = objects[tissue].copy()
                raw.obs = m[external.OBS_COLUMNS].copy()
                raw.uns = dict(metadata=dict(full_metadata="train/metadata.tsv.gz", direction=direction, role="train"))
                raw.write_h5ad(dd / "source_full_counts.h5ad", compression="gzip")
                del raw
            else:
                control = objects[tissue][m.condition.eq("PBS").to_numpy()]
                np.save(folder / "control_y.npy", task.normalized(control, panel.gene_id))
            for condition, n in m.condition.value_counts().items():
                counts.append(dict(direction=direction, role=role, tissue=tissue, condition=condition, cells=int(n)))
        task.atomic_json(dd / "config.json", dc)
        manifest = dict(config=dc, shared_manifest_sha256=shared_hash,
            panel_sha256=task.file_hash(dd / "gene_panel.tsv"), cell_map_sha256=hashes,
            training_statistics_sha256=task.file_hash(dd / "train/statistics.npz"),
            output_genes=len(panel), predefined_functional_genes=int(panel.functional.sum()),
            train_unsupported_panel_genes=int((~panel.train_supported).sum()), validation=False,
            epochs=cfg["epochs"], source_raw_UMI=shared["raw_UMI"][source],
            fitting_scope="source tissue only; no target expression used to select panel or fit statistics")
        task.atomic_json(dd / "prepare_manifest.json", manifest)
        task.atomic_json(oo / "prepare_report.json", manifest)
        print(f"Prepared {direction}: output genes={len(panel)}, functional={int(panel.functional.sum())}", flush=True)
    pd.DataFrame(counts).to_csv(out / "split_counts.csv", index=False)
    task.atomic_json(data / "prepare_complete.json", dict(config=cfg, shared_manifest_sha256=shared_hash))


def ready(cfg):
    data, _, _ = task.paths(cfg)
    complete = json.loads((data / "prepare_complete.json").read_text())
    if complete["config"] != cfg or complete["shared_manifest_sha256"] != task.file_hash(data / "shared_manifest.json"):
        raise ValueError("Prepared version changed")
    for direction in DIRECTIONS:
        _, _, _, manifest, _ = task.prepared(direction_config(cfg, direction))
        if manifest["shared_manifest_sha256"] != complete["shared_manifest_sha256"]:
            raise ValueError("Directional preparation differs")


def train_fixed(cfg, encoder, seed, device, resume, *, expression_input=False,
                head_factory=task.DirectDeltaHead, expression_factory=task.ExpressionDeltaMLP):
    data, out, _, manifest, fingerprint = task.prepared(cfg)
    model_dir, result_dir = task.run_paths(cfg, encoder, seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (model_dir / "complete.json").exists():
            c = json.loads((model_dir / "complete.json").read_text())
            if c["fingerprint"] != fingerprint or c["final_sha256"] != task.file_hash(model_dir / "final.pt") or c["epoch"] != cfg["epochs"]:
                raise ValueError("Completed final checkpoint changed")
            print(f"Already complete: {cfg['direction']}/{encoder}/seed{seed}", flush=True)
            return
        latest = model_dir / "latest.pt"
        if latest.exists() and not resume:
            raise ValueError("Incomplete run exists; use --resume")
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        dataset = (task.expression_dataset(data, manifest, "train") if expression_input
                   else task.load_split(data, manifest, "train", encoder))
        m, y, h = dataset
        pm = m[m.condition.eq("PBS")]
        x = h if expression_input else y[m.condition.eq("PBS").to_numpy()]
        panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
        with np.load(data / "train/statistics.npz") as values:
            stats = dict(values)
        model = (expression_factory(len(panel), cfg["hidden_size"]) if expression_input else
                 head_factory(torch.from_numpy(np.array(h)), len(panel), cfg["hidden_size"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["minimum_learning_rate"])
        start, history = 1, []
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state["fingerprint"] != fingerprint:
                raise ValueError("Resume configuration/data differs")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            start, history = state["next_epoch"], state["history"]
        targets = {g: torch.tensor(stats[g + "__delta"], dtype=torch.float32, device=device) for g in cfg["backgrounds"]}
        for epoch in range(start, cfg["epochs"]+1):
            rate = optimizer.param_groups[0]["lr"]
            loss, grad = task.fit_epoch(model, optimizer, pm, h, x, targets, cfg, rng, device)
            rows, macro = task.score_model(model, dataset, cfg, panel, stats, device)
            scheduler.step()
            history.append(dict(epoch=epoch, weighted_batch_loss=loss, gradient_norm=grad, learning_rate=rate,
                **{"train_"+k: v for k, v in macro.items()}, functional_delta_RMSE=rows[0]["functional__delta_RMSE"]))
            pd.DataFrame(history).to_csv(result_dir / "history.csv", index=False)
            pd.DataFrame(rows).to_csv(result_dir / "latest_train_metrics.csv", index=False)
            task.atomic_torch(latest, dict(fingerprint=fingerprint, encoder=encoder, seed=seed, epoch=epoch,
                model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                next_epoch=epoch+1, history=history, numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
            print(f"{cfg['direction']}/{encoder}/seed{seed} epoch={epoch}/{cfg['epochs']} "
                  f"train={macro['macro_delta_MSE']:.6g} lr={rate:.3g}", flush=True)
        task.atomic_torch(model_dir / "final.pt", dict(fingerprint=fingerprint, encoder=encoder, seed=seed,
            epoch=cfg["epochs"], model=model.state_dict(), selection="fixed_final_epoch", history=history))
        task.atomic_json(model_dir / "complete.json", dict(fingerprint=fingerprint, encoder=encoder, seed=seed,
            epoch=cfg["epochs"], selection="fixed_final_epoch", validation_used=False,
            final_sha256=task.file_hash(model_dir / "final.pt")))


def final_protocol(cfg):
    ready(cfg)
    records = {}
    for direction in DIRECTIONS:
        dc = direction_config(cfg, direction)
        _, _, _, _, fingerprint = task.prepared(dc)
        for encoder in cfg["models"]:
            for seed in cfg["seeds"]:
                folder, _ = task.run_paths(dc, encoder, seed)
                if not (folder / "complete.json").exists():
                    raise ValueError(f"Test closed: {direction}/{encoder}/seed{seed} is not complete")
                c = json.loads((folder / "complete.json").read_text())
                if (c["fingerprint"] != fingerprint or c["epoch"] != cfg["epochs"]
                        or c["selection"] != "fixed_final_epoch" or c["validation_used"]
                        or c["final_sha256"] != task.file_hash(folder / "final.pt")):
                    raise ValueError("Final epoch checkpoint changed")
                records[f"{direction}/{encoder}/seed{seed}"] = c
    return dict(epochs=cfg["epochs"], selection="fixed_final_epoch", runs=records)


def materialize_target(cfg, dc, manifest):
    data, _, _ = task.paths(dc)
    shared = json.loads((task.paths(cfg)[0] / "shared_manifest.json").read_text())
    for path in cfg["tissues"][dc["target_tissue"]].values():
        if task.file_hash(ROOT / path) != shared["source_sha256"][path]:
            raise ValueError("Target source changed since preparation")
    a, _ = load_tissue(cfg, dc["target_tissue"])
    m = pd.read_csv(data / "test/metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
    if (not a.obs_names.equals(m.index)
            or task.file_hash(data / "test/metadata.tsv.gz") != manifest["cell_map_sha256"]["test"]):
        raise ValueError("Target cell order changed")
    panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
    y = task.normalized(a, panel.gene_id)
    if not np.isfinite(y).all():
        raise ValueError("Missing fixed target genes")
    np.save(data / "test/y.npy", y)


def evaluate(cfg, device):
    protocol = final_protocol(cfg)  # All directions/models/seeds, before scoring any target.
    _, out, _ = task.paths(cfg)
    path = out / "test_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Test already opened for a different final checkpoint set")
    task.atomic_json(path, protocol)
    all_rows = []
    for direction in DIRECTIONS:
        dc = direction_config(cfg, direction)
        data, result, _, manifest, fingerprint = task.prepared(dc)
        materialize_target(cfg, dc, manifest)
        panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
        with np.load(data / "train/statistics.npz") as values:
            stats = dict(values)
        rows = []
        for encoder in cfg["models"]:
            dataset = task.load_split(data, manifest, "test", encoder)
            if encoder == next(iter(cfg["models"])):
                for name, baseline in (("control_copy", "control_copy"), ("source_mean_delta", "global_mean_delta")):
                    scores, _ = task.score_model(None, dataset, dc, panel, stats, device, baseline=baseline,
                        full_de=True, save_to=result / "test" / name,
                        groups=dc["target_backgrounds"], training_reference="training_average")
                    rows.extend(dict(method=name, seed=-1, **r) for r in scores)
            for seed in cfg["seeds"]:
                folder, _ = task.run_paths(dc, encoder, seed)
                state = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
                if state["epoch"] != cfg["epochs"] or state["fingerprint"] != fingerprint:
                    raise ValueError("Expected the prespecified final epoch")
                model = task.restore_head(state, dc, len(panel), device)
                scores, _ = task.score_model(model, dataset, dc, panel, stats, device, full_de=True,
                    save_to=result / "test" / encoder / f"seed{seed}",
                    groups=dc["target_backgrounds"], training_reference="training_average")
                rows.extend(dict(method=encoder, seed=seed, **r) for r in scores)
                print(f"TEST {direction}/{encoder}/seed{seed}: RMSE={scores[0]['delta_RMSE']:.6g}", flush=True)
                del model
        frame = pd.DataFrame(rows).assign(direction=direction, source_tissue=dc["source_tissue"],
                                          target_tissue=dc["target_tissue"], output_genes=len(panel))
        frame.to_csv(result / "test_metrics.csv", index=False)
        all_rows.append(frame)
    pd.concat(all_rows, ignore_index=True).to_csv(out / "test_metrics.csv", index=False)
    report(cfg)


def report(cfg):
    _, out, _ = task.paths(cfg)
    metrics = pd.read_csv(out / "test_metrics.csv")
    columns = ["delta_RMSE", "Pearson_delta", "R2_delta_all", "functional__delta_RMSE", "DEG_overlap_20",
               "DEG_overlap_100", "R2_var_all"]
    metrics.groupby(["direction", "method"], sort=False)[columns].agg(["mean", "std"]).to_csv(out / "seed_summary.csv")
    runs = []
    lines = ["# GSE189285肺↔肝：无验证集的固定epoch比较", "",
        f"三encoder冻结、各三个seed，两个方向各自只用源组织拟合。固定{cfg['epochs']}epoch，"
        "AdamW与预定余弦学习率；无validation、无早停、无best epoch或best seed选择，仅final.pt。", "",
        "Loss为群体平均effective Δ-MSE，Δ单位log1p(CP10K)，非负重建口径与基线一致。"
        "两个方向panel仅据各自源组织筛选，因此不直接比较跨方向绝对RMSE；同时看相对基线和固定功能候选。", ""]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    curves, curve_axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, curve_ax, direction in zip(axes, curve_axes, DIRECTIONS):
        dc = direction_config(cfg, direction)
        data, _, _, manifest, _ = task.prepared(dc)
        group = metrics[metrics.direction.eq(direction)]
        summary = group.groupby("method", sort=False)[columns].agg(["mean", "std"])
        names = ["control_copy", "source_mean_delta", *cfg["models"]]
        values = summary.delta_RMSE.reindex(names)
        ax.bar(names, values["mean"], yerr=values["std"].fillna(0), capsize=4)
        ax.set(title=direction, ylabel="Mean delta RMSE")
        ax.tick_params(axis="x", labelrotation=25)
        lines += [f"## {direction}", "", f"输出panel={manifest['output_genes']}，功能候选={manifest['predefined_functional_genes']}，"
            f"源组织全零但预先保留的候选={manifest['train_unsupported_panel_genes']}。", "",
            "| 方法 | Δ-RMSE ↓ | Δ-Pearson ↑ | 功能候选Δ-RMSE ↓ | DEG overlap@100 ↑ |",
            "|---|---:|---:|---:|---:|"]
        for name in names:
            cells = []
            for key in ("delta_RMSE", "Pearson_delta", "functional__delta_RMSE", "DEG_overlap_100"):
                mean, std = summary.loc[name, (key, "mean")], summary.loc[name, (key, "std")]
                cells.append("—" if not np.isfinite(mean) else f"{mean:.6f} ± {std:.6f}" if np.isfinite(std) else f"{mean:.6f}")
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
        for encoder in cfg["models"]:
            for seed in cfg["seeds"]:
                folder, result = task.run_paths(dc, encoder, seed)
                complete = json.loads((folder / "complete.json").read_text())
                h = pd.read_csv(result / "history.csv")
                runs.append(dict(direction=direction, encoder=encoder, seed=seed, epochs=complete["epoch"],
                    final_train_MSE=h.train_macro_delta_MSE.iloc[-1], final_sha256=complete["final_sha256"]))
                curve_ax.plot(h.epoch, h.train_macro_delta_MSE, label=f"{encoder}/{seed}")
        curve_ax.set(title=direction, xlabel="Epoch", ylabel="Source training delta MSE", yscale="log")
        curve_ax.legend(fontsize=6)
        lines.append("")
    for plot, filename in ((fig, "test_delta_RMSE.png"), (curves, "training_curves.png")):
        plot.tight_layout()
        plot.savefig(out / filename, dpi=180)
        plt.close(plot)
    pd.DataFrame(runs).to_csv(out / "training_runs.csv", index=False)
    lines += ["## 边界", "",
        "- 每组织×条件一个混池文库；随机细胞mini-batch不是独立实验重复。seed SD不是动物级CI。",
        "- 两方向角色独立，但同一组织IL33在反向实验中是训练数据；此为已开发数据上的双向比较，不是从未读取过目标的盲测。",
        "- 所有18组final checkpoints固定后才统一评分；测试不选择epoch、seed、缩放系数或输出基因。",
        "- 只监督平均响应，不能据此证明单细胞命运、完整分布或因果基因关系。",
        "- 旧版15epoch线性/PCA头的结果属于另一个实验，不混入本版比较。", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"REPORT {out / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "train", "test", "report", "all"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--direction", choices=list(DIRECTIONS), help="Limit train/check to one direction")
    parser.add_argument("--encoder", choices=["GeneCompass", "ILC2_official", "ILC2_tissue"], help="Train only")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], help="Train only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if ((args.direction and args.command not in ("train", "check"))
            or ((args.encoder or args.seed is not None) and args.command != "train")):
        parser.error("Filters apply to train; --direction also applies to check. Test/all use both complete directions.")
    cfg = configuration(args.config)
    torch.set_num_threads(cfg["cpu_threads"])
    if args.command in ("prepare", "all"):
        prepare(cfg)
    if args.command == "prepare":
        return
    ready(cfg)
    if args.command == "report":
        report(cfg)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; explicitly choose --device cpu")
    if args.command == "check":
        for direction in ([args.direction] if args.direction else DIRECTIONS):
            task.check(direction_config(cfg, direction), device)
    if args.command in ("train", "all"):
        for direction in ([args.direction] if args.direction else DIRECTIONS):
            dc = direction_config(cfg, direction)
            for encoder in ([args.encoder] if args.encoder else cfg["models"]):
                for seed in ([args.seed] if args.seed is not None else cfg["seeds"]):
                    train_fixed(dc, encoder, seed, device, args.resume)
    if args.command in ("test", "all"):
        evaluate(cfg, torch.device("cpu") if args.command == "all" else device)


if __name__ == "__main__":
    main()
