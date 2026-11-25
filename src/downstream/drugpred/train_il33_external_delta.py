"""80/20 source-cell split and locked external-study evaluation of direct delta.

prepare reads only external IL33 obs/var annotations; its counts are opened by
test after all nine heads complete. The existing mean-response trainer is reused.
"""
import argparse
import gc
import json
import pickle

import h5py
import anndata as ad
import numpy as np
import pandas as pd
import torch

import train_il33_mixed_delta as task
from il33_official_extension import align_features

ROOT = task.ROOT
CONFIG = "configs/drugpred_il33_external_delta.json"
OBS_COLUMNS = ["study", "response_group", "library_id", "condition", "tissue", "qc_version", "split",
               "source_h5ad", "source_cell_id", "source_row"]


def configuration(path=CONFIG):
    extension = json.loads((ROOT / path).read_text())
    base = json.loads((ROOT / extension.pop("base_config")).read_text())
    cfg = dict(base, **extension)
    cfg["models"] = dict(GeneCompass=base["models"]["GeneCompass"],
        ILC2_official=cfg.pop("official_checkpoint"), ILC2_tissue=base["models"]["ILC2_tissue"])
    return base, cfg


def new_roles(meta, roles):
    meta = meta.copy()
    for col in ("split", "role", "data_role", "evaluation_role", "allowed_for_fit", "allowed_for_tuning", "split_stratum"):
        if col in meta:
            meta[f"previous_{col}"] = meta[col]
    for col in ("split", "role", "data_role", "evaluation_role"):
        meta[col] = roles
    meta["allowed_for_fit"] = meta.split.eq("train")
    meta["allowed_for_tuning"] = meta.split.eq("validation")
    meta["response_pair_id"] = meta.response_group
    return meta


def annotations(path):
    # No X/layers access: external gene presence is structural, not detection.
    with h5py.File(path, "r") as handle:
        return ad.io.read_elem(handle["obs"]), ad.io.read_elem(handle["var"])


def valid_mouse_ids(ids):
    ids = task._gene_ids(ids)
    if not ids.is_unique or not ids.str.match(r"^ENSMUSG\d+$").all():
        raise ValueError("Expected unique stable mouse gene IDs")
    return ids


def raw_object(path, metadata):
    a = ad.read_h5ad(path)
    if not a.obs_names.equals(metadata.index):
        raise ValueError(f"Source cell order changed: {path}")
    a.var_names = valid_mouse_ids(a.var_names)
    a.X = task._raw_counts(a.X)
    if "counts" not in a.layers or (a.X != a.layers["counts"]).nnz:
        raise ValueError("Full raw counts and counts layer disagree")
    return a


def prepare(base, cfg):
    parent, _, _, parent_manifest, parent_hash = task.prepared(base)
    data, out, _ = task.paths(cfg)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = data / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise ValueError("Configuration changed; use a new version")
    if (data / "prepare_manifest.json").exists():
        task.prepared(cfg)
        print("External split already prepared", flush=True)
        return
    task.atomic_json(snapshot, cfg)
    meta = pd.read_csv(parent / "cell_manifest.tsv.gz", sep="\t", index_col=0, low_memory=False)
    split = task.mixed_splits(meta, cfg["split_fractions"], cfg["split_seed"], labels=("train", "validation"))
    meta = new_roles(meta, split)
    meta["split_stratum"] = meta[["study", "response_group", "library_id", "condition"]].astype(str).agg("|".join, axis=1)
    external_parts, external_var = [], []
    for condition, key in (("PBS", "external_control"), ("IL33", "external_target")):
        path = ROOT / cfg[key]
        obs, var = annotations(path)
        m = pd.read_csv(path.parent / "metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
        if not obs.index.equals(m.index) or not obs[["study", "response_group", "library_id", "condition"]].astype(str).equals(
                m[["study", "response_group", "library_id", "condition"]].astype(str)):
            raise ValueError("External source metadata changed")
        if not (m.study.eq("GSE189285") & m.condition.eq(condition)).all():
            raise ValueError("Unexpected external study/condition")
        m = new_roles(m, "test")
        m["source_h5ad"], m["source_cell_id"], m["source_row"] = cfg[key], m.index, np.arange(len(m))
        external_parts.append(m)
        external_var.append(valid_mouse_ids(var.index))
    external = pd.concat(external_parts)
    combined = pd.concat([meta, external])
    if (not combined.index.is_unique or set(meta.study) != set(base["sources"])
            or set(meta.response_group) != set(cfg["backgrounds"])
            or set(external.response_group) != set(cfg["external_backgrounds"])
            or not combined.tissue.eq("Lung").all() or not combined.qc_version.eq("ilc2_balanced_v1").all()
            or not combined.selected_ILC2.eq(True).all()):
        raise ValueError("Cell uniqueness, provenance or existing balanced_v1 QC failed")
    for group in cfg["backgrounds"]:
        for role in ("train", "validation"):
            if set(meta.loc[meta.response_group.eq(group) & meta.split.eq(role), "condition"]) != {"PBS", "IL33"}:
                raise ValueError(f"Missing within-background PBS/IL33: {group}/{role}")
    a = raw_object(parent / "full_counts.h5ad", meta)
    if int(a.X.sum(dtype=np.int64)) != parent_manifest["raw_UMI"]:
        raise ValueError("Source UMI total changed")
    a.var["present_GSE189285"] = a.var_names.isin(external_var[0].intersection(external_var[1]))
    functional = pd.read_csv(parent / "functional_gene_sets.tsv", sep="\t")
    panel = task.select_panel(a.X, a.var, meta, cfg["backgrounds"],
        [*base["sources"], "GSE189285"], functional.gene_id, cfg["minimum_train_detection"])
    cache = ROOT / cfg["encoder_cache"]
    signature = json.loads((cache / "signature.json").read_text())
    tok_cfg = json.loads((ROOT / cfg["tokenization_config"]).read_text())
    tokens = pickle.loads((ROOT / tok_cfg["token_dictionary"]).read_bytes())
    medians = pickle.loads((ROOT / tok_cfg["gene_median_reference"]).read_bytes())
    if (signature["encoding"] != "median" or signature["max_gene_tokens"] != 2048
            or signature["precision"] != "float32" or signature["pooling"] != "masked_gene_mean"
            or signature["tokens"] != task.file_hash(ROOT / tok_cfg["token_dictionary"])
            or signature["medians"] != task.file_hash(ROOT / tok_cfg["gene_median_reference"])):
        raise ValueError("Frozen feature tokenization/pooling differs")
    checkpoints = {name: task.file_hash(ROOT / path / "pytorch_model.bin") for name, path in cfg["models"].items()}
    if any(sha != signature["checkpoints"][name] for name, sha in checkpoints.items()):
        raise ValueError("Encoder checkpoint differs from cached features")
    for name, sha in parent_manifest["feature_cache_sha256"].items():
        if task.file_hash(cache / name / "PBS_h.npy") != sha:
            raise ValueError("Previously verified encoder cache changed")
    panel["in_encoder_vocab"] = panel.gene_id.isin(tokens)
    panel.to_csv(data / "gene_panel.tsv", sep="\t", index=False)
    functional.to_csv(data / "functional_gene_sets.tsv", sep="\t", index=False)
    combined.to_csv(data / "cell_manifest.tsv.gz", sep="\t", index_label="cell_id")
    counts = combined.groupby(["study", "response_group", "library_id", "condition", "split"], observed=True).size().unstack(fill_value=0)
    counts.to_csv(out / "split_counts.tsv", sep="\t")
    counts.groupby(level="condition").sum().to_csv(out / "condition_counts.tsv", sep="\t")
    a.obs = meta[OBS_COLUMNS].copy()
    a.uns = dict(metadata=dict(version=cfg["version"], full_metadata="cell_manifest.tsv.gz", counts="full raw integer UMI"))
    a.write_h5ad(data / "source_full_counts.h5ad", compression="gzip")
    cache_metadata = pd.read_csv(cache / "PBS_metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
    hashes, feature_hashes = {}, {}
    for role in task.ROLES:
        folder = data / role
        folder.mkdir(exist_ok=True)
        m = combined[combined.split.eq(role)]
        m.to_csv(folder / "metadata.tsv.gz", sep="\t", index_label="cell_id")
        hashes[role] = task.file_hash(folder / "metadata.tsv.gz")
        if role != "test":
            y = task.normalized(a[meta.index.get_indexer(m.index)], panel.gene_id)
            if not np.isfinite(y).all():
                raise ValueError("A fixed target gene is missing")
            np.save(folder / "y.npy", y)
            if role == "train":
                statistics = task.training_statistics(y, m, panel, cfg["backgrounds"], out / "training_DEG")
                np.savez_compressed(folder / "statistics.npz", **statistics)
            del y
        for name in cfg["models"]:
            feature_path = cache / name / "PBS_h.npy"
            h = align_features(cache_metadata, m, np.load(feature_path, mmap_mode="r"))
            np.save(folder / f"{name}_h.npy", h)
            if role == "train":
                feature_hashes[name] = task.file_hash(feature_path)
        print(f"Prepared {role}: {len(m)} cells, PBS={int(m.condition.eq('PBS').sum())}", flush=True)
    del a
    gc.collect()
    control = raw_object(ROOT / cfg["external_control"], external_parts[0])
    col, median, _ = task.map_genes(control.var_names, tokens, medians)
    cached_inputs = np.load(cache / "PBS_inputs.npz")
    loc = cache_metadata.index.get_indexer(control.obs_names)
    for left in range(0, control.n_obs, 512):
        actual = task.encode(control.X[left:left+512], col, median, 2048, "median")
        for key, values in zip(("ids", "vals", "length"), actual):
            if not np.array_equal(values, cached_inputs[key][loc[left:left+512]]):
                raise ValueError("External PBS inputs differ from the frozen encoder cache")
    np.save(data / "test/control_y.npy", task.normalized(control, panel.gene_id))
    control.obs = external_parts[0][OBS_COLUMNS].copy()
    control.uns = dict(metadata=dict(version=cfg["version"], full_metadata="cell_manifest.tsv.gz", role="test_input"))
    control.write_h5ad(data / "external_control_counts.h5ad", compression="gzip")
    pd.Series(external_var[1], name="gene_id").to_csv(data / "external_target_gene_ids.tsv", sep="\t", index=False)
    manifest = dict(config=cfg, parent_manifest_sha256=parent_hash, checkpoint_sha256=checkpoints,
        feature_cache_sha256=feature_hashes, panel_sha256=task.file_hash(data / "gene_panel.tsv"), cell_map_sha256=hashes,
        training_statistics_sha256=task.file_hash(data / "train/statistics.npz"), cells=len(combined),
        output_genes=len(panel), predefined_functional_genes=int(panel.functional.sum()),
        train_unsupported_panel_genes=int((~panel.train_supported).sum()), source_raw_UMI=parent_manifest["raw_UMI"],
        source_genes_full=parent_manifest["genes_full"], external_control_raw_UMI=int(control.X.sum(dtype=np.int64)),
        external_gene_annotation_used=True, external_target_expression_read=False,
        external_PBS_inputs_verified=control.n_obs, source_PBS_inputs_previously_verified=parent_manifest["real_encoder_input_cells_verified"],
        roles_by_condition=combined.groupby(["condition", "split"]).size().unstack(fill_value=0).to_dict(),
        split_scope="within-source cell validation plus external study test; historical external evaluation already opened")
    task.atomic_json(data / "prepare_manifest.json", manifest)
    task.atomic_json(out / "prepare_report.json", manifest)
    print(f"PREPARE COMPLETE: fixed panel={len(panel)}; external IL33 counts unopened", flush=True)


def materialize_test(data, cfg, manifest):
    """Called only after all configured heads and their protocol are locked."""
    m = pd.read_csv(data / "test/metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
    if task.file_hash(data / "test/metadata.tsv.gz") != manifest["cell_map_sha256"]["test"]:
        raise ValueError("External cell map changed")
    target_meta = m[m.condition.eq("IL33")]
    target = raw_object(ROOT / cfg["external_target"], target_meta)
    expected_genes = pd.read_csv(data / "external_target_gene_ids.tsv", sep="\t").gene_id
    if not np.array_equal(target.var_names, expected_genes):
        raise ValueError("External target gene annotation changed")
    panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
    y = np.empty((len(m), len(panel)), dtype=np.float32)
    y[m.condition.eq("PBS").to_numpy()] = np.load(data / "test/control_y.npy")
    y[m.condition.eq("IL33").to_numpy()] = task.normalized(target, panel.gene_id)
    if not np.isfinite(y).all():
        raise ValueError("Missing external target genes")
    np.save(data / "test/y.npy", y)
    target.obs = target_meta[OBS_COLUMNS].copy()
    target.uns = dict(metadata=dict(version=cfg["version"], full_metadata="cell_manifest.tsv.gz", role="test_target"))
    target.write_h5ad(data / "test/target_full_counts.h5ad", compression="gzip")


def test(cfg, device):
    data, out, _, manifest, fingerprint = task.prepared(cfg)
    protocol = task.checkpoint_lock(cfg, fingerprint)
    path = out / "test_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("External test already opened for different heads")
    task.atomic_json(path, protocol)
    materialize_test(data, cfg, manifest)
    panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
    with np.load(data / "train/statistics.npz") as values:
        statistics = dict(values)
    rows = []
    for encoder in cfg["models"]:
        dataset = task.load_split(data, manifest, "test", encoder)
        if encoder == next(iter(cfg["models"])):
            for baseline in ("control_copy", "global_mean_delta"):
                scores, _ = task.score_model(None, dataset, cfg, panel, statistics, device, full_de=True,
                    save_to=out / "test" / baseline, baseline=baseline,
                    groups=cfg["external_backgrounds"], training_reference="training_average")
                rows.extend(dict(method=baseline, seed=-1, **r) for r in scores)
        for seed in cfg["seeds"]:
            model_dir, _ = task.run_paths(cfg, encoder, seed)
            state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
            model = task.restore_head(state, cfg, len(panel), device)
            scores, _ = task.score_model(model, dataset, cfg, panel, statistics, device, full_de=True,
                save_to=out / "test" / encoder / f"seed{seed}",
                groups=cfg["external_backgrounds"], training_reference="training_average")
            rows.extend(dict(method=encoder, seed=seed, **r) for r in scores)
            print(f"External test scored: {encoder}/seed{seed}", flush=True)
    pd.DataFrame(rows).to_csv(out / "test_background_metrics.csv", index=False)
    task.report(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "train", "test", "report"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--encoder", choices=["GeneCompass", "ILC2_official", "ILC2_tissue"])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base, cfg = configuration(args.config)
    torch.set_num_threads(cfg["cpu_threads"])
    if args.command == "prepare":
        prepare(base, cfg)
    elif args.command == "report":
        task.report(cfg)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable; select --device cpu explicitly")
        if args.command == "check":
            task.check(cfg, device)
        elif args.command == "test":
            test(cfg, device)
        else:
            for encoder in ([args.encoder] if args.encoder else cfg["models"]):
                for seed in ([args.seed] if args.seed is not None else cfg["seeds"]):
                    task.train_one(cfg, encoder, seed, device, args.resume)


if __name__ == "__main__":
    main()
