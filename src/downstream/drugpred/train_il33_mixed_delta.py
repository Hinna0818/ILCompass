"""Fixed-panel, mixed-cell IL33 direct-delta workflow: prepare/check/train/test/report.

Training and check never load test expression; final scoring requires all heads.
Counts remain intact; normalization and frozen per-cell features are separate.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ilc2_mpl")
import argparse
import fcntl
import gc
import json
import pickle
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from posttrain_data import SPLITS, assign_splits, encode, file_hash
from genecompass_coverage import map_genes
from qc_utils import _gene_ids, _raw_counts
from residual import ResidualPredictor
from prepare_il33_expanded_compare import normalized
from il33_lung_metrics import differential, evaluate

DEFAULT_CONFIG = "configs/drugpred_il33_mixed_delta.json"
ROLES = ("train", "validation", "test")


def atomic_json(path, value):
    partial = path.with_suffix(".partial.json")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    partial.replace(path)


def atomic_torch(path, value):
    partial = path.with_suffix(".partial.pt")
    torch.save(value, partial)
    partial.replace(path)


def paths(cfg):
    return tuple(ROOT / cfg[k] for k in ("data_directory", "result_directory", "model_directory"))


def mixed_splits(meta, fractions, seed, labels=ROLES):
    meta = meta.copy()
    meta["split_stratum"] = meta[["study", "response_group", "library_id", "condition"]].astype(str).agg("|".join, axis=1)
    return assign_splits(meta, fractions, seed, labels=labels)


def select_panel(counts, var, meta, backgrounds, studies, functional_ids, minimum):
    """Only train expression selects genes; feature-space presence is structural."""
    common = var[[f"present_{s}" for s in studies]].all(axis=1).to_numpy()
    detection = {}
    for group in backgrounds:
        for condition in ("PBS", "IL33"):
            rows = (meta.split.eq("train") & meta.response_group.eq(group) & meta.condition.eq(condition)).to_numpy()
            if not rows.any():
                raise ValueError(f"Missing training {group}/{condition}")
            detection[f"train_detection_{group}_{condition}"] = np.asarray((counts[rows] > 0).mean(axis=0)).ravel()
    det = pd.DataFrame(detection, index=var.index)
    functional = var.index.isin(functional_ids)
    expressed = det.max(axis=1).ge(minimum).to_numpy()
    selected = common & (expressed | functional)
    panel = var.loc[selected, ["gene_symbol"]].copy().join(det.loc[selected])
    panel.index.name = "gene_id"
    panel = panel.sort_index().reset_index()
    panel.insert(0, "output_index", np.arange(len(panel)))
    panel["functional"] = panel.gene_id.isin(functional_ids)
    panel["train_supported"] = panel[list(det.columns)].max(axis=1).gt(0)
    panel["selection_reason"] = np.where(panel[list(det.columns)].max(axis=1).ge(minimum), "train_detection", "predefined_functional")
    if not len(panel):
        raise ValueError("Empty output panel")
    return panel


def training_statistics(y, meta, panel, backgrounds, output):
    if not meta.split.eq("train").all():
        raise ValueError("Response targets and training DEGs require train cells only")
    output.mkdir(parents=True, exist_ok=True)
    statistics = {}
    for group in backgrounds:
        ctrl = y[(meta.response_group.eq(group) & meta.condition.eq("PBS")).to_numpy()]
        target = y[(meta.response_group.eq(group) & meta.condition.eq("IL33")).to_numpy()]
        if not len(ctrl) or not len(target):
            raise ValueError(f"Missing matched training PBS/IL33 in {group}")
        statistics[group + "__delta"] = target.mean(0, dtype=np.float64) - ctrl.mean(0, dtype=np.float64)
        de, order = differential(target, ctrl, np.ones(len(panel), dtype=bool))
        de.insert(0, "gene_id", panel.gene_id)
        de.to_csv(output / f"{group}.tsv.gz", sep="\t", index=False)
        statistics[group + "__DE20"] = order[:20]
        statistics[group + "__DE100"] = order[:100]
    return statistics


class DirectDeltaHead(ResidualPredictor):
    def __init__(self, train_h, genes, hidden_size):
        # Reuse the existing residual head with a permanently zero mean offset.
        super().__init__(train_h, train_h.new_zeros(2, genes), hidden_size)
        self.condition = nn.Sequential(nn.Linear(2, 32, bias=False), nn.GELU(), nn.Linear(32, 32, bias=False))

    def forward(self, h):
        condition = h.new_tensor([1., 0.]).expand(len(h), -1)
        return super().forward(h, h.new_zeros(len(h), self.mean_effects.shape[1]), condition)


class ExpressionDeltaMLP(nn.Module):
    """Full fixed-panel log expression + stimulus -> delta, without an encoder."""
    def __init__(self, genes, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(genes+2, hidden_size), nn.GELU(), nn.Linear(hidden_size, genes))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, condition=None):
        # This experiment has only IL33 targets; the code does not imply unseen-stimulus generalization.
        if condition is None:
            condition = x.new_tensor([1., 0.]).expand(len(x), -1)
        return self.net(torch.cat([x, condition], dim=1))


def effective_delta(x, raw_delta):
    return (x + raw_delta).clamp_min(0) - x


def delta_loss(x, raw_delta, target):
    return (effective_delta(x, raw_delta).mean(0) - target).square().mean()


def epoch_batches(meta, backgrounds, batch_size, rng):
    """Each control once; factors sum to equal-background weights, not cell weights."""
    batches = []
    for group in backgrounds:
        rows = np.flatnonzero(meta.response_group.eq(group))
        order = rng.permutation(rows)
        batches.extend((group, order[i:i+batch_size], len(order[i:i+batch_size]) / len(rows) / len(backgrounds))
                       for i in range(0, len(order), batch_size))
    return [batches[j] for j in rng.permutation(len(batches))]


def fit_epoch(model, optimizer, controls, h, control_y, targets, cfg, rng, device):
    model.train()
    batches = epoch_batches(controls, cfg["backgrounds"], cfg["batch_size"], rng)
    batch_loss, grad_norm = 0., 0.
    for group, idx, factor in batches:
        hh = torch.as_tensor(np.array(h[idx]), device=device)
        xx = torch.as_tensor(np.array(control_y[idx]), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = delta_loss(xx, model(hh), targets[group])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        (loss * factor * len(batches)).backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        batch_loss += float(loss.detach()) * factor
        grad_norm += float(norm) / len(batches)
    return batch_loss, grad_norm


def prepare(cfg):
    data, out, _ = paths(cfg)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = data / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise ValueError("This data version has a different configuration; use a new version")
    atomic_json(snapshot, cfg)
    if (data / "prepare_manifest.json").exists():
        print("Prepared version already complete", flush=True)
        return
    cache = ROOT / cfg["encoder_cache"]
    signature = json.loads((cache / "signature.json").read_text())
    tok_cfg = json.loads((ROOT / cfg["tokenization_config"]).read_text())
    tokens = pickle.loads((ROOT / tok_cfg["token_dictionary"]).read_bytes())
    medians = pickle.loads((ROOT / tok_cfg["gene_median_reference"]).read_bytes())
    if any(signature[k] != file_hash(ROOT / tok_cfg[v]) for k, v in
           [("tokens", "token_dictionary"), ("medians", "gene_median_reference")]):
        raise ValueError("Encoder cache token/median reference differs")
    if signature["encoding"] != tok_cfg["encoding"] or signature["max_gene_tokens"] != tok_cfg["max_gene_tokens"]:
        raise ValueError("Encoder cache tokenization differs")
    checkpoints = {name: file_hash(ROOT / path / "pytorch_model.bin") for name, path in cfg["models"].items()}
    if any(sha != signature["checkpoints"][name] for name, sha in checkpoints.items()):
        raise ValueError("Encoder cache checkpoint differs")
    meta = pd.concat([pd.read_csv(cache / f"{role}_metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
                      for role in ("train", "development")])
    if (not meta.index.is_unique or set(meta.study) != set(cfg["sources"])
            or set(meta.response_group) != set(cfg["backgrounds"]) or not meta.condition.isin(["PBS", "IL33"]).all()
            or not meta.tissue.eq("Lung").all() or not meta.qc_version.eq("ilc2_balanced_v1").all()
            or not meta.selected_ILC2.all()):
        raise ValueError("Unexpected source cells/QC; this version requires the two reviewed lung studies")
    new_split = mixed_splits(meta, cfg["split_fractions"], cfg["split_seed"])
    for col in ("split", "role", "data_role", "evaluation_role", "allowed_for_fit", "allowed_for_tuning"):
        if col in meta:
            meta[f"previous_{col}"] = meta[col]
    meta["split"] = new_split
    meta["role"] = new_split
    meta["evaluation_role"] = new_split
    meta["allowed_for_fit"] = new_split.eq("train")
    meta["allowed_for_tuning"] = new_split.eq("validation")
    cache_meta = pd.read_csv(cache / "PBS_metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
    cache_inputs = np.load(cache / "PBS_inputs.npz")
    parts, symbols, source_hashes = [], {}, {}
    total_umi = verified = 0
    for study, path in cfg["sources"].items():
        source_hashes[study] = file_hash(ROOT / path)
        a = ad.read_h5ad(ROOT / path)
        ids = _gene_ids(a.var_names)
        if not ids.is_unique or not ids.str.match(r"^ENSMUSG\d+$").all():
            raise ValueError(f"{study}: ambiguous/non-mouse stable gene IDs")
        a.var_names = ids
        chosen = meta.index[meta.study.eq(study)]
        loc = a.obs_names.get_indexer(chosen)
        if (loc < 0).any():
            raise ValueError(f"{study}: source cell IDs missing")
        x = _raw_counts(a.layers["counts"][loc])
        meta.loc[chosen, "source_h5ad"] = path
        meta.loc[chosen, "source_cell_id"] = chosen
        meta.loc[chosen, "source_row"] = loc
        symbols[study] = pd.Series(a.var.gene_symbol.astype(str).to_numpy(), index=ids)
        col, median, _ = map_genes(ids, tokens, medians)
        pbs_rows = np.flatnonzero(meta.loc[chosen].condition.eq("PBS"))
        for left in range(0, len(pbs_rows), 512):
            rows = pbs_rows[left:left+512]
            ii, vv, ll = encode(x[rows], col, median, tok_cfg["max_gene_tokens"], tok_cfg["encoding"])
            old = cache_meta.index.get_indexer(chosen[rows])
            if ((old < 0).any() or not np.array_equal(ii, cache_inputs["ids"][old])
                    or not np.array_equal(vv, cache_inputs["vals"][old]) or not np.array_equal(ll, cache_inputs["length"][old])):
                raise ValueError(f"{study}: raw source no longer reproduces cached encoder inputs")
            verified += len(rows)
        total_umi += int(x.sum(dtype=np.int64))
        parts.append(ad.AnnData(x, obs=pd.DataFrame(index=chosen), var=pd.DataFrame(index=ids)))
        del a
        gc.collect()
        print(f"Loaded intact counts and verified PBS tokens: {study}, {x.shape}", flush=True)
    a = ad.concat(parts, join="outer", fill_value=0)
    del parts
    a = a[meta.index].copy()
    if a.n_obs != len(meta) or int(a.X.sum(dtype=np.int64)) != total_umi:
        raise ValueError("Counts/cells lost during source merge")
    annotations = pd.concat(symbols, axis=1).reindex(a.var_names)
    a.var["gene_symbol"] = annotations.bfill(axis=1).iloc[:, 0]
    a.var["gene_symbol_conflict"] = annotations.nunique(axis=1).gt(1)
    for study in cfg["sources"]:
        a.var[f"present_{study}"] = annotations[study].notna()
    a.obs = meta[["study", "response_group", "library_id", "condition", "tissue", "qc_version", "split",
                  "source_h5ad", "source_cell_id", "source_row"]].copy()
    a.layers["counts"] = a.X.copy()
    a.uns["metadata"] = dict(version=cfg["version"], counts="X and layers/counts: full-gene raw integer UMI",
                             full_metadata="cell_manifest.tsv.gz", split_scope="within-library cells; not independent animals")
    functional = pd.read_csv(ROOT / cfg["functional_gene_sets"])
    panel = select_panel(a.X, a.var, meta, cfg["backgrounds"], cfg["sources"], functional.gene_id, cfg["minimum_train_detection"])
    panel["in_encoder_vocab"] = panel.gene_id.isin(tokens)
    panel.to_csv(data / "gene_panel.tsv", sep="\t", index=False)
    functional.to_csv(data / "functional_gene_sets.tsv", sep="\t", index=False)
    meta.to_csv(data / "cell_manifest.tsv.gz", sep="\t", index_label="cell_id")
    meta.groupby(["study", "response_group", "library_id", "condition", "split"], observed=True).size().unstack(fill_value=0).to_csv(out / "split_counts.tsv", sep="\t")
    partial = data / "full_counts.partial.h5ad"
    a.write_h5ad(partial, compression="gzip")
    partial.replace(data / "full_counts.h5ad")
    split_hashes = {}
    feature_hashes = {name: file_hash(cache / name / "PBS_h.npy") for name in cfg["models"]}
    for role in ROLES:
        folder = data / role
        folder.mkdir(exist_ok=True)
        rows = np.flatnonzero(meta.split.eq(role))
        m = meta.iloc[rows]
        m.to_csv(folder / "metadata.tsv.gz", sep="\t", index_label="cell_id")
        y = normalized(a[rows], panel.gene_id)
        if not np.isfinite(y).all():
            raise ValueError("A fixed output gene is missing from the source feature space")
        np.save(folder / "y.npy", y)
        pbs = m.index[m.condition.eq("PBS")]
        loc = cache_meta.index.get_indexer(pbs)
        for name in cfg["models"]:
            h = np.asarray(np.load(cache / name / "PBS_h.npy", mmap_mode="r")[loc])
            if h.shape != (len(pbs), 768) or not np.isfinite(h).all():
                raise ValueError("Invalid encoder features")
            np.save(folder / f"{name}_h.npy", h)
        split_hashes[role] = file_hash(folder / "metadata.tsv.gz")
        print(f"Prepared {role}: {len(m)} cells, {len(pbs)} PBS, fixed panel {len(panel)}", flush=True)
        if role == "train":
            statistics = training_statistics(y, m, panel, cfg["backgrounds"], out / "training_DEG")
            np.savez_compressed(folder / "statistics.npz", **statistics)
        del y
    manifest = dict(config=cfg, sources_sha256=source_hashes, checkpoint_sha256=checkpoints,
        feature_cache_sha256=feature_hashes, functional_source_sha256=file_hash(ROOT / cfg["functional_gene_sets"]),
        panel_sha256=file_hash(data / "gene_panel.tsv"), cell_map_sha256=split_hashes,
        training_statistics_sha256=file_hash(data / "train/statistics.npz"),
        cells=len(meta), genes_full=a.n_vars, raw_UMI=total_umi, output_genes=len(panel),
        common_measured_genes=int(a.var[[f"present_{s}" for s in cfg["sources"]]].all(axis=1).sum()),
        predefined_functional_genes=int(panel.functional.sum()), train_unsupported_panel_genes=int((~panel.train_supported).sum()),
        real_encoder_input_cells_verified=verified, test_used_for_panel_or_training_statistics=False,
        split_scope="Seen-library cell-level internal test; previously developed datasets, not new blind validation")
    atomic_json(data / "prepare_manifest.json", manifest)
    atomic_json(out / "prepare_report.json", manifest)
    print(f"PREPARE COMPLETE: fixed output genes={len(panel)}, verified PBS={verified}", flush=True)


def prepared(cfg):
    data, out, models = paths(cfg)
    manifest = json.loads((data / "prepare_manifest.json").read_text())
    if cfg != manifest["config"] or file_hash(data / "gene_panel.tsv") != manifest["panel_sha256"]:
        raise ValueError("Prepared configuration/panel changed; use a new version")
    if file_hash(data / "train/statistics.npz") != manifest["training_statistics_sha256"]:
        raise ValueError("Train-derived targets changed")
    return data, out, models, manifest, file_hash(data / "prepare_manifest.json")


def load_expression_split(data, manifest, role):
    folder = data / role
    if file_hash(folder / "metadata.tsv.gz") != manifest["cell_map_sha256"][role]:
        raise ValueError(f"{role}: cell map changed")
    m = pd.read_csv(folder / "metadata.tsv.gz", sep="\t", index_col=0, low_memory=False)
    y = np.load(folder / "y.npy", mmap_mode="r")
    if not m.split.eq(role).all() or y.shape != (len(m), manifest["output_genes"]):
        raise ValueError(f"{role}: input rows/roles/panel disagree")
    return m, y


def expression_dataset(data, manifest, role):
    m, y = load_expression_split(data, manifest, role)
    return m, y, y[m.condition.eq("PBS").to_numpy()]


def load_split(data, manifest, role, encoder):
    m, y = load_expression_split(data, manifest, role)
    h = np.load(data / role / f"{encoder}_h.npy", mmap_mode="r")
    dimensions = manifest.get("encoder_dimensions", {}).get(encoder, 768)
    if h.shape != (int(m.condition.eq("PBS").sum()), dimensions):
        raise ValueError(f"{role}: encoder rows disagree")
    return m, y, h


@torch.no_grad()
def predict_delta(model, h, x, batch_size, device):
    model.eval()
    result = np.empty_like(x, dtype=np.float32)
    for left in range(0, len(x), batch_size):
        xx = torch.as_tensor(np.array(x[left:left+batch_size]), device=device)
        hh = torch.as_tensor(np.array(h[left:left+batch_size]), device=device)
        result[left:left+len(xx)] = effective_delta(xx, model(hh)).cpu().numpy()
    return result


def subset_metrics(pred, truth, ctrl, indices):
    if not len(indices):
        return dict(genes=0, delta_RMSE=None, Pearson_delta=None, direction_accuracy=None)
    metrics, moments = evaluate(pred[:, indices], truth[:, indices], ctrl[:, indices], [], [], top_n=())
    dt, dp = moments["observed_delta"], moments["predicted_delta"]
    active = dt != 0
    return dict(genes=len(indices), delta_RMSE=metrics["delta_RMSE"], Pearson_delta=metrics["Pearson_delta"],
                direction_accuracy=float(np.mean(np.sign(dp[active]) == np.sign(dt[active]))) if active.any() else None)


def aggregate_metrics(rows):
    mse = float(np.mean([r["delta_MSE"] for r in rows]))
    correlations = [r["Pearson_delta"] for r in rows
                    if r["Pearson_delta"] is not None and np.isfinite(r["Pearson_delta"])]
    return dict(macro_delta_MSE=mse, sqrt_macro_delta_MSE=float(np.sqrt(mse)),
                mean_background_delta_RMSE=float(np.mean([r["delta_RMSE"] for r in rows])),
                macro_Pearson_delta=float(np.mean(correlations)) if correlations else None)


def score_model(model, dataset, cfg, panel, statistics, device, *, full_de=False, save_to=None, baseline=None,
                groups=None, training_reference="background"):
    if training_reference not in ("background", "training_average"):
        raise ValueError("Unknown training reference")
    if baseline == "background_mean_delta" and training_reference != "background":
        raise ValueError("An external background has no training mean delta")
    m, y, h = dataset
    pm = m[m.condition.eq("PBS")]
    deltas = np.empty((len(pm), len(panel)), dtype=np.float32) if save_to is not None else None
    rows = []
    global_delta = np.mean([statistics[g + "__delta"] for g in cfg["backgrounds"]], axis=0)
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)
    for group in cfg["backgrounds"] if groups is None else groups:
        source = np.flatnonzero(pm.response_group.eq(group))
        ctrl = np.asarray(y[(m.response_group.eq(group) & m.condition.eq("PBS")).to_numpy()])
        truth = np.asarray(y[(m.response_group.eq(group) & m.condition.eq("IL33")).to_numpy()])
        if not len(ctrl) or not len(truth):
            raise ValueError(f"Missing matched evaluation PBS/IL33 in {group}")
        if baseline is None:
            delta = predict_delta(model, h[source], ctrl, cfg["prediction_batch_size"], device)
        else:
            if baseline == "control_copy":
                offset = np.zeros(len(panel))
            elif baseline == "global_mean_delta":
                offset = global_delta
            elif baseline == "background_mean_delta":
                offset = statistics[group + "__delta"]
            else:
                raise ValueError("Unknown baseline")
            delta = (ctrl + offset).clip(0).astype(np.float32) - ctrl
        pred = ctrl + delta
        true_order = predicted_order = []
        if full_de:
            de, true_order = differential(truth, ctrl, np.ones(len(panel), dtype=bool))
            _, predicted_order = differential(pred, ctrl, np.ones(len(panel), dtype=bool))
            de.insert(0, "gene_id", panel.gene_id)
            if save_to is not None:
                de.to_csv(save_to / f"{group}_observed_DEG.tsv.gz", sep="\t", index=False)
        metrics, moments = evaluate(pred, truth, ctrl, true_order, predicted_order, top_n=(20, 50, 100, 200) if full_de else ())
        metrics.update(group=group, study=m.loc[m.response_group.eq(group), "study"].iloc[0],
                       control_cells=len(ctrl), target_cells=len(truth), delta_MSE=metrics["delta_RMSE"]**2)
        subsets = dict(functional=np.flatnonzero(panel.functional))
        if training_reference == "background":
            subsets.update(train_DE20=statistics[group + "__DE20"], train_DE100=statistics[group + "__DE100"])
        else:
            for n in (20, 100):
                subsets[f"train_DE{n}_union"] = np.unique(np.concatenate(
                    [statistics[g + f"__DE{n}"] for g in cfg["backgrounds"]]))
        for label, idx in subsets.items():
            metrics.update({label + "__" + key: value for key, value in subset_metrics(pred, truth, ctrl, idx).items()})
        rows.append(metrics)
        if save_to is not None:
            deltas[source] = delta
            pd.DataFrame(dict(gene_id=panel.gene_id, **moments)).to_csv(save_to / f"{group}_gene_predictions.tsv.gz", sep="\t", index=False)
    if save_to is not None:
        np.save(save_to / "effective_delta.npy", deltas)
        pm.to_csv(save_to / "control_metadata.tsv.gz", sep="\t", index_label="cell_id")
        pd.DataFrame(rows).to_csv(save_to / "background_metrics.csv", index=False)
    return rows, aggregate_metrics(rows)


def restore_head(checkpoint, cfg, genes, device):
    weights = checkpoint["model"]
    if "net.0.weight" in weights:
        model = ExpressionDeltaMLP(genes, cfg["hidden_size"])
    else:
        dimensions = weights["h_mean"].numel()
        model = DirectDeltaHead(torch.zeros(2, dimensions), genes, cfg["hidden_size"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def run_paths(cfg, encoder, seed):
    _, out, models = paths(cfg)
    suffix = Path(encoder) / f"seed{seed}"
    return models / suffix, out / suffix


def train_one(cfg, encoder, seed, device, resume, *, expression_input=False,
              head_factory=DirectDeltaHead, expression_factory=ExpressionDeltaMLP):
    data, out, _, manifest, fingerprint = prepared(cfg)
    model_dir, result_dir = run_paths(cfg, encoder, seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    # One process owns a run, including the completion marker and resume files.
    with (model_dir / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = model_dir / "complete.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            if record["fingerprint"] != fingerprint or record["best_sha256"] != file_hash(model_dir / "best.pt"):
                raise ValueError("Completed run changed")
            print(f"Already complete: {encoder}/seed{seed}", flush=True)
            return
        latest = model_dir / "latest.pt"
        if latest.exists() and not resume:
            raise ValueError(f"{latest} exists; use --resume to continue this run")
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        train = expression_dataset(data, manifest, "train") if expression_input else load_split(data, manifest, "train", encoder)
        validation = expression_dataset(data, manifest, "validation") if expression_input else load_split(data, manifest, "validation", encoder)
        panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
        with np.load(data / "train/statistics.npz") as values:
            statistics = dict(values)
        m, y, h = train
        controls = m[m.condition.eq("PBS")]
        control_y = y[m.condition.eq("PBS").to_numpy()]
        model = (expression_factory(len(panel), cfg["hidden_size"]) if expression_input else
                 head_factory(torch.from_numpy(np.array(h)), len(panel), cfg["hidden_size"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg["lr_factor"], patience=cfg["lr_patience"],
            threshold=cfg["minimum_relative_improvement"], threshold_mode="rel", min_lr=cfg["minimum_learning_rate"])
        start, best, reference, stale, history = 1, float("inf"), float("inf"), 0, []
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state["fingerprint"] != fingerprint:
                raise ValueError("Resume data/configuration differs")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            start, best, reference, stale, history = (state[k] for k in ("next_epoch", "best", "reference", "stale", "history"))
        baseline_rows = []
        for baseline in ("control_copy", "global_mean_delta", "background_mean_delta"):
            rows, macro = score_model(None, validation, cfg, panel, statistics, device, baseline=baseline)
            baseline_rows.extend(dict(method=baseline, **r) for r in rows)
            if baseline == "background_mean_delta":
                baseline_mse = macro["macro_delta_MSE"]
        pd.DataFrame(baseline_rows).to_csv(result_dir / "validation_baselines.csv", index=False)
        targets = {g: torch.tensor(statistics[g + "__delta"], dtype=torch.float32, device=device) for g in cfg["backgrounds"]}
        # Epoch-boundary resume repeats no completed epoch; early-stop state persists.
        for epoch in range(start, cfg["max_epochs"] + 1):
            if epoch > cfg["min_epochs"] and stale >= cfg["early_stop_patience"]:
                break
            batch_loss, grad_norm = fit_epoch(model, optimizer, controls, h, control_y, targets, cfg, rng, device)
            train_rows, train_macro = score_model(model, train, cfg, panel, statistics, device)
            val_rows, val_macro = score_model(model, validation, cfg, panel, statistics, device)
            score = val_macro["macro_delta_MSE"]
            scheduler.step(score)
            record = dict(epoch=epoch, weighted_batch_loss=batch_loss, gradient_norm=grad_norm,
                          learning_rate=optimizer.param_groups[0]["lr"],
                          validation_minus_background_baseline=score-baseline_mse,
                          **{"train_"+k: v for k, v in train_macro.items()},
                          **{"validation_"+k: v for k, v in val_macro.items()})
            for split, rows in (("train", train_rows), ("validation", val_rows)):
                for label in ("functional", "train_DE20", "train_DE100"):
                    errors = [r[label+"__delta_RMSE"]**2 for r in rows if r[label+"__delta_RMSE"] is not None]
                    record[f"{split}_{label}_macro_delta_MSE"] = float(np.mean(errors)) if errors else None
            history.append(record)
            state = dict(fingerprint=fingerprint, encoder=encoder, seed=seed, epoch=epoch, model=model.state_dict())
            if score < best:
                best = score
                atomic_torch(model_dir / "best.pt", dict(state, validation=val_macro))
                pd.DataFrame(val_rows).to_csv(result_dir / "best_validation_metrics.csv", index=False)
            if score < reference * (1-cfg["minimum_relative_improvement"]):
                reference, stale = score, 0
            else:
                stale += 1
            pd.DataFrame([dict(split="train", **r) for r in train_rows] + [dict(split="validation", **r) for r in val_rows]).to_csv(
                result_dir / "latest_background_metrics.csv", index=False)
            pd.DataFrame(history).to_csv(result_dir / "history.csv", index=False)
            atomic_torch(latest, dict(state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                next_epoch=epoch+1, best=best, reference=reference, stale=stale, history=history,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
            print(f"{encoder}/seed{seed} epoch={epoch} train={train_macro['macro_delta_MSE']:.6g} "
                  f"val={score:.6g} best={best:.6g} stale={stale}", flush=True)
        best_state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
        atomic_json(complete, dict(fingerprint=fingerprint, encoder=encoder, seed=seed,
            epochs=len(history), best_epoch=best_state["epoch"], best_validation=best_state["validation"],
            best_sha256=file_hash(model_dir / "best.pt"),
            stop_reason="early_stop" if stale >= cfg["early_stop_patience"] else "max_epochs"))


def checkpoint_lock(cfg, fingerprint):
    records = {}
    for encoder in cfg["models"]:
        for seed in cfg["seeds"]:
            folder, _ = run_paths(cfg, encoder, seed)
            if not (folder / "complete.json").exists():
                raise ValueError(f"Test remains closed: {encoder}/seed{seed} has not completed training")
            record = json.loads((folder / "complete.json").read_text())
            if record["fingerprint"] != fingerprint or record["best_sha256"] != file_hash(folder / "best.pt"):
                raise ValueError("A completed checkpoint changed")
            records[f"{encoder}/seed{seed}"] = record
    return dict(fingerprint=fingerprint, checkpoints=records,
                primary_metric="equal-background macro delta MSE; no best-seed selection")


def test(cfg, device):
    data, out, _, manifest, fingerprint = prepared(cfg)
    protocol = checkpoint_lock(cfg, fingerprint)  # Gate before loading any test target.
    protocol_path = out / "test_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Test was already opened for a different set of heads")
    atomic_json(protocol_path, protocol)
    panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
    with np.load(data / "train/statistics.npz") as values:
        statistics = dict(values)
    all_rows = []
    for encoder in cfg["models"]:
        dataset = load_split(data, manifest, "test", encoder)
        if encoder == next(iter(cfg["models"])):
            for baseline in ("control_copy", "global_mean_delta", "background_mean_delta"):
                rows, _ = score_model(None, dataset, cfg, panel, statistics, device, full_de=True,
                    save_to=out / "test" / baseline, baseline=baseline)
                all_rows.extend(dict(method=baseline, seed=-1, **r) for r in rows)
        for seed in cfg["seeds"]:
            model_dir, _ = run_paths(cfg, encoder, seed)
            state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
            model = restore_head(state, cfg, len(panel), device)
            rows, _ = score_model(model, dataset, cfg, panel, statistics, device, full_de=True,
                save_to=out / "test" / encoder / f"seed{seed}")
            all_rows.extend(dict(method=encoder, seed=seed, **r) for r in rows)
            print(f"Scored test: {encoder}/seed{seed}", flush=True)
            del model
    pd.DataFrame(all_rows).to_csv(out / "test_background_metrics.csv", index=False)
    report(cfg)


def report(cfg):
    _, out, _ = paths(cfg)
    metrics = pd.read_csv(out / "test_background_metrics.csv")
    summaries = []
    for (method, seed), frame in metrics.groupby(["method", "seed"], sort=False):
        for scope, part in [("all", frame), *list(frame.groupby("study", sort=False))]:
            macro = aggregate_metrics(part.where(pd.notna(part), None).to_dict("records"))
            summaries.append(dict(method=method, seed=int(seed), scope=scope, **macro))
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(out / "test_macro_metrics.csv", index=False)
    columns = [c for c in summaries if c not in ("method", "seed", "scope")]
    summaries.groupby(["method", "scope"])[columns].agg(["mean", "std"]).to_csv(out / "test_seed_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = summaries[summaries.scope.eq("all")]
    grouped = frame.groupby("method", sort=False).sqrt_macro_delta_MSE.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0), capsize=4)
    ax.set_ylabel("sqrt(equal-background mean delta MSE)")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out / "test_delta_RMSE.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for encoder in cfg["models"]:
        for seed in cfg["seeds"]:
            _, folder = run_paths(cfg, encoder, seed)
            history = pd.read_csv(folder / "history.csv")
            ax.plot(history.epoch, history.validation_macro_delta_MSE, label=f"{encoder}/{seed}")
    ax.set(xlabel="Epoch", ylabel="Validation macro delta MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "validation_learning_curves.png", dpi=180)
    plt.close(fig)
    print(f"Report: {out}", flush=True)


def check(cfg, device):
    data, out, _, manifest, fingerprint = prepared(cfg)
    panel = pd.read_csv(data / "gene_panel.tsv", sep="\t")
    with np.load(data / "train/statistics.npz") as values:
        statistics = dict(values)
    records = []
    for encoder in cfg["models"]:
        m, y, h = load_split(data, manifest, "train", encoder)
        pm = m[m.condition.eq("PBS")]
        x = y[m.condition.eq("PBS").to_numpy()]
        idx = np.flatnonzero(pm.response_group.eq(cfg["backgrounds"][0]))[:32]
        torch.manual_seed(0)
        model = DirectDeltaHead(torch.from_numpy(np.array(h)), len(panel), cfg["hidden_size"]).to(device)
        hh, xx = (torch.tensor(np.array(v), device=device) for v in (h[idx], x[idx]))
        if not torch.equal(model(hh), torch.zeros_like(xx)):
            raise ValueError("Head must initialize to control-copy")
        target = torch.tensor(statistics[cfg["backgrounds"][0]+"__delta"], device=device, dtype=torch.float32)
        before = float(delta_loss(xx, model(hh), target).detach())
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = delta_loss(xx, model(hh), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        after = float(delta_loss(xx, model(hh), target).detach())
        if not after < before:
            raise ValueError("Small-batch optimization did not reduce the loss")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "head.pt"
            atomic_torch(p, dict(model=model.state_dict()))
            restored = restore_head(torch.load(p, map_location="cpu", weights_only=False), cfg, len(panel), device)
            if not torch.equal(restored(hh), model(hh)):
                raise ValueError("Checkpoint reload changed predictions")
        records.append(dict(encoder=encoder, loss_before=before, loss_after_3_updates=after, checkpoint_roundtrip=True))
    atomic_json(out / "check_report.json", dict(fingerprint=fingerprint, device=str(device), checks=records,
        test_expression_loaded=False, formal_training_started=False))
    print(json.dumps(records, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "train", "test", "report"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--encoder", choices=["GeneCompass", "ILC2_tissue"])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--resume", action="store_true", help="Resume incomplete runs from the last finished epoch")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    if args.command == "prepare":
        prepare(cfg)
    elif args.command == "report":
        report(cfg)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable; select --device cpu explicitly")
        if args.command == "check":
            check(cfg, device)
        elif args.command == "test":
            test(cfg, device)
        else:
            for encoder in ([args.encoder] if args.encoder else cfg["models"]):
                for seed in ([args.seed] if args.seed is not None else cfg["seeds"]):
                    train_one(cfg, encoder, seed, device, args.resume)


if __name__ == "__main__":
    main()
