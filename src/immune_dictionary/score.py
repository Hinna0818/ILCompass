"""Score complete out-of-fold mouse dictionary predictions against held-out responses."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
import torch

if __package__:
    from .gene_scores import DE_METRICS, PANEL_METRICS, alternative_references, deg_metrics, pcc_delta, top_effect_panels
    from .train import (CONDITION, DATASET, RESULTS, Dataset, complete_folds, fold_directory, fold_layout,
                        outer_fit, routed_rows, target_id, training_truth)
else:
    from gene_scores import DE_METRICS, PANEL_METRICS, alternative_references, deg_metrics, pcc_delta, top_effect_panels
    from train import (CONDITION, DATASET, RESULTS, Dataset, complete_folds, fold_directory, fold_layout,
                       outer_fit, routed_rows, target_id, training_truth)


METRICS = ["MAE_all", "Delta_R2", "PCC_delta_all", "PCC_delta_testable", *DE_METRICS]
FIXED_AXIS = {"A": "cyt_display", "B": "celltype"}
POOL = ["processing_batch", "celltype", "ligand_group"]
PDS = ["PDS_L1_arc2025", "PDS_L1_normalized", "top1_credit", "random_arc", "random_top1"]
EXTRA_PDS = ["PDS_cosine_normalized", "cosine_top1", "MRR_L1", "MRR_cosine", "random_MRR"]


def load_predictions(ds, runs):
    """Verify every head, gene axis and mouse-to-vector assignment before any truth is opened."""
    saved = {(model, task): {} for model in runs for task in ("A", "B")}
    versions = set()
    for model, root in runs.items():
        for fold in complete_folds(ds):
            directory = fold_directory(root, fold)
            fit = outer_fit(ds.examples, fold)
            expected_rows = routed_rows(ds.examples, fit, fold)
            rows = pd.read_csv(directory / "predictions.tsv", sep="\t", keep_default_na=False)
            if list(rows.columns) != list(expected_rows.columns) or not rows.astype(str).equals(expected_rows.astype(str)):
                raise ValueError(f"{directory}: sample IDs or prediction/target indices disagree with the fold")
            checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            if (checkpoint["input_id"] != ds.input_id or checkpoint["gene_id"] != ds.genes
                    or checkpoint["outer_fit_target_indices"] != fit.target_index.tolist()
                    or checkpoint["fold"] != fold_layout(fold)
                    or checkpoint["settings"]["shuffle_ligands"] != (model == "shuffled")):
                raise ValueError(f"{directory}: incompatible checkpoint, fitting rows or gene axis")
            versions.add(checkpoint["settings"]["target_id"])
            delta = np.load(directory / "delta.npy", mmap_mode="r")
            perturbed = np.load(directory / "perturbed.npy", mmap_mode="r")
            if (delta.shape != (len(rows), len(ds.genes)) or delta.shape != perturbed.shape
                    or not np.isfinite(delta).all() or not np.allclose(
                        perturbed, ds.baseline[expected_rows.context_index] + delta, rtol=1e-6, atol=1e-6)):
                raise ValueError(f"{directory}: inconsistent expression matrices")
            for row in rows.itertuples(index=False):
                entries = saved[model, row.task]
                if row.target_index in entries:
                    raise ValueError(f"Duplicate {model} {row.task} prediction for {row.sample_id}")
                entries[row.target_index] = delta[row.prediction_index]
        for task in ("A", "B"):
            flag = f"eligible_{task}"
            if flag in ds.examples and set(ds.examples.loc[ds.examples[flag], "target_index"]) != saved[model, task].keys():
                raise ValueError(f"{model} {task}: global coverage disagrees with the frozen roster")
    if len(versions) != 1:
        raise ValueError("OOF checkpoints were fitted using different target versions")
    return saved, versions.pop()


def add_baselines(ds, values, truth):
    zero = np.zeros(len(ds.genes), dtype=np.float32)
    for task in ("A", "B"):
        values["zero", task] = dict.fromkeys(values["film", task], zero)
        values["train_mean", task] = {}
    for fold in ds.folds.itertuples(index=False):
        fit = outer_fit(ds.examples, fold)
        queries = routed_rows(ds.examples, fit, fold)
        group = fit.groupby(["celltype", "canonical_id"], sort=False)
        means = {key: truth[mice.target_index].mean(axis=0, dtype=np.float64)
                 for key, mice in group}
        for task, dim in (("A", "canonical_id"), ("B", "celltype")):
            for name, rows in queries[queries.task == task].groupby(dim):
                matches = [x for (cell, ligand), x in means.items()
                           if (ligand if task == "A" else cell) == name]
                if matches:
                    avg = np.mean(matches, axis=0).astype(np.float32)
                    values["train_mean", task].update(dict.fromkeys(rows.target_index, avg))


def load_deg(folder, ds):
    if folder is None:
        return {}
    genes = pd.read_csv(folder / "genes.tsv", sep="\t", usecols=["gene_id"]).gene_id.tolist()
    conditions = pd.read_csv(folder / "conditions.tsv", sep="\t")
    codes = np.load(folder / "gene_codes.npy", mmap_mode="r")
    if (genes != ds.genes or codes.dtype != np.uint8 or codes.shape != (len(conditions), len(genes))
            or conditions.duplicated(CONDITION).any() or not np.array_equal(
                np.sort(conditions.condition_index), np.arange(len(conditions)))):
        raise ValueError("edgeR condition or gene axis mismatch")
    expected = set(ds.examples.loc[ds.examples.complete_three, CONDITION].itertuples(index=False, name=None))
    if set(conditions[CONDITION].itertuples(index=False, name=None)) != expected:
        raise ValueError("edgeR and dataset use different sets of three-mouse conditions")
    out = {}
    for condition in conditions.itertuples(index=False):
        row = codes[condition.condition_index]
        if ((row > 4).any() or np.count_nonzero(row) != condition.n_tested
                or np.count_nonzero(row >= 2) != condition.n_deg):
            raise ValueError(f"Inconsistent edgeR tested/significant counts for {condition}")
        out[condition.celltype, condition.cyt_display] = row
    return out


def expression_scores(ds, values, truth, deg, alternatives):
    mice_out, conditions, references, panels = [], [], [], []
    for (model, task), vectors in values.items():
        rows = ds.examples[ds.examples.target_index.isin(vectors)]
        for key, mice in rows.groupby(CONDITION, sort=False):
            indices = mice.target_index.to_numpy()
            pred = np.stack([vectors[i] for i in indices])
            observed = truth[indices]
            if not np.isfinite(observed).all():
                raise ValueError(f"Nonfinite observed expression in {key}")
            errors = np.abs(pred - observed).mean(axis=1, dtype=np.float64)
            mouse_pcc = np.array([pcc_delta(p, y) for p, y in zip(pred, observed)])
            for r, error, pcc in zip(mice.itertuples(index=False), errors, mouse_pcc):
                mice_out.append((task, model, r.target_index, r.sample_id, r.rep, r.processing_batch,
                                 r.celltype, r.cyt_display, r.n_cells, r.complete_three, error, pcc))
            primary = bool(mice.complete_three.all() and len(mice) == 3)
            p, y = pred.mean(axis=0, dtype=np.float64), observed.mean(axis=0, dtype=np.float64)
            residual = p - y
            centered = y - y.mean()
            ss = np.sum(centered * centered)
            meta = dict(task=task, model=model, celltype=key[0], cyt_display=key[1])
            codes = deg.get(key) if primary else None
            base = deg_metrics(p, y, codes)
            if codes is not None:
                references.append(dict(**meta, reference="edgeR", **base))
                for name, other in alternatives.get(key, {}).items():
                    references.append(dict(**meta, reference=name, **deg_metrics(p, y, other)))
                panels.extend(dict(**meta, **panel) for panel in top_effect_panels(p, y, codes))
            conditions.append(dict(**meta,
                                   n_mice=len(mice), n_cells=mice.n_cells.sum(), primary_three_mice=primary,
                                   MAE_all=np.abs(residual).mean(), mean_mouse_MAE_all=errors.mean(),
                                   PCC_delta_all=pcc_delta(p, y),
                                   mean_mouse_PCC_delta_all=np.nanmean(mouse_pcc) if np.isfinite(mouse_pcc).any() else np.nan,
                                   PCC_delta_testable=pcc_delta(p[codes > 0], y[codes > 0]) if codes is not None else np.nan,
                                   Delta_R2=1 - np.sum(residual * residual) / ss if ss else np.nan,
                                   **base))
    mouse = pd.DataFrame(mice_out, columns=["task", "model", "target_index", "sample_id", "rep",
                        "processing_batch", *CONDITION, "n_cells", "complete_three", "MAE_all", "PCC_delta_all"])
    return mouse, pd.DataFrame(conditions), pd.DataFrame(references), pd.DataFrame(panels)


def summarize_panels(table, panel_key, metrics):
    out = []
    for (task, model, panel), rows in table.groupby(["task", "model", panel_key]):
        fixed = FIXED_AXIS[task]
        macro = rows.groupby(fixed)[metrics].mean().mean()
        out.append({"task": task, "model": model, panel_key: panel, "n_conditions": len(rows),
                    **macro.to_dict(), **{f"n_{c}": int(rows[c].count()) for c in metrics}})
    return pd.DataFrame(out)


def paired_reference_summary(table):
    keys = ["task", "model", *CONDITION]
    left = table.loc[table.reference == "edgeR_common", keys + DE_METRICS]
    right = table.loc[table.reference == "Cui_common", keys + DE_METRICS]
    pairs = left.merge(right, on=keys, suffixes=("_edgeR", "_Cui"), validate="1:1")
    result = []
    for (task, model), rows in pairs.groupby(["task", "model"]):
        fixed = FIXED_AXIS[task]
        for metric in DE_METRICS:
            columns = [metric + suffix for suffix in ("_edgeR", "_Cui")]
            common = rows.dropna(subset=columns)
            values = common.groupby(fixed)[columns].mean().mean()
            result.append(dict(task=task, model=model, metric=metric, n_paired=len(common),
                               edgeR=values.iloc[0], Cui=values.iloc[1]))
    return pd.DataFrame(result)


def summarize_conditions(ds, mouse, conditions):
    primary = conditions[conditions.primary_three_mice]
    n_conditions = len(ds.examples[CONDITION].drop_duplicates())
    n_three = len(ds.examples.loc[ds.examples.complete_three, CONDITION].drop_duplicates())
    macros, differences, strata = [], [], []
    for (task, model), rows in conditions.groupby(["task", "model"]):
        outer = FIXED_AXIS[task]
        scored = rows[rows.primary_three_mice]
        by_fixed = scored.groupby(outer)[[*METRICS, "mean_mouse_MAE_all", "mean_mouse_PCC_delta_all"]].mean()
        macro = by_fixed.mean()
        by_fixed["n_three_mouse_conditions"] = scored.groupby(outer).size()
        strata.append(by_fixed.rename_axis("fixed_label").reset_index().assign(task=task, model=model, fixed_axis=outer))
        macros.append(dict(task=task, model=model, n_dataset_conditions=n_conditions,
                           n_three_mouse_conditions=n_three, n_predicted_conditions=len(rows),
                           n_predicted_mouse_types=int(((mouse.task == task) & (mouse.model == model)).sum()),
                           n_primary_conditions=len(scored), **macro.to_dict(),
                           **{f"n_{key}": int(scored[key].count()) for key in METRICS}))
        for baseline in ("zero", "train_mean"):
            if model == baseline:
                continue
            ref = primary[(primary.task == task) & (primary.model == baseline)]
            pairs = scored.set_index(CONDITION)[METRICS].subtract(ref.set_index(CONDITION)[METRICS])
            for metric in METRICS:
                available = pairs[metric].dropna()
                differences.append((task, model, baseline, metric, len(available),
                                    available.groupby(level=outer).mean().mean()))
    diff = pd.DataFrame(differences, columns=["task", "model", "baseline", "metric", "n_paired_conditions",
                                             "model_minus_baseline"])
    return pd.DataFrame(macros), diff, pd.concat(strata, ignore_index=True)


def pds_ranks(distances):
    n = len(distances)
    ranks = rankdata(distances, method="average", axis=1)
    diag = ranks.diagonal()
    ties = (distances == distances.min(axis=1, keepdims=True))
    if n == 1:
        return ranks, np.full((1, len(PDS)), np.nan)
    metrics = np.column_stack((1 - (diag - 1) / n, 1 - (diag - 1) / (n - 1),
                              ties.diagonal() / ties.sum(axis=1),
                              np.full(n, (n + 1) / (2 * n)), np.full(n, 1 / n)))
    return ranks, metrics


def reciprocal_ranks(distances):
    """Expected reciprocal rank under uniform ordering of exact ties."""
    n = len(distances)
    if n < 2:
        return np.array([np.nan]), np.nan
    true = distances.diagonal()[:, None]
    better = (distances < true).sum(axis=1)
    tied = (distances == true).sum(axis=1)
    harmonic = np.r_[0, np.cumsum(1 / np.arange(1, n + 1))]
    return (harmonic[better + tied] - harmonic[better]) / tied, harmonic[n] / n


def cosine_distances(pred, actual):
    denom = np.linalg.norm(pred, axis=1)[:, None] * np.linalg.norm(actual, axis=1)[None, :]
    similarity = np.divide(pred @ actual.T, denom, out=np.zeros_like(denom), where=denom != 0)
    return 1 - np.clip(similarity, -1, 1)


def discrimination(ds, values, truth, cond):
    complete = cond[cond.primary_three_mice].groupby(CONDITION).size()
    common = set(complete.index[complete.eq(len(values))])
    rows = ds.examples[pd.MultiIndex.from_frame(ds.examples[CONDITION]).isin(common)]
    distances_out, query_out = [], []
    for (batch, cell, ligand_group), pool in rows.groupby(POOL):
        if pool.context_index.nunique() != 1:
            raise ValueError("A discrimination pool does not share a single PBS context")
        # Chemical aliases identify one candidate. Distinct reagents with identical ESM2 stay distinct.
        grouped = list(pool.groupby("canonical_id"))
        names = [name for name, mice in grouped]
        source = [mice.target_index.to_numpy() for _, mice in grouped]
        actual = np.stack([truth[i].mean(axis=0, dtype=np.float64) for i in source])
        n = len(names)
        for (model, task), vector in values.items():
            pred = np.stack([np.mean([vector[j] for j in i], axis=0, dtype=np.float64) for i in source])
            distances = cdist(pred, actual, metric="cityblock")
            ranks, metrics = pds_ranks(distances)
            cosine = cosine_distances(pred, actual)
            cos_ranks, cos_scores = pds_ranks(cosine)
            l1_mrr, chance_mrr = reciprocal_ranks(distances)
            cos_mrr, _ = reciprocal_ranks(cosine)
            if model == "zero" and n > 1:
                np.testing.assert_allclose(metrics[:, :3].mean(axis=0), [(n + 1) / (2 * n), 0.5, 1 / n],
                                           rtol=1e-10, atol=1e-10, err_msg="Zero baseline rank sanity check")
            for i, (name, mice) in enumerate(grouped):
                base = (task, model, batch, cell, ligand_group)
                query_out.append((*base, name, "|".join(sorted(set(mice.cyt_display))), len(mice), n,
                                  ranks[i, i] if n > 1 else np.nan, *metrics[i], cos_ranks[i, i] if n > 1 else np.nan,
                                  cos_scores[i, 1], cos_scores[i, 2], l1_mrr[i], cos_mrr[i], chance_mrr))
                for j, candidate in enumerate(names):
                    distances_out.append((*base, name, candidate, n, distances[i, j], ranks[i, j], i == j,
                                          cosine[i, j], cos_ranks[i, j]))
    index = ["task", "model", *POOL]
    distances = pd.DataFrame(distances_out, columns=[*index, "query_id", "candidate_id", "n_candidates",
                                                     "L1_distance", "rank", "correct", "cosine_distance", "cosine_rank"])
    queries = pd.DataFrame(query_out, columns=[*index, "query_id", "cyt_displays", "n_query_mice",
                                              "n_candidates", "correct_rank", *PDS, "cosine_correct_rank", *EXTRA_PDS])
    measures = PDS + EXTRA_PDS
    pools = queries.groupby(index, as_index=False)[["n_candidates", *measures]].mean()
    valid = pools[pools.n_candidates > 1]
    by_batch = valid.groupby(["task", "model", "celltype", "processing_batch"])[measures].mean()
    by_cell = by_batch.groupby(["task", "model", "celltype"]).mean()
    summary = by_cell.groupby(["task", "model"]).mean()
    counts = pools.groupby(["task", "model"]).agg(n_pools=("n_candidates", "size"),
                      n_rankable_pools=("n_candidates", lambda n: (n > 1).sum()),
                      n_queries=("n_candidates", "sum"))
    counts["n_rankable_queries"] = valid.groupby(["task", "model"]).n_candidates.sum()
    summary = counts.join(summary).reset_index()
    return distances, queries, pools, summary, len(common)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--predictions", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=RESULTS.parent / "scores")
    parser.add_argument("--shuffled", "--shuffle-out", type=Path, help="Complete mismatch-control OOF run")
    parser.add_argument("--deg", type=Path, help="Score-only edgeR tables and gene codes")
    parser.add_argument("--cui", type=Path, help="Tidy Cui Supplementary Table 3; requires --deg with significant_genes.tsv.gz")
    args = parser.parse_args(argv)
    ds = Dataset(args.dataset)
    runs = {"film": args.predictions}
    if args.shuffled is not None:
        runs["shuffled"] = args.shuffled
    values, training_id = load_predictions(ds, runs)
    # The first treated-expression read is deliberately after both rosters have been validated.
    truth = training_truth(ds)
    if training_id != target_id(ds) or ((args.dataset / "dataset.json").is_file() and
            hashlib.sha256(np.ascontiguousarray(truth)).hexdigest() != training_id):
        raise ValueError("Observed target values do not match the frozen training dataset")
    deg = load_deg(args.deg, ds)
    alternatives, reference_agreement = {}, None
    if args.cui is not None:
        if args.deg is None:
            parser.error("--cui requires --deg")
        alternatives, reference_agreement = alternative_references(ds, deg, args.cui, args.deg / "significant_genes.tsv.gz")
    add_baselines(ds, values, truth)
    mouse, conditions, references, panels = expression_scores(ds, values, truth, deg, alternatives)
    summary, differences, strata = summarize_conditions(ds, mouse, conditions)
    distances, queries, pools, pds, common = discrimination(ds, values, truth, conditions)
    tables = {"mouse_metrics.tsv": mouse, "condition_metrics.tsv": conditions, "summary.tsv": summary,
              "baseline_differences.tsv": differences, "pds_distances.tsv.gz": distances,
              "pds_queries.tsv": queries, "pds_pools.tsv": pools, "pds_summary.tsv": pds,
              "fixed_axis_summary.tsv": strata,
              "condition_equal_weight.tsv": conditions[conditions.primary_three_mice].groupby(
                  ["task", "model"], as_index=False)[METRICS].mean()}
    if not references.empty:
        tables["deg_reference_conditions.tsv.gz"] = references
        tables["deg_reference_summary.tsv"] = summarize_panels(references, "reference", DE_METRICS)
    if not panels.empty:
        tables["top_gene_conditions.tsv.gz"] = panels
        tables["top_gene_summary.tsv"] = summarize_panels(panels, "top_n", PANEL_METRICS)
    if reference_agreement is not None:
        tables["deg_reference_agreement.tsv"] = reference_agreement
        tables["deg_reference_paired.tsv"] = paired_reference_summary(references)
    report = dict(n_heads_per_fitted_model=len(ds.folds), n_genes=len(ds.genes), input_id=ds.input_id,
                  models=list(dict.fromkeys(model for model, task in values)),
                  n_common_three_mouse_conditions_for_PDS=common,
                  edgeR_truth=str(args.deg) if args.deg else None,
                  Cui_signatures=str(args.cui) if args.cui else None)
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp)
        for filename, frame in tables.items():
            frame.to_csv(stage / filename, sep="\t", index=False, na_rep="NA")
        (stage / "scoring.json").write_text(json.dumps(report, indent=2) + "\n")
        for filename in [*tables, "scoring.json"]:
            os.replace(stage / filename, args.out / filename)
    print(summary[["task", "model", "n_primary_conditions", "MAE_all", "DEG_overlap"]].to_string(index=False))


if __name__ == "__main__":
    main()
