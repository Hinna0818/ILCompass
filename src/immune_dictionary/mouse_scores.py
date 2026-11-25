"""Expression and stimulus-discrimination scores on frozen mouse-donor predictions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__:
    from .gene_scores import pcc_delta, ranked_credit
    from .mouse_donor import MAJOR_FRACTION, NEW, SEEN
    from .mouse_panels import TOP_K, panel_indices
    from .score import cosine_distances, load_deg, pds_ranks
else:
    from gene_scores import pcc_delta, ranked_credit
    from mouse_donor import MAJOR_FRACTION, NEW, SEEN
    from mouse_panels import TOP_K, panel_indices
    from score import cosine_distances, load_deg, pds_ranks


METRICS = ["PCC_top20_DE", "MAE_top20_DE", "overlap_top20_DE", "direction_top20_DE",
           "PCC_delta_all", "MAE_all", "Delta_R2_all", "PCC_top20_replicated_DE"]


def replicated_panels(ds, folder):
    if folder is None:
        return {}
    codes = load_deg(folder, ds)
    tab = pd.read_csv(folder / "significant_genes.tsv.gz", sep="\t")
    if (not np.array_equal(np.asarray(ds.genes)[tab.gene_index], tab.gene_id)
            or not tab.FDR.le(.05).all()):
        raise ValueError("Replicated edgeR results disagree with the gene axis or FDR threshold")
    panels = {}
    for key, genes in tab.groupby(["celltype", "cyt_display"]):
        indices = genes.gene_index.to_numpy()
        if not np.array_equal(np.sort(indices), np.flatnonzero(codes[key] >= 2)):
            raise ValueError(f"Incomplete edgeR results for {key}")
        if len(genes) >= TOP_K:
            panels[key] = genes.sort_values(["PValue", "gene_index"]).gene_index.to_numpy()[:TOP_K]
    return panels


def expression(p, y, panel, replicated):
    err = p - y
    denom = np.var(y, dtype=np.float64)
    scores = dict.fromkeys(METRICS, np.nan)
    scores.update(MAE_all=np.mean(np.abs(err)), PCC_delta_all=pcc_delta(p, y),
                  Delta_R2_all=1 - np.mean(err ** 2, dtype=np.float64) / denom if denom else np.nan)
    if len(panel) == TOP_K:
        a, b = p[panel], y[panel]
        scores.update(PCC_top20_DE=pcc_delta(a, b), MAE_top20_DE=np.abs(a - b).mean(),
                      direction_top20_DE=np.mean(np.sign(a) == np.sign(b)),
                      overlap_top20_DE=ranked_credit(np.abs(p), TOP_K)[panel].mean())
    if replicated is not None:
        scores["PCC_top20_replicated_DE"] = pcc_delta(p[replicated], y[replicated])
    return scores


def predictions_and_baselines(ds, truth, fit, test, dest):
    saved = pd.read_csv(dest / "predictions.tsv", sep="\t")
    pred = np.load(dest / "delta.npy", mmap_mode="r")
    if (saved.target_index.tolist() != test.target_index.tolist() or saved.mouse_id.tolist() != test.mouse_id.tolist()
            or not np.array_equal(saved.prediction_index, np.arange(len(test)))
            or pred.shape != (len(test), len(ds.genes)) or not np.isfinite(pred).all()):
        raise ValueError(f"Prediction rows do not match the test mice in {dest}")
    by_condition = {}
    for key, mice in fit.groupby(["celltype", "canonical_id"]):
        by_condition[key] = truth[mice.target_index].mean(axis=0, dtype=np.float64).astype(np.float32)
    # Give observed reagents equal weight in the type-specific pooled baseline.
    by_type = {c: np.mean([v for (ct, _), v in by_condition.items() if ct == c], axis=0)
               for c in fit.celltype.unique()}
    zero = np.zeros(len(ds.genes), dtype=np.float32)
    return {"FiLM": pred, "PBS_zero": [zero] * len(test),
            "type_train_mean": [by_type.get(r.celltype) for r in test.itertuples()],
            "same_cytokine_train_mean": [by_condition.get((r.celltype, r.canonical_id)) if r.outer_role == SEEN
                                         else None for r in test.itertuples()]}


def pds_pools(test, truth, preds, repeat):
    results = []
    for (regime, batch, ct), pool in test.groupby(["outer_role", "processing_batch", "celltype"]):
        groups = [r.index.to_numpy() for _, r in pool.groupby("canonical_id")]
        if len(groups) < 2:
            continue
        observed = np.stack([truth[test.iloc[ix].target_index].mean(axis=0, dtype=np.float64) for ix in groups])
        for model, vectors in preds.items():
            # Every candidate in the same arena needs a prediction. Missing baselines retain their own coverage.
            if any(vectors[i] is None for i in pool.index):
                continue
            predicted = np.stack([np.mean([vectors[i] for i in ix], axis=0, dtype=np.float64) for ix in groups])
            _, scores = pds_ranks(cosine_distances(predicted, observed))
            results.append(dict(repeat=repeat, regime=regime, model=model, processing_batch=batch, celltype=ct,
                                major=bool(pool.major.iloc[0]), n_candidates=len(groups),
                                PDS_cosine=scores[:, 1].mean(), top1=scores[:, 2].mean(), chance_top1=1/len(groups)))
    return results


def by_scope(rows, metrics, *, expression_rows):
    # Average eligible types within a mouse first. Across repeats, each random split receives one vote.
    out = []
    scopes = [("major_celltypes", rows[rows.major]), *rows.groupby("celltype")]
    for scope, selection in scopes:
        for (repeat, regime, model), r in selection.groupby(["repeat", "regime", "model"]):
            if expression_rows:
                mice = r.groupby("mouse_id")[metrics].mean()
                counts = dict(n_mice=len(mice), n_mouse_types=len(r),
                              n_mice_top20=mice.PCC_top20_DE.count(), n_types_top20=r.PCC_top20_DE.count(),
                              n_mice_replicated=mice.PCC_top20_replicated_DE.count(),
                              n_types_replicated=r.PCC_top20_replicated_DE.count())
                means = mice.mean()
            else:
                counts = dict(n_pools=len(r), n_queries=r.n_candidates.sum())
                means = r[metrics].mean()
            out.append(dict(repeat=repeat, regime=regime, model=model, scope=scope, **counts, **means.to_dict()))
    return pd.DataFrame(out)


def evaluate(ds, truth, runs, out, edger):
    panels = panel_indices(ds, out)
    replicated = replicated_panels(ds, edger)
    per_mouse, pools, counts = [], [], []
    for dest, roles, design in runs:
        rep = design["repeat"]
        for role in ["fit", SEEN, NEW, "seen_mouse_new_cytokine", "excluded"]:
            rows = roles[roles.outer_role.eq(role)]
            counts.append(dict(repeat=rep, role=role, estimable=(role != "seen_mouse_new_cytokine"),
                               n_mice=rows.mouse_id.nunique(), n_ligand_groups=rows.split_ligand.nunique(),
                               n_batches=rows.processing_batch.nunique(), n_mouse_types=len(rows)))
        test = roles[roles.outer_role.isin([SEEN, NEW])].reset_index(drop=True).copy()
        test["major"] = test.celltype.map(design["outer"]["pbs_cell_fraction"]).ge(MAJOR_FRACTION)
        fit = roles[roles.outer_role.eq("fit")]
        preds = predictions_and_baselines(ds, truth, fit, test, dest)
        for i, r in enumerate(test.itertuples()):
            k = r.target_index
            # For known reagents a three-mouse reference could include fitting mice. Only novel reagents qualify.
            ref = replicated.get((r.celltype, r.cyt_display)) if r.outer_role == NEW else None
            for model, values in preds.items():
                p = values[i]
                if p is None:
                    continue
                per_mouse.append(dict(repeat=rep, regime=r.outer_role, model=model, mouse_id=r.mouse_id,
                                      celltype=r.celltype, cyt_display=r.cyt_display, target_index=k,
                                      processing_batch=r.processing_batch, major=r.major, n_panel=len(panels[k]),
                                      **expression(p, truth[k], panels[k], ref)))
        with threadpool_limits(limits=1, user_api="blas"):
            pools.extend(pds_pools(test, truth, preds, rep))
    per_mouse, pools = pd.DataFrame(per_mouse), pd.DataFrame(pools)
    scores = by_scope(per_mouse, METRICS, expression_rows=True)
    discrimination = by_scope(pools, ["PDS_cosine", "top1", "chance_top1"], expression_rows=False)
    group = ["scope", "regime", "model"]
    agg = scores.groupby(group)
    summary = agg[METRICS].mean().join(agg[METRICS].std().add_suffix("_SD"))
    summary["n_repeats"] = agg.size()
    summary["n_repeats_with_top20_PCC"] = agg.PCC_top20_DE.count()
    summary["n_repeats_with_replicated_PCC"] = agg.PCC_top20_replicated_DE.count()
    summary = summary.join(discrimination.groupby(group)[["PDS_cosine", "top1", "chance_top1"]].mean())
    tables = {"split_counts.tsv": pd.DataFrame(counts), "mouse_metrics.tsv.gz": per_mouse,
              "pds_pools.tsv.gz": pools, "pds_by_repeat.tsv": discrimination,
              "metrics_by_repeat.tsv": scores, "summary.tsv": summary.reset_index()}
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        for name, table in tables.items():
            table.to_csv(stage / name, sep="\t", index=False)
        for name in tables:
            (stage / name).replace(out / name)
    print(f"Scored {len(runs)} repeats. Summary: {out / 'summary.tsv'}", flush=True)
