"""Compare frozen CD8 T loss-ablation predictions on the original mouse-donor test rosters."""

from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__:
    from . import mouse_scores as scores
    from .gene_scores import pcc_delta
    from .mouse_donor import NEW, N_REPEATS
    from .train import ROOT
else:
    import mouse_scores as scores
    from gene_scores import pcc_delta
    from mouse_donor import NEW, N_REPEATS
    from train import ROOT


REFERENCE = ROOT / "results/GSE202186_PULSAR_mouse_donor"
EDGER = ROOT / "results/GSE202186_perturbation_model/edger"
KEY = ["repeat", "regime", "model"]
POOL_METRICS = ["PDS_cosine", "top1", "chance_top1", "pair_delta_RMSE", "pair_delta_PCC",
                "pair_amplitude_ratio", "zero_pair_RMSE"]
ROSTER = ["target_index", "mouse_id", "processing_batch", "celltype", "canonical_id", "outer_role", "cyt_display"]


def contrast_pools(test, truth, predictions, repeat):
    records = []
    for (regime, batch), pool in test.groupby(["outer_role", "processing_batch"]):
        groups = [g for _, g in pool.groupby("canonical_id")]
        n = len(groups)
        if n < 2:
            continue
        y = np.stack([truth[g.target_index].mean(axis=0, dtype="f8") for g in groups])
        y -= y.mean(axis=0)
        scale = 2 * n / (n - 1)
        zero_rmse = np.sqrt(scale * np.mean(y**2))
        for model, vectors in predictions.items():
            p = np.stack([vectors[g.index].mean(axis=0, dtype="f8") for g in groups])
            p -= p.mean(axis=0)
            # Mean squared difference over unordered pairs = 2n/(n-1) times the centered variance.
            rmse = np.sqrt(scale * np.mean((p - y)**2))
            corr = [pcc_delta(p[i] - p[j], y[i] - y[j]) for i, j in combinations(range(n), 2)]
            records.append(dict(repeat=repeat, regime=regime, processing_batch=batch, model=model,
                                n_pairs=len(corr), pair_delta_RMSE=rmse, zero_pair_RMSE=zero_rmse,
                                pair_delta_PCC=pd.Series(corr).mean(),
                                pair_amplitude_ratio=np.linalg.norm(p) / np.linalg.norm(y) if np.any(y) else np.nan))
    return records


def repeat_summary(mice, pools, arms=("pair",)):
    by_mouse = mice.groupby(KEY)
    by_pool = pools.groupby(KEY)
    values = by_mouse[scores.METRICS].mean().join(by_pool[POOL_METRICS].mean())
    differences = []
    for arm in arms:
        delta = values.xs(arm, level="model") - values.xs("mse", level="model")
        delta["model"] = f"{arm}_minus_mse"
        differences.append(delta.set_index("model", append=True))
    values = pd.concat([values, *differences])
    # Each CD8 mouse supplies one row. Candidates within a PDS pool can have several mice.
    counts = mice[mice.model == "mse"].groupby(KEY[:2]).agg(n_mice=("mouse_id", "nunique"),
                                                             n_factors=("canonical_id", "nunique"),
                                                             n_mice_replicated=("has_replicated_panel", "sum"))
    n_pools = pools[pools.model == "mse"].groupby(KEY[:2]).processing_batch.nunique().rename("n_pools")
    by_repeat = values.reset_index().merge(counts.join(n_pools), on=KEY[:2], validate="m:1")
    grouped = by_repeat.groupby(["regime", "model"])
    totals = grouped[scores.METRICS + POOL_METRICS].mean()
    totals["n_repeats"] = grouped.repeat.nunique()
    return by_repeat, totals.reset_index()


def evaluate(ds, truth, out, *, mse_root=None, selected_arms=("pair",), epoch30_arms=("pair",)):
    out = Path(out)
    panels = scores.panel_indices(ds, REFERENCE)
    replicated = scores.replicated_panels(ds, EDGER)
    tables = {}
    for prefix, suffix, arms in [("", "", selected_arms), ("epoch30_", "_epoch30", epoch30_arms)]:
        all_mice, all_pools, all_contrasts = [], [], []
        for rep in range(1, N_REPEATS + 1):
            dest = out / f"repeat_{rep}"
            test = pd.read_csv(dest / "predictions.tsv", sep="\t")
            reference = pd.read_csv(REFERENCE / dest.name / "predictions.tsv", sep="\t", usecols=ROSTER)
            expected = reference.loc[reference.celltype.eq("T_cell_CD8"), ROSTER].sort_values("target_index")
            if (not test[ROSTER].sort_values("target_index").reset_index(drop=True).equals(expected.reset_index(drop=True))
                    or not np.array_equal(test.prediction_index, np.arange(len(test)))):
                raise ValueError(f"{dest}: predictions disagree with the frozen CD8 mouse-donor roster")
            mse_dest = dest if mse_root is None else mse_root / dest.name
            if mse_dest != dest:
                order = ["prediction_index", *ROSTER]
                baseline = pd.read_csv(mse_dest / "predictions.tsv", sep="\t", usecols=order)
                if not test[order].equals(baseline[order]):
                    raise ValueError(f"{dest}: MSE baseline predictions have a different row order")
            test["major"] = True
            predictions = {"mse": np.load(mse_dest / f"mse{suffix}_delta.npy", mmap_mode="r")}
            predictions.update({m: np.load(dest / f"{m}{suffix}_delta.npy", mmap_mode="r") for m in arms})
            for model, pred in predictions.items():
                if pred.shape != (len(test), len(ds.genes)) or not np.isfinite(pred).all():
                    raise ValueError(f"{dest}: invalid {model}{suffix} delta matrix")
                for r in test.itertuples():
                    k = r.target_index
                    panel = replicated.get((r.celltype, r.cyt_display)) if r.outer_role == NEW else None
                    all_mice.append(dict(repeat=rep, regime=r.outer_role, model=model, mouse_id=r.mouse_id,
                                         target_index=k, processing_batch=r.processing_batch, cyt_display=r.cyt_display,
                                         canonical_id=r.canonical_id, has_replicated_panel=panel is not None,
                                         **scores.expression(pred[r.prediction_index], truth[k], panels[k], panel)))
            with threadpool_limits(limits=1, user_api="blas"):
                all_pools.extend(scores.pds_pools(test, truth, predictions, rep))
                all_contrasts.extend(contrast_pools(test, truth, predictions, rep))
        mice = pd.DataFrame(all_mice)
        pools = pd.DataFrame(all_pools).merge(pd.DataFrame(all_contrasts), on=KEY + ["processing_batch"],
                                               validate="1:1")
        repeats, summary = repeat_summary(mice, pools, arms)
        tables["epoch30_by_repeat.tsv" if prefix else "metrics_by_repeat.tsv"] = repeats
        tables[prefix + "summary.tsv"] = summary
        if not prefix:
            tables["mouse_metrics.tsv.gz"] = mice
            by_factor = mice.groupby(KEY + ["cyt_display"])
            tables["factor_metrics.tsv"] = by_factor[scores.METRICS].mean().join(
                by_factor.mouse_id.nunique().rename("n_mice")).reset_index()
            tables["pools.tsv"] = pools
        elif mse_root is not None:
            tables["epoch30_pools.tsv"] = pools
    with TemporaryDirectory(dir=out) as tmp:
        for name, table in tables.items():
            table.to_csv(Path(tmp) / name, sep="\t", index=False)
        for name in tables:
            (Path(tmp) / name).replace(out / name)
