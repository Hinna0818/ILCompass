"""Evaluate cross-batch retrieval against the frozen CD8 mouse and ligand baselines."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

if __package__:
    from . import cd8_loss_scores as shared
    from .mouse_donor import SEEN
else:
    import cd8_loss_scores as shared
    from mouse_donor import SEEN


MSE_RUN = shared.ROOT / "results/GSE202186_CD8_pair_loss"


def archived_baselines(out):
    key = ["repeat", "regime", "model"]
    metrics = shared.scores.METRICS + ["PDS_cosine", "top1", "chance_top1"]
    names = ("PBS_zero", "type_train_mean", "same_cytokine_train_mean")

    def cd8_baselines(name):
        table = pd.read_csv(shared.REFERENCE / name, sep="\t")
        return table[table.scope.eq("T_cell_CD8") & table.model.isin(names)]

    expression = cd8_baselines("metrics_by_repeat.tsv")
    ranking = cd8_baselines("pds_by_repeat.tsv")
    comparisons = expression[key + ["n_mice", *shared.scores.METRICS]].merge(
        ranking[key + ["PDS_cosine", "top1", "chance_top1"]], on=key, validate="1:1")
    ridge = pd.read_csv(shared.REFERENCE / "CD8_ESM2_ridge.tsv", sep="\t")
    ridge = ridge.loc[ridge.model.eq("ridge"), key + ["n_mice", "PDS_cosine", "PCC_top20_DE"]]
    comparisons = pd.concat([comparisons, ridge.assign(model="ESM2_ridge")], ignore_index=True)

    rows = pd.read_csv(out / "metrics_by_repeat.tsv", sep="\t")
    counts = rows[rows.model.eq("mse")].set_index(key[:2]).n_mice.sort_index()
    for model in (*names, "ESM2_ridge"):
        observed = comparisons[comparisons.model.eq(model)].set_index(key[:2]).n_mice.sort_index()
        expected = counts[counts.index.get_level_values("regime") == SEEN] if model == names[-1] else counts
        if not expected.equals(observed):
            raise ValueError(f"{model}: archived CD8 baseline coverage differs from this run")

    grouped = comparisons.groupby(["regime", "model"])
    summary = grouped[metrics].mean()
    summary["n_repeats"] = grouped.repeat.nunique()
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        comparisons.sort_values(key).to_csv(stage / "baseline_by_repeat.tsv", sep="\t", index=False)
        summary.reset_index().to_csv(stage / "baseline_summary.tsv", sep="\t", index=False)
        for path in stage.iterdir():
            path.replace(out / path.name)


def evaluate(ds, truth, out):
    out = Path(out)
    shared.evaluate(ds, truth, out, mse_root=MSE_RUN, selected_arms=("retrieval", "mse_matched_epoch"),
                    epoch30_arms=("retrieval", "fixed001"))
    archived_baselines(out)
