"""Score-only Top-20 measured DE panels for individual GSE202186 mouse/type contrasts."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import stdtr


TOP_K = 20


def moments(counts):
    """Mean and sampling variance of the mean of log1p(CP10k) across cells."""
    n = counts.shape[0]
    sizes = np.asarray(counts.sum(axis=1)).ravel()
    if n < 2 or np.any(sizes <= 0):
        raise ValueError("A sample needs at least two nonempty measured cells")
    values = np.log1p(counts.data * np.repeat(1e4 / sizes, np.diff(counts.indptr)))
    total = np.bincount(counts.indices, weights=values, minlength=counts.shape[1])
    squares = np.bincount(counts.indices, weights=values ** 2, minlength=counts.shape[1])
    mean = total / n
    var_mean = np.maximum(squares - total * mean, 0) / (n * (n - 1))
    return mean, var_mean, n


def measured_panel(treated, controls):
    """Two-sided Welch ranking, weighting PBS mice equally when the batch contains two."""
    mean, variance, n = treated
    m = len(controls)
    effect = mean - np.mean([c[0] for c in controls], axis=0)
    # Satterthwaite degrees of freedom for one treated mean minus a mean of PBS-mouse means.
    var = variance.copy()
    denominator = variance ** 2 / (n - 1)
    for _, control_variance, n_control in controls:
        term = control_variance / m ** 2
        var += term
        denominator += term ** 2 / (n_control - 1)
    idx = np.flatnonzero((denominator > 0) & (var > 0))
    t = np.abs(effect[idx]) / np.sqrt(var[idx])
    p = np.minimum(1., 2 * stdtr(var[idx] ** 2 / denominator[idx], -t))
    order = np.lexsort((idx, -np.abs(effect[idx]), p))[:TOP_K]
    return idx[order], p[order]


def write_panels(ds, path):
    # The GEO reader also verifies the per-cell UMI totals against the author's annotations.
    src = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src))
    from gse202186 import read_annotated_rna

    root = src.parent
    source = root / "immune-dict/data"
    libraries = pd.read_csv(root / "results/GSE202186_overview/libraries.tsv", sep="\t").set_index("channel")
    contexts = pd.read_csv(ds.folder / "contexts.tsv", sep="\t").set_index("context_index")
    pbs = contexts.pbs_sample_ids.str.split(",")
    needed = set(ds.examples.sample_id) | {s for i in ds.examples.context_index.unique() for s in pbs[i]}
    cells = pd.read_csv(source / "SCP2554_author_annotations.tsv.gz", sep="\t").set_index("NAME")
    cells["sample_id"] = cells.cyt_display + "|" + cells.rep + "|" + cells.celltype
    a = read_annotated_rna(source / "GSE202186_RAW.tar", cells[cells.sample_id.isin(needed)], libraries)
    groups = a.obs.groupby("sample_id", observed=True).indices
    if list(a.var_names) != ds.genes or set(groups) != needed:
        raise ValueError("Measured-cell gene or sample axis disagrees with the training package")
    controls = {s: moments(a.X[groups[s]]) for s in needed - set(ds.examples.sample_id)}
    panels = []
    for row in ds.examples.itertuples():
        ids = groups[row.sample_id]
        if len(ids) != row.n_cells or sum(controls[s][2] for s in pbs[row.context_index]) != contexts.at[row.context_index, "n_pbs_cells"]:
            raise ValueError(f"Stimulated or PBS cell count mismatch for {row.sample_id}")
        idx, p = measured_panel(moments(a.X[ids]), [controls[s] for s in pbs[row.context_index]])
        panels.extend((row.target_index, row.sample_id, rank, int(g), ds.genes[g], prob)
                      for rank, (g, prob) in enumerate(zip(idx, p), start=1))
    pd.DataFrame(panels, columns=["target_index", "sample_id", "rank", "gene_index", "gene_id", "welch_p"]
                 ).to_csv(path, sep="\t", index=False)


def panel_indices(ds, out):
    path = out / "observed_top20.tsv.gz"
    if not path.exists():
        with TemporaryDirectory(dir=out) as tmp:
            stage = Path(tmp) / path.name
            write_panels(ds, stage)
            stage.replace(path)
    table = pd.read_csv(path, sep="\t")
    samples = ds.examples.set_index("target_index").sample_id
    genes = table.gene_index.to_numpy(dtype=int)
    if (not table.sample_id.equals(table.target_index.map(samples))
            or not np.array_equal(np.asarray(ds.genes)[genes], table.gene_id)
            or set(table.target_index) != set(samples.index)
            or table.duplicated(["target_index", "gene_index"]).any()
            or not table["rank"].eq(table.groupby("target_index").cumcount() + 1).all()
            or table["rank"].max() > TOP_K):
        raise ValueError("Measured Top-20 gene or mouse/type axis disagrees with the training package")
    return {key: group.gene_index.to_numpy() for key, group in table.groupby("target_index")}
