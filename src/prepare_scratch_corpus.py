"""Append reviewed BM ILC2 to the existing corpus without normalization or splitting.

Run: python src/prepare_scratch_corpus.py
"""
import gc
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from qc_utils import _gene_ids, _raw_counts

ROOT = Path(__file__).resolve().parents[1]
BASE = "data/processed/ILC2_posttrain_update/ILC2_posttrain_20261021_update.h5ad"
BM = {s: f"data/processed/{s}_qc/{s}_BM_ILC2.h5ad"
      for s in ("GSE117567", "GSE153748")}
OUT = ROOT / "scratch/data/pretrained_corpus"
REPORT = ROOT / "results/ILC2_scratch_pretrained_corpus"
VERSION = "ilc2_balanced_v1"


def verify_saved_counts(path, matrix):
    """Check the serialized sparse arrays in chunks without loading another corpus."""
    with h5py.File(path, "r") as f:
        for layer in ("X", "layers/counts"):
            if tuple(f[layer].attrs["shape"]) != matrix.shape:
                raise ValueError(f"{layer}: saved shape mismatch")
            for key in ("data", "indices", "indptr"):
                expected = getattr(matrix, key)
                saved = f[f"{layer}/{key}"]
                if saved.shape != expected.shape:
                    raise ValueError(f"{layer}/{key}: saved length mismatch")
                for start in range(0, len(expected), 2_000_000):
                    if not np.array_equal(saved[start:start + 2_000_000], expected[start:start + 2_000_000]):
                        raise ValueError(f"{layer}/{key}: saved counts mismatch")


def read_counts(path):
    a = ad.read_h5ad(ROOT / path)
    ## qc version
    if a.uns.get("metadata", {}).get("version") != VERSION:
        raise ValueError(f"{path}: QC version mismatch")

    ## unique cell and qc
    if not a.obs_names.is_unique or not a.obs.qc_pass.all() or not a.obs.selected_ILC2.all():
        raise ValueError(f"{path}: duplicate cells or unselected cells")

    genes = _gene_ids(a.var_names)
    ## gene names
    if not genes.is_unique or not genes.str.startswith("ENSMUSG").all():
        raise ValueError(f"{path}: invalid stable gene IDs")
    a.var_names = genes
    x = _raw_counts(a.X)

    ## counts
    if "counts" not in a.layers or (x != a.layers["counts"]).nnz:
        raise ValueError(f"{path}: X differs from counts")
    # Do not carry source embeddings or graphs into a newly merged corpus
    b = ad.AnnData(x, obs=a.obs.copy(), var=a.var.copy())
    for col in b.obs.select_dtypes("category").columns:
        b.obs[col] = b.obs[col].astype(object)
    b.obs["corpus_input_h5ad"] = path
    b.obs["qc_version"] = VERSION
    b.obs["corpus_source_cell_id"] = a.obs_names.astype(str)
    return b


def main():
    output = OUT / "ILC2_steady_state_with_BM_v1.h5ad"
    if output.exists():
        raise FileExistsError(output)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    base = read_counts(BASE)
    if base.obs.tissue.eq("BM").any():
        raise ValueError("Base already contains BM; review inclusion before appending")
    var = base.var.drop(columns=[c for c in base.var if c.startswith("genecompass_")
                                or c == "has_median_reference"]).copy()
    parts = [base]
    for study, path in BM.items():
        a = read_counts(path)
        if set(a.obs.study) != {study} or set(a.obs.tissue) != {"BM"}:
            raise ValueError(f"{study}: unexpected tissue/study")
        if set(a.obs.condition) != {"steady_state_no_exogenous_stimulation"}:
            raise ValueError(f"{study}: unexpected BM condition")
        # BM and peripheral libraries must share a feature space for study-level
        # presence flags to remain a valid missingness representation.
        if set(a.var_names) != set(var.index[var[f"present_{study}"]]):
            raise ValueError(f"{study}: BM feature space differs; library-level presence needed")
        a.obs["source_cell_id"] = a.obs_names.astype(str)
        a.obs_names = [s if s.startswith(study + ":") else study + ":" + s
                       for s in a.obs_names]
        a.obs["cell_uid"] = a.obs_names
        a.obs["source_h5ad"] = path
        a.obs["tissue_original"] = "BM"
        a.obs["library_id"] = study + ":" + a.obs["sample"].astype(str)
        a.obs["split_stratum"] = a.obs.library_id
        parts.append(a)
        print(f"Loaded {study} BM: {a.shape}", flush=True)

    sources = []
    for a in parts:
        sources.append({"path": a.obs.corpus_input_h5ad.iloc[0], "cells": a.n_obs,
                        "genes": a.n_vars, "umi": int(a.X.sum(dtype=np.int64))})
        for col in ("split", "role", "data_role"):
            if col in a.obs:
                a.obs[f"previous_corpus_{col}"] = a.obs[col].astype(str)
            a.obs[col] = "unassigned"
        if "genecompass_length" in a.obs:
            a.obs.rename(columns={"genecompass_length": "previous_genecompass_length"}, inplace=True)
    expected_totals = np.concatenate([np.asarray(a.X.sum(axis=1)).ravel() for a in parts])
    expected_nnz = np.concatenate([np.diff(a.X.indptr) for a in parts])
    merged = ad.concat(parts, join="outer", fill_value=0, merge=None)
    merged.var = var.reindex(merged.var_names)
    merged.var.index.name = "gene_id"
    merged.obs.index.name = "cell_id"
    if not merged.obs_names.is_unique or merged.n_obs != sum(x["cells"] for x in sources):
        raise ValueError("Duplicate or lost cells")
    if merged.obs[["study", "sample", "barcode"]].duplicated().any():
        raise ValueError("Duplicate source library/barcode")
    if not np.array_equal(expected_totals, np.asarray(merged.X.sum(axis=1)).ravel()):
        raise ValueError("Per-cell counts changed")
    if not np.array_equal(expected_nnz, np.diff(merged.X.indptr)):
        raise ValueError("Per-cell detected genes changed")
    # Verify exact values after gene alignment, not just total UMI preservation.
    start = 0
    for a in parts:
        cols = merged.var_names.get_indexer(a.var_names)
        if (merged.X[start:start + a.n_obs, cols] != a.X).nnz:
            raise ValueError("Gene-aligned counts changed during concat")
        start += a.n_obs
    for col in merged.obs.select_dtypes("object").columns:
        merged.obs[col] = merged.obs[col].fillna("unknown").astype(str)
    merged.layers["counts"] = merged.X.copy()
    common = merged.var.filter(like="present_").all(axis=1)
    merged.var["present_in_all_studies"] = common
    report = {"output": str(output.relative_to(ROOT)), "sources": sources,
              "cells": merged.n_obs, "genes_union": merged.n_vars,
              "genes_common": int(common.sum()), "libraries": merged.obs.library_id.nunique(),
              "tissue_cells": merged.obs.tissue.value_counts().to_dict(),
              "qc_version": VERSION, "split": "unassigned",
              "checks": {"integer_nonnegative_counts": True, "unique_source_cells": True,
                         "exact_gene_aligned_counts_preserved": True,
                         "per_cell_totals_and_detection_preserved": True,
                         "BM_feature_spaces_match_study_presence": True},
              "limitations": ["Identity is computational support, not independent experimental validation",
                              "Absent study features are structural zeros; consult var.present_<study>",
                              "Historical splits are provenance only; new split not assigned",
                              "No normalization, tokenization, batch correction or additional QC"]}
    merged.uns["metadata"] = {"version": VERSION, "dataset_version": "ILC2_steady_state_with_BM_v1",
                               "counts": "X and layers/counts: raw nonnegative integer UMI",
                               "split": "unassigned"}
    merged.uns["corpus_provenance_json"] = json.dumps(report, ensure_ascii=False)
    del parts, base, a
    gc.collect()
    print(f"Saving {merged.shape} to {output}", flush=True)
    partial = output.with_suffix(".partial.h5ad")
    merged.write_h5ad(partial, compression="gzip")
    verify_saved_counts(partial, merged.X)
    partial.rename(output)
    report["checks"]["saved_X_and_counts_exactly_reloaded"] = True
    merged.obs.to_csv(OUT / "cell_metadata.tsv.gz", sep="\t")
    merged.var.to_csv(OUT / "gene_metadata.tsv.gz", sep="\t")
    merged.obs.groupby(["study", "library_id", "tissue"], observed=True).size().rename("cells").to_csv(
        REPORT / "library_counts.tsv", sep="\t")
    (REPORT / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# ILC2 steady-state corpus with bone marrow\n\n"
        f"`{output.name}`: {merged.n_obs:,} cells × {merged.n_vars:,} union genes; "
        f"{int(common.sum()):,} common genes.\n\n"
        "X and layers/counts retain integer raw counts. No normalization, tokenization or new split. "
        "All split/role/data_role values are unassigned; previous_corpus_* are historical provenance only. "
        "Study-specific unmeasured genes have structural zeros: use var.present_<study>, "
        "not X==0, to determine whether a feature was measured.\n\n"
        "BM sources: GSE117567/GSM3303970 and GSE153748/GSM4653701, both ilc2_balanced_v1. "
        "Identity is computationally supported, not independently validated.\n\n"
        "Reproduce from repository root: `python src/prepare_scratch_corpus.py`. "
        "Existing final output is never overwritten. "
        "Report: `results/ILC2_scratch_pretrained_corpus/prepare_report.json`.\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
