"""构建 GSE117567 首版高可信外周 ILC2 数据。

先运行本脚本产生按文库的双细胞/聚类/marker 证据；审查群体后，用
--finalize --review <JSON> 导出 data/train_data/GSE117567.h5ad。
数值阈值为本项目明确的保守筛选规则，不是文献验证的统一分类器。
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/ilc2_model_numba")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ilc2_model_mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import argparse
import gc
import hashlib
import json
from pathlib import Path
import importlib.metadata
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/GSE117567_ilc2_selection"
import sys
sys.path.insert(0, str(ROOT / "src"))
from ilc2_qc import PANELS, add_evidence, marker_values


def prepare() -> None:
    """按样本处理原始计数的基础 QC 输出，先保存证据而不自动发布训练集。"""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "data/metadata/sample_manifest.tsv", sep="\t", dtype=str)
    cluster_rows = []
    all_decisions = []
    missing_all = {}
    for row in manifest.to_dict("records"):
        print(f"Preparing {row['sample']} / {row['tissue']}", flush=True)
        a = sc.read_h5ad(ROOT / "data/processed/GSE117567_basic_qc" / row["sample"] / "full_counts_qc.h5ad")
        assert a.obs["sample"].eq(row["sample"]).all()
        # GSM 文库为当前可追溯处理单位；不能从细胞数反推动物数或真实上样 doublet rate。
        sc.pp.scrublet(a, expected_doublet_rate=0.05, n_prin_comps=30,
                       use_approx_neighbors=True, random_state=0, verbose=False)
        if not np.isfinite(a.obs.doublet_score).all() or a.obs.predicted_doublet.isna().any():
            raise ValueError("Scrublet 未返回可用结果，需要复核而不能默认全为 singlet。")
        a.obs["scrublet_call_status"] = "auto_resolved" if "threshold" in a.uns["scrublet"] else "auto_unresolved_requires_review"
        missing_all[row["sample"]] = add_evidence(a)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=2500, flavor="seurat")
        sc.pp.pca(a, n_comps=30, mask_var="highly_variable", svd_solver="arpack", random_state=0)
        sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=0)
        sc.tl.leiden(a, resolution=0.6, flavor="igraph", n_iterations=2, directed=False, random_state=0)
        sc.tl.umap(a, random_state=0)
        a.obs["selection_cluster"] = row["tissue"] + ":" + a.obs.leiden.astype(str)
        sc.tl.rank_genes_groups(a, "leiden", method="t-test_overestim_var", n_genes=25, use_raw=False)
        markers = sc.get.rank_genes_groups_df(a, group=None)
        markers["gene_symbol"] = markers.names.map(a.var.gene_symbol)
        markers.to_csv(OUT / f"{row['sample']}_top_markers.tsv", sep="\t", index=False)
        genes = list(dict.fromkeys(g for panel in PANELS.values() for g in panel))
        gene_counts = marker_values(a, genes, "counts")
        for cluster in a.obs.selection_cluster.unique():
            mask = a.obs.selection_cluster.eq(cluster).values
            for j, gene in enumerate(genes):
                cluster_rows.append([row["sample"], row["tissue"], cluster, int(mask.sum()), gene,
                                     float((gene_counts[mask, j] > 0).mean()), float(gene_counts[mask, j].mean())])
        summary = a.obs.groupby("selection_cluster", observed=True).agg(
            n_cells=("sample", "size"), identity_positive_fraction=("identity_positive", "mean"),
            predicted_doublet_fraction=("predicted_doublet", "mean"),
            median_doublet_score=("doublet_score", "median"),
            median_genes=("n_genes_by_counts", "median"), median_UMI=("total_counts", "median"))
        summary.to_csv(OUT / f"{row['sample']}_cluster_stats.tsv", sep="\t")
        a.write_h5ad(OUT / f"{row['sample']}_annotated.h5ad", compression="gzip")
        all_decisions.append(a.obs.copy())
        sc.pl.umap(a, color=["leiden", "doublet_score", "identity_positive", "counts_Gata3", "counts_Rorc"], show=False)
        plt.savefig(OUT / f"{row['sample']}_umap.png", dpi=150, bbox_inches="tight"); plt.close("all")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(a.obs.doublet_score, bins=50, density=True, alpha=0.5, label="Observed")
        ax.hist(a.uns["scrublet"]["doublet_scores_sim"], bins=50, density=True, alpha=0.5, label="Simulated")
        ax.set_xlabel("Scrublet score"); ax.legend(); fig.tight_layout()
        fig.savefig(OUT / f"{row['sample']}_doublet_scores.png", dpi=150); plt.close(fig)
        threshold = float(a.uns["scrublet"].get("threshold", np.nan))
        print(f"  {a.n_obs} cells; {a.obs.leiden.nunique()} clusters; doublets={int(a.obs.predicted_doublet.sum())}; threshold={threshold:.3f}", flush=True)
        del a, gene_counts
        gc.collect()
    pd.DataFrame(cluster_rows, columns=["sample", "tissue", "cluster", "n_cells", "gene", "fraction_detected", "mean_raw_UMI"]).to_csv(OUT / "cluster_marker_evidence.tsv", sep="\t", index=False)
    pd.concat(all_decisions).to_csv(OUT / "basic_qc_pass_cell_evidence.tsv.gz", sep="\t")
    (OUT / "missing_markers.json").write_text(json.dumps(missing_all, indent=2))
    print("Evidence ready for cluster review", flush=True)


def finalize(review_path: Path) -> None:
    """群体复核+细胞程序复核，合并首版训练数据并保留可统计 metadata。"""
    review = json.loads(review_path.read_text())
    sample_metadata = pd.read_csv(ROOT / "data/metadata/GSE117567_sample_metadata.tsv", sep="\t", keep_default_na=False)
    retained_parts, decisions_parts, summary_rows, sensitivity_rows = [], [], [], []
    for row in sample_metadata.to_dict("records"):
        a = sc.read_h5ad(OUT / f"{row['sample_id']}_annotated.h5ad")
        threshold = review["scrublet_thresholds"][row["sample_id"]]
        if threshold == "auto":
            if "threshold" not in a.uns["scrublet"]:
                raise ValueError("自动阈值未识别，必须提供已审查的显式阈值。")
            threshold = float(a.uns["scrublet"]["threshold"])
            threshold_source = "auto_per_GSM"
        else:
            threshold = float(threshold)
            threshold_source = "explicit_reviewed_exploratory_threshold"
        a.obs["scrublet_auto_predicted_doublet"] = a.obs.predicted_doublet.astype(bool)
        a.obs["predicted_doublet"] = a.obs.doublet_score > threshold
        a.obs["scrublet_threshold"] = threshold
        a.obs["scrublet_threshold_source"] = threshold_source
        a.obs["doublet_screening_interpretation"] = "score_tail_review_not_groundtruth"
        clusters = set(a.obs.selection_cluster)
        mapping = review["cluster_labels"]
        if not clusters <= set(mapping):
            raise ValueError("有细胞群未经明确复核：" + str(clusters - set(mapping)))
        a.obs["cluster_annotation"] = a.obs.selection_cluster.map(mapping)
        alternative_cols = [c for c in a.obs if c.startswith("lineage_")]
        alternative = a.obs[alternative_cols].any(axis=1)
        flags = {
            "additional_quality": ~a.obs.additional_quality_pass,
            "doublet_review": a.obs.predicted_doublet.astype(bool),
            "non_ILC2_cluster": ~a.obs.cluster_annotation.eq("ILC2_supported"),
            "weak_ILC2_evidence": ~a.obs.identity_positive,
            "alternative_lineage": alternative,
            "ILC3_program_review": a.obs.ilc3_program_review,
            "BM_background_deferred": a.obs.tissue.eq("BM"),
        }
        for name, values in flags.items():
            a.obs["selection_fail_" + name] = values
        keep = ~np.column_stack(list(flags.values())).any(axis=1)
        a.obs["selected_for_training"] = keep
        a.obs["selection_reason"] = [";".join(name for name, fail in zip(flags, values) if fail) or "retained" for values in a.obs[["selection_fail_" + key for key in flags]].itertuples(index=False, name=None)]
        a.obs["cell_type"] = np.where(keep, "ILC2", a.obs.cluster_annotation)
        a.obs["identity_confidence"] = np.where(keep, "high", "review_or_excluded")
        a.obs["identity_confidence_method"] = "literature_informed_computational_screen_not_flow_validated"
        a.obs["training_tier"] = np.where(keep, "core_v1_peripheral", "not_in_core_v1")
        for key, value in row.items():
            a.obs[key] = value
        a.obs["split"] = "unassigned"  # 数据角色已确定，划分留给阶段2；不伪造独立验证。
        a.obs["data_role"] = "posttrain_candidate"
        a.obs["age_weeks"] = np.nan  # 只有研究年龄范围，不能填个体精确年龄。
        a.obs["activation_context"] = "steady_state_basal_activation"
        a.obs["proliferation_markers_detected"] = (a.obs.counts_Mki67 > 0) & (a.obs.counts_Top2a > 0)
        a.obs["type2_effector_detected"] = (a.obs.counts_Il5 > 0) | (a.obs.counts_Il13 > 0)
        a.obs["Areg_detected"] = a.obs.counts_Areg > 0
        flags_without_doublet = np.column_stack([value for name, value in flags.items() if name != "doublet_review"]).any(axis=1)
        for cutoff in [0.20, 0.25, 0.30]:
            sensitivity_rows.append({"sample_id": row["sample_id"], "tissue": row["tissue"], "scrublet_cutoff": cutoff,
                                     "retained_if_cutoff": int((~flags_without_doublet & (a.obs.doublet_score <= cutoff)).sum())})
        decisions_parts.append(a.obs.copy())
        s = {"sample_id": row["sample_id"], "tissue": row["tissue"],
             "raw_cells": int(row["raw_cells"]), "basic_qc_pass": a.n_obs, "retained_ILC2": int(keep.sum()),
             "predicted_doublet_flags": int(a.obs.predicted_doublet.sum())}
        s.update({"flag_" + name: int(value.sum()) for name, value in flags.items()})
        summary_rows.append(s)
        if keep.any():
            selected = a[keep].copy()
            # 最终文件的 X 与 counts 都是原始整数计数；不将归一化值冒充模型输入。
            selected.X = selected.layers["counts"].copy()
            selected.obsm.clear(); selected.obsp.clear(); selected.varm.clear()
            selected.uns.clear()
            retained_parts.append(selected)
        del a
        gc.collect()
    merged = ad.concat(retained_parts, join="inner", merge="same")
    assert merged.n_vars == 27998 and merged.obs_names.is_unique
    assert merged.obs.selected_for_training.all() and merged.obs.tissue.ne("BM").all()
    assert sparse.issparse(merged.layers["counts"]) and np.issubdtype(merged.X.dtype, np.integer)
    assert (merged.X != merged.layers["counts"]).nnz == 0
    assert np.array_equal(np.asarray(merged.X.sum(axis=1)).ravel(), merged.obs.total_counts.to_numpy())
    metadata = {
        "dataset_id": "GSE117567", "dataset_version": "core_v1_2026-10-14",
        "selection_type": "literature_informed_cluster_review_and_multigene_cell_screen",
        "confidence_interpretation": "computational_high_confidence_not_flow_validated",
        "samples": sample_metadata,
        "thresholds_json": json.dumps({"min_genes": 500, "min_UMI": 1000, "identity_detected_min": 3, "identity_requires_Gata3_or_Rora": True, "support_detected_min": 1, "scrublet_expected_rate": 0.05, "scrublet_thresholds": review["scrublet_thresholds"], "main_training_tissues": ["Lung", "Skin", "Gut", "Fat"]}),
        "cluster_review_json": json.dumps(review),
        "sources": ["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117567", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6202223/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7062532/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6625319/"],
        "counts_description": "X and layers/counts: raw integer UMI; Ensembl IDs; all original genes",
        "missing_marker_features_json": (OUT / "missing_markers.json").read_text(),
        "marker_count_columns_note": "Missing Trac/Trdc features are not biological-negative evidence; T program uses available CD3 genes",
        "missing_data": "individual age, sex, animal IDs, pool size and capture units unavailable; study range is not individual metadata",
    }
    merged.uns["metadata"] = metadata
    target = ROOT / "data/train_data/GSE117567.h5ad"
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(target, compression="gzip")
    pd.concat(decisions_parts).to_csv(OUT / "cell_selection_decisions.tsv.gz", sep="\t")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "sample_summary.tsv", sep="\t", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "doublet_threshold_sensitivity.tsv", sep="\t", index=False)
    merged.obs.to_csv(OUT / "train_cell_metadata.tsv.gz", sep="\t")
    merged.obs.groupby(["sample_id", "tissue", "cell_type"], observed=True).size().rename("n_cells").to_csv(OUT / "cell_counts_by_sample_tissue.tsv", sep="\t")
    check = sc.read_h5ad(target, backed="r")
    assert check.shape == merged.shape and check.obs_names.equals(merged.obs_names)
    assert set(check.obs.sample_id) == {"GSM3303971", "GSM3303972", "GSM3303973", "GSM3303974"}
    check.file.close()
    (OUT / "validation.json").write_text(json.dumps({"passed": True, "n_cells": merged.n_obs, "n_genes": merged.n_vars, "unique_cell_ids": True, "raw_counts_totals_preserved": True, "all_selected_pass_rules": True, "output_reloaded": True}, indent=2))
    print(summary[["sample_id", "tissue", "raw_cells", "basic_qc_pass", "retained_ILC2"]].to_string(index=False))
    print(f"Saved {target}: {merged.shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    sc.settings.n_jobs = 4
    sc.settings.verbosity = 1
    if args.finalize:
        if args.review is None:
            parser.error("--finalize 需要 --review")
        finalize(args.review)
    else:
        prepare()
