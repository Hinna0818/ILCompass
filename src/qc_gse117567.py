"""GSE117567 按统一 ilc2_balanced_v1 从原始 10x 三件套重跑（旧 strict_v1 对象不覆盖）。

运行：python src/qc_gse117567.py。规则唯一来源 configs/1QC.json；单样本流程与 qc_gse153748.py 相同
（候选→基础QC→Scrublet→按文库聚类→apply_standard），不含该研究特有的作者 signature/状态注释/UMAP。
外周四组织写 data/train_data/GSE117567_balanced_v1.h5ad，骨髓独立写 data/processed/GSE117567_qc/。
"""
from qc_gse153748 import cluster  # 先导入：它在 numba/scanpy 加载前设置线程环境变量
import json
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from qc_utils import read_10x_sample, read_gene_annotation, qc_sample
from ilc2_qc import add_evidence, apply_standard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/GSE117567_qc_balanced_v1"
RAW = ROOT / "data/GSE117567_RAW"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed/GSE117567_qc").mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy = config["basic_qc"], config["identity"]
    annotation = read_gene_annotation(ROOT / config["core_genes"]["ensembl_annotation"])
    metadata = pd.read_csv(ROOT / "data/metadata/GSE117567_sample_metadata.tsv", sep="\t")
    prefixes = {p.name.split("_barcodes")[0].split("_")[0]: p.name.removesuffix("_barcodes.tsv.gz") for p in RAW.glob("*_barcodes.tsv.gz")}
    parts, summaries, decisions_all = [], [], []
    for _, row in metadata.iterrows():
        raw = read_10x_sample(RAW, prefixes[row.sample_id], sample_id=row.sample_id)
        umi = np.asarray(raw.layers["counts"].sum(axis=1)).ravel()
        detected = np.diff(raw.layers["counts"].tocsr().indptr)
        candidate = (detected >= q["min_detected_genes"]) & (umi >= policy["min_candidate_UMI"])
        cand = raw[candidate].copy()
        result = qc_sample(cand, annotation, min_genes=q["min_detected_genes"], min_cells=q["min_cells_per_sample"],
                           min_coding_mirna=q["min_detected_protein_coding_or_miRNA"], max_mt_pct=q["max_mt_pct_inclusive"],
                           n_genes_upper_sd=q["n_genes_upper_sd"])
        a = result["core"]
        sc.pp.scrublet(a, expected_doublet_rate=policy["scrublet_expected_rate"], threshold=policy["scrublet_review_threshold"],
                       n_prin_comps=30, use_approx_neighbors=True, random_state=0, verbose=False)
        if not np.isfinite(a.obs.doublet_score).all():
            raise ValueError("Scrublet分数非有限，不能默认singlet")
        add_evidence(a)
        cluster(a)
        a.obs["library_cluster"] = row.sample_id + ":" + a.obs.leiden.astype(str)
        apply_standard(a, policy)
        a.obs["low_RNA_review"] = (a.obs.n_genes_by_counts < 500) | (a.obs.total_counts < 1000)
        for k, v in {"study": "GSE117567", "tissue": row.tissue, "species": "Mouse", "condition": row.condition,
                     "animal_id": "not_available", "biological_rep": "not_available", "split": "unassigned",
                     "qc_version": policy["preset"],
                     "identity_confidence": "computational_supported_not_flow_validated",
                     "training_tier": "BM_background_review" if row.tissue == "BM" else "peripheral_balanced"}.items():
            a.obs[k] = v
        d = result["decisions"].copy()
        d["candidate_pass"] = True
        d["selected_ILC2"] = d.index.map(a.obs.selected_ILC2).fillna(False).astype(bool)
        decisions_all.append(d)
        summaries.append({**result["summary"], "sample": row.sample_id, "tissue": row.tissue, "raw_cells": raw.n_obs,
                          "candidate_cells": int(candidate.sum()), "basic_qc_cells": a.n_obs,
                          "selected_ILC2": int(a.obs.selected_ILC2.sum()), "doublet_review_selected": int((a.obs.doublet_review & a.obs.selected_ILC2).sum()),
                          "low_RNA_review_selected": int((a.obs.low_RNA_review & a.obs.selected_ILC2).sum()),
                          "alternative_program": int(a.obs.alternative_program.sum()), "weak_identity": int((~a.obs.identity_positive).sum())})
        a.obs.groupby("library_cluster", observed=True).agg(n_cells=("sample", "size"), identity_fraction=("identity_positive", "mean"),
            alternative_fraction=("alternative_program", "mean"), doublet_flags=("doublet_review", "sum"),
            retained=("selected_ILC2", "sum")).to_csv(OUT / f"{row.sample_id}_cluster_review.tsv", sep="\t")
        sel = a[a.obs.selected_ILC2].copy()
        sel.X = sel.layers["counts"].copy()
        sel.uns.clear(); sel.obsm.clear(); sel.obsp.clear(); sel.varm.clear()
        parts.append(sel)
        print(row.tissue, raw.n_obs, "->", int(candidate.sum()), "->", a.n_obs, "->", sel.n_obs, flush=True)
    counts = ad.concat(parts, join="inner", merge="same")
    assert counts.obs_names.is_unique and np.issubdtype(counts.X.dtype, np.integer)
    for sub, path in [(counts[counts.obs.tissue.ne("BM")].copy(), ROOT / "data/train_data/GSE117567_balanced_v1.h5ad"),
                      (counts[counts.obs.tissue.eq("BM")].copy(), ROOT / "data/processed/GSE117567_qc/GSE117567_BM_ILC2.h5ad")]:
        sub.uns["metadata"] = {"dataset_id": "GSE117567", "version": policy["preset"], "basic_qc": q, "identity_policy": policy,
                               "samples": metadata, "batch_correction": False,
                               "identity_interpretation": "computational_supported_not_independently_validated"}
        sub.write_h5ad(path, compression="gzip")
        saved = sc.read_h5ad(path)
        assert saved.obs_names.equals(sub.obs_names) and (saved.X != sub.X).nnz == 0
    pd.DataFrame(summaries).to_csv(OUT / "sample_summary.tsv", sep="\t", index=False)
    pd.concat(decisions_all).to_csv(OUT / "cell_decisions.tsv.gz", sep="\t")
    files = [Path(__file__), ROOT / "src/ilc2_qc.py", ROOT / "src/qc_utils.py", ROOT / "configs/1QC.json",
             ROOT / config["core_genes"]["ensembl_annotation"], ROOT / "data/metadata/GSE117567_sample_metadata.tsv"]
    (OUT / "provenance.json").write_text(json.dumps({"version": policy["preset"], "seed": 0,
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "calling": "10x filtered matrix; genes>=200 & UMI>=500 candidate; heuristic_not_EmptyDrops"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
