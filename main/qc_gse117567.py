"""GSE117567 初步 QC 和 ILC3 污染审查；保留完整基因及原始 counts。

运行：conda activate ilc2_model && python main/qc_gse117567.py
先产生聚类与 marker 表；审查后用 --review JSON --finalize 完成可追溯过滤。
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ilc2_model_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/ilc2_model_numba")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "GSM3303970_BAM": "BM", "GSM3303971_Adult_Lung": "Lung",
    "GSM3303972_Adult_skin": "Skin", "GSM3303973_Adult_gut": "Gut",
    "GSM3303974_FAT": "Fat",
}
PANELS = {
    "ILC2": ["Gata3", "Rora", "Il7r", "Il1rl1", "Il17rb", "Il5", "Il13", "Areg", "Arg1", "Klrg1", "Icos"],
    "ILC3": ["Rorc", "Il23r", "Il22", "Il17a", "Il17f", "Ccr6", "Kit", "Ahr", "Il1r1", "Lta", "Ltb"],
    "T_cell": ["Cd3d", "Cd3e", "Cd3g", "Trac", "Trdc"],
    "NK_ILC1": ["Ncr1", "Nkg7", "Tbx21", "Eomes", "Prf1", "Gzmb"],
    "Other": ["Cd79a", "Ms4a1", "Lyz2", "Adgre1", "Epcam", "Krt19", "Pecam1", "Col1a1", "Mki67"],
}


import sys
sys.path.insert(0, str(ROOT / "src"))
from ilc2_qc import marker_values


def marker_tables(a, out):
    genes = list(dict.fromkeys(g for panel in PANELS.values() for g in panel))
    genes = [g for g in genes if g in set(a.var.gene_symbol)]
    counts = marker_values(a, genes, "counts")
    expr = marker_values(a, genes)
    rows = []
    for group in ["tissue", "leiden", "tissue_cluster"]:
        for key in a.obs[group].unique():
            m = a.obs[group].eq(key).values
            for j, gene in enumerate(genes):
                rows.append([group, key, int(m.sum()), gene, float((counts[m, j] > 0).mean()), float(expr[m, j].mean())])
    pd.DataFrame(rows, columns=["groupby", "group", "n_cells", "gene", "fraction_detected", "mean_log1p"]).to_csv(out / "marker_summary.tsv", sep="\t", index=False)
    for j, gene in enumerate(genes):
        a.obs[f"counts_{gene}"] = counts[:, j].astype(int)
    # Rorc 单阳性与泛 ILC marker（Kit/Ahr/Il1r1）不能独立决定删除。
    core = ["Rorc", "Il23r", "Il22", "Il17a", "Il17f", "Ccr6"]
    c = marker_values(a, core, "counts")
    a.obs["ilc3_core_detected"] = (c > 0).sum(axis=1)
    a.obs["ilc3_candidate"] = ((c[:, 0] > 0) & ((c[:, 1:] > 0).sum(axis=1) >= 1)) | ((c[:, 1:] > 0).sum(axis=1) >= 3)
    a.obs.loc[a.obs.ilc3_candidate].to_csv(out / "ilc3_candidates.tsv", sep="\t")
    return genes


def analyze(args, out):
    parts, raw_obs, hashes = [], [], {}
    gene_reference = None
    for prefix, tissue in SAMPLES.items():
        print(f"Reading {prefix}", flush=True)
        paths = {kind: args.input / f"{prefix}_{suffix}" for kind, suffix in {
            "matrix": "matrix.mtx.gz", "genes": "genes.tsv.gz", "barcodes": "barcodes.tsv.gz"}.items()}
        for path in paths.values():
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path.name] = digest.hexdigest()
        genes = pd.read_csv(paths["genes"], sep="\t", header=None, names=["gene_id", "gene_symbol"])
        barcodes = pd.read_csv(paths["barcodes"], sep="\t", header=None)[0]
        with gzip.open(paths["matrix"], "rb") as fh:
            x = mmread(fh).T.tocsr()
        assert x.shape == (len(barcodes), len(genes)) and genes.gene_id.is_unique
        assert np.all(x.data >= 0) and np.all(x.data == np.floor(x.data))
        if gene_reference is None:
            gene_reference = genes.copy()
        else:
            assert genes.equals(gene_reference), "基因顺序或注释不一致"
        a = ad.AnnData(x.astype(np.float32), var=genes.set_index("gene_id"))
        a.obs_names = prefix.split("_")[0] + ":" + barcodes.astype(str)
        a.obs["barcode"] = barcodes.to_numpy()
        a.obs["sample"] = prefix.split("_")[0]
        a.obs["tissue"] = tissue
        a.obs["study"] = "GSE117567"
        a.obs["condition"] = "steady_state"
        a.obs["animal_id"] = "not_available"
        a.var["mt"] = a.var.gene_symbol.str.lower().str.startswith("mt-")
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
        a.obs["low_genes"] = a.obs.n_genes_by_counts < args.min_genes
        a.obs["low_counts"] = a.obs.total_counts < args.min_counts
        a.obs["high_mt"] = a.obs.pct_counts_mt >= args.max_mt
        a.obs["qc_pass"] = ~(a.obs.low_genes | a.obs.low_counts | a.obs.high_mt)
        # 高 UMI 是复核提示；ILC2 激活也能提高 RNA 含量，不直接当作双细胞。
        logcounts = np.log1p(a.obs.total_counts.to_numpy())
        med = np.median(logcounts)
        mad = np.median(np.abs(logcounts - med))
        a.obs["high_counts_flag"] = logcounts > med + 5 * mad
        raw_obs.append(a.obs.copy())
        parts.append(a[a.obs.qc_pass].copy())
        del a, x
        gc.collect()
    raw = pd.concat(raw_obs)
    raw.to_csv(out / "all_cells_qc.tsv.gz", sep="\t")
    a = ad.concat(parts, merge="same")
    del parts
    gc.collect()
    assert a.obs_names.is_unique
    a.layers["counts"] = a.X.astype(np.int32)
    print(f"QC retained {a.n_obs} cells; normalization and clustering", flush=True)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # 样本仅用于 HVG 选择；组织和文库混杂，不能据此直接消除组织差异。
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=3000, batch_key="sample")
    sc.pp.pca(a, n_comps=40, mask_var="highly_variable", svd_solver="arpack", random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=40, random_state=0)
    sc.tl.leiden(a, resolution=0.8, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    # 每个组织另行聚类，防止全局图被组织差异支配而掩盖少量污染。
    a.obs["tissue_cluster"] = ""
    for tissue in SAMPLES.values():
        print(f"Within-tissue clusters: {tissue}", flush=True)
        w = a[a.obs.tissue.eq(tissue), a.var.highly_variable].copy()
        w.layers.clear()
        sc.pp.pca(w, n_comps=30, svd_solver="arpack", random_state=0)
        sc.pp.neighbors(w, n_neighbors=15, n_pcs=30, random_state=0)
        sc.tl.leiden(w, resolution=0.6, flavor="igraph", n_iterations=2, directed=False, random_state=0)
        a.obs.loc[w.obs_names, "tissue_cluster"] = tissue + ":" + w.obs.leiden.astype(str)
        del w
    sc.tl.umap(a, random_state=0)
    genes = marker_tables(a, out)
    print("Exploratory cluster markers", flush=True)
    sc.tl.rank_genes_groups(a, "tissue_cluster", method="t-test_overestim_var", use_raw=False, n_genes=30)
    markers = sc.get.rank_genes_groups_df(a, group=None)
    markers["gene_symbol"] = markers.names.map(a.var.gene_symbol)
    markers.to_csv(out / "cluster_top_markers.tsv", sep="\t", index=False)
    a.obs.to_csv(out / "qc_pass_cells_metadata.tsv.gz", sep="\t")
    a.uns["qc_parameters"] = {"min_genes": args.min_genes, "min_counts": args.min_counts, "max_mt_exclusive": args.max_mt}
    a.write_h5ad(out / "GSE117567_qc_annotated.h5ad", compression="gzip")
    versions = {p: importlib.metadata.version(p) for p in ["scanpy", "anndata", "numpy", "scipy", "pandas", "igraph"]}
    (out / "provenance.json").write_text(json.dumps({"input": os.path.relpath(args.input.resolve(), ROOT), "sha256": hashes, "versions": versions, "seed": 0, "qc_parameters": a.uns["qc_parameters"]}, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["n_genes_by_counts", "total_counts", "pct_counts_mt"]):
        ax.boxplot([raw.loc[raw.tissue.eq(t), col] for t in SAMPLES.values()], labels=list(SAMPLES.values()), showfliers=False)
        ax.set_title(col)
    fig.tight_layout(); fig.savefig(out / "qc_by_tissue.png", dpi=180); plt.close(fig)
    sc.pl.umap(a, color=["tissue", "leiden", "ilc3_candidate", "high_counts_flag"], show=False)
    plt.savefig(out / "umap_overview.png", dpi=180, bbox_inches="tight"); plt.close("all")
    plot_genes = [g for g in ["Gata3", "Il5", "Rorc", "Il23r", "Il22", "Ccr6", "Cd3e", "Ncr1"] if g in genes]
    vmax = [max(1, float(np.quantile(a.obs.loc[a.obs["counts_" + g] > 0, "counts_" + g], 0.95))) if (a.obs["counts_" + g] > 0).any() else 1 for g in plot_genes]
    sc.pl.umap(a, color=["counts_" + g for g in plot_genes], vmin=0, vmax=vmax, show=False)
    plt.savefig(out / "umap_markers.png", dpi=180, bbox_inches="tight"); plt.close("all")
    print("Analysis ready for ILC3 review", flush=True)


def finalize(args, out):
    review = json.loads(args.review.read_text())
    a = sc.read_h5ad(out / "GSE117567_qc_annotated.h5ad")
    raw = pd.read_csv(out / "all_cells_qc.tsv.gz", sep="\t", index_col=0)
    removed_clusters = review["remove_ilc3_clusters"]
    removed_cells = review.get("remove_ilc3_cells", [])
    assert set(removed_clusters) <= set(a.obs.tissue_cluster)
    assert set(removed_cells) <= set(a.obs_names)
    a.obs["remove_ilc3"] = a.obs.tissue_cluster.isin(removed_clusters) | a.obs_names.isin(removed_cells)
    if review.get("rule") == "Rorc_Il22_conservative_v1":
        o = a.obs
        rule = (o.counts_Rorc >= 1) & (o.counts_Il22 >= 5) & ((o.counts_Il5 + o.counts_Il13) == 0) & (o.counts_Il17rb == 0) & ((o.counts_Cd3d + o.counts_Cd3e + o.counts_Cd3g) == 0)
        assert np.array_equal(a.obs.remove_ilc3.to_numpy(), rule.to_numpy()), "审查细胞清单与记录的规则不符"
    a.obs["annotation"] = np.where(a.obs.remove_ilc3, "ILC3_like_excluded", np.where(a.obs.ilc3_candidate, "ILC2_enriched_ILC3_signal_review", "ILC2_enriched"))
    other_labels = review.get("other_cluster_flags", {})
    assert set(other_labels) <= set(a.obs.tissue_cluster)
    a.obs["other_lineage_flag"] = a.obs.tissue_cluster.isin(other_labels)
    for cluster, label in other_labels.items():
        a.obs.loc[a.obs.tissue_cluster.eq(cluster) & ~a.obs.remove_ilc3, "annotation"] = label
    bm = a.obs.tissue.eq("BM") & ~a.obs.other_lineage_flag & ~a.obs.remove_ilc3
    a.obs.loc[bm, "annotation"] = "BM_ILC_precursor_review"
    a.uns["ilc3_review_json"] = json.dumps(review)
    a.obs.to_csv(out / "qc_pass_cells_metadata.tsv.gz", sep="\t")
    raw["final_status"] = "QC_failed"
    raw.loc[a.obs_names, "final_status"] = np.where(a.obs.remove_ilc3, "ILC3_like_excluded", "retained")
    raw.to_csv(out / "all_cells_qc.tsv.gz", sep="\t")
    a.write_h5ad(out / "GSE117567_qc_annotated.h5ad", compression="gzip")
    retained = a[~a.obs.remove_ilc3].copy()
    retained.write_h5ad(out / "GSE117567_qc_noILC3.h5ad", compression="gzip")
    retained.obs.to_csv(out / "retained_cells_metadata.tsv.gz", sep="\t")
    a.obs.loc[a.obs.remove_ilc3].to_csv(out / "excluded_ilc3_cells.tsv", sep="\t")
    rows = []
    for tissue in SAMPLES.values():
        r = raw[raw.tissue.eq(tissue)]
        q = a.obs[a.obs.tissue.eq(tissue)]
        f = retained.obs[retained.obs.tissue.eq(tissue)]
        rows.append({"tissue": tissue, "raw_cells": len(r), "qc_failed": int((~r.qc_pass).sum()), "qc_pass": len(q), "ilc3_excluded": int(q.remove_ilc3.sum()), "retained": len(f), "ilc3_signal_retained": int(f.ilc3_candidate.sum()), "other_lineage_flag": int(f.other_lineage_flag.sum()), "median_genes": float(f.n_genes_by_counts.median()), "median_UMI": float(f.total_counts.median()), "median_mt_pct": float(f.pct_counts_mt.median())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "sample_summary.tsv", sep="\t", index=False)
    sc.pl.umap(a, color=["tissue", "annotation"], show=False)
    plt.savefig(out / "umap_filter_decisions.png", dpi=180, bbox_inches="tight"); plt.close("all")
    plot_genes = ["Gata3", "Il5", "Rorc", "Il23r", "Il22", "Ccr6", "Cd3e", "Ncr1"]
    vmax = [max(1, float(np.quantile(a.obs.loc[a.obs["counts_" + g] > 0, "counts_" + g], 0.95))) if (a.obs["counts_" + g] > 0).any() else 1 for g in plot_genes]
    sc.pl.umap(a, color=["counts_" + g for g in plot_genes], vmin=0, vmax=vmax, show=False)
    plt.savefig(out / "umap_markers.png", dpi=180, bbox_inches="tight"); plt.close("all")
    groups = a.obs.groupby(["tissue", "annotation"], observed=True).size().rename("n_cells").reset_index()
    groups.to_csv(out / "annotation_counts.tsv", sep="\t", index=False)
    columns = ["tissue", "raw_cells", "qc_failed", "ilc3_excluded", "retained", "median_genes", "median_UMI", "median_mt_pct"]
    qc = a.uns["qc_parameters"]
    table = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    table += "\n".join("| " + " | ".join(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row) + " |" for row in summary[columns].itertuples(index=False, name=None))
    report = f'''# GSE117567 初步 Scanpy QC 与 ILC3-like 排除报告

原始 {len(raw):,} 个细胞，QC 保留 {a.n_obs:,} 个；保守排除 {int(a.obs.remove_ilc3.sum())} 个 ILC3-like 候选，最终保留 {retained.n_obs:,} 个、27,998 个基因。这是初步清理数据，不能等同于已确认的纯 ILC2 训练集。

{table}

## 处理方法

- 每个文库分别计算 QC；保留检测基因数 ≥{qc['min_genes']}、UMI ≥{qc['min_counts']}、线粒体计数比例 <{qc['max_mt_exclusive']}% 的细胞。失败条件可能重叠，总删除数按细胞去重。
- 全基因原始整数 counts 保存在 `layers["counts"]`；`X` 是每细胞总量归一化到 10,000 后自然对数 log1p，只用于 Scanpy 探索，不是 GeneCompass 的最终输入。
- HVG 3,000 个（按 sample 选择），40 PCs、15 邻居；全局 Leiden resolution=0.8，各组织内 30 PCs、Leiden resolution=0.6，随机种子 0。未做组织间批次校正。
- `cluster_top_markers.tsv` 为探索性聚类 marker；细胞级 P 值不能当作独立动物重复的差异表达证据。
- 对每个组织 log1p(UMI) 超出中位数 +5 MAD 的细胞仅标记 `high_counts_flag`，未自动删除；未运行专门的双细胞检测或环境 RNA 去污染。

## ILC3 审查与删除依据

没有发现可按整群统一删除的典型 ILC3 群。共 {int(a.obs.ilc3_candidate.sum())} 个细胞进入宽松的 ILC3 信号候选清单：Rorc 与另一个核心基因共同检出，或 Il23r/Il22/Il17a/Il17f/Ccr6 中至少三个检出。

进一步保守排除采用原始 UMI：Rorc ≥1 且 Il22 ≥5，同时 Il5、Il13、Il17rb 及 Cd3d/e/g 都为 0。阈值是本次初筛的启发式规则，不是文献验证的分类器；缺失转录本也不能证明蛋白阴性。四个肺细胞位于 Lung:4，一个肠细胞位于 Gut:5；所在整个细胞群未删除。详细细胞 ID 与表达证据见 `excluded_ilc3_cells.tsv`，固定审查决定见 `ilc3_review.json`。

剩余 {int(retained.obs.ilc3_candidate.sum())} 个带有 ILC3 相关信号的细胞保留并标记，不能都称为污染。例如皮肤细胞可共同表达 Gata3/Il5/Il13 和 Ccr6/Il17a/f；骨髓 Rorc 信号也可能涉及发育阶段。仅凭 Rorc、Ccr6、Kit 或 Ahr 阳性不予删除。本轮排除结果只能称为 ILC3-like 候选，不能声称完全清除了所有 ILC3。

## 其他需要注意的细胞

额外标记 {int(retained.obs.other_lineage_flag.sum())} 个其他谱系可疑细胞，按本轮 ILC3 过滤范围保留，未静默删除。包括骨髓 NK/ILC1-like、T-like、B/myeloid-like，肺内皮样，肠/脂肪基质样，以及皮肤其他谱系待复核群。依据是聚类 top markers 与多基因表达；详见 `annotation_counts.tsv` 和 `ilc3_review.json`。剩余骨髓 ILC 也标记为发育/前体待复核。

这些标记意味着最终文件不能直接全部用于“成熟纯 ILC2”post-training；先完成其他谱系和骨髓发育状态复核。每个组织一套公开文库，组织与样本来源混杂，无独立动物 ID。

## 文件与复现

- `GSE117567_qc_noILC3.h5ad`：最终保留细胞，全基因 counts、QC、注释、PCA/UMAP。
- `GSE117567_qc_annotated.h5ad`：QC 后、删除 ILC3-like 前的对象，包含删除标记，可恢复审查。
- `all_cells_qc.tsv.gz`：全部原始细胞及 QC/最终保留状态。
- `retained_cells_metadata.tsv.gz`：最终细胞信息；`sample_summary.tsv`：组织级数量与 QC 中位数。
- `marker_summary.tsv`、`cluster_top_markers.tsv`、`ilc3_candidates.tsv`：注释证据。
- UMAP 在 QC 后全体细胞上计算，过滤对象保留原坐标及诱导子图，未重算邻居或 UMAP。
- `provenance.json`：输入 SHA256、软件版本、QC 参数。

```bash
conda activate ilc2_model
python main/qc_gse117567.py
python main/qc_gse117567.py --finalize --review results/GSE117567_qc/ilc3_review.json
```

原始研究：[GEO GSE117567](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117567)，[Nature Immunology 2018](https://doi.org/10.1038/s41590-018-0201-4)。ILC3 marker 的参考：[Reciprocal Transcription Factor Networks Govern Tissue-Resident ILC3 Subset Function and Identity](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611981/)。
'''
    (out / "QC_report.md").write_text(report)
    assert len(raw) == int(summary.qc_failed.sum()) + int(summary.ilc3_excluded.sum()) + retained.n_obs
    assert a.n_vars == retained.n_vars == 27998
    assert sparse.issparse(retained.layers["counts"]) and np.issubdtype(retained.layers["counts"].dtype, np.integer)
    assert np.array_equal(np.asarray(retained.layers["counts"].sum(axis=1)).ravel(), retained.obs.total_counts.to_numpy())
    assert retained.obs.qc_pass.all() and not retained.obs.remove_ilc3.any()
    # 小型非稠密校验：重新读取文件，确保细胞清单和 counts 持久化正确。
    check = sc.read_h5ad(out / "GSE117567_qc_noILC3.h5ad", backed="r")
    assert check.shape == retained.shape and check.obs_names.equals(retained.obs_names)
    check.file.close()
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/GSE117567_RAW")
    parser.add_argument("--output", type=Path, default=ROOT / "results/GSE117567_qc")
    parser.add_argument("--min-genes", type=int, default=500)
    parser.add_argument("--min-counts", type=int, default=1000)
    parser.add_argument("--max-mt", type=float, default=15)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sc.settings.n_jobs = 4
    sc.settings.verbosity = 1
    if args.finalize:
        if args.review is None:
            parser.error("--finalize 需要 --review")
        finalize(args, args.output)
    else:
        analyze(args, args.output)
