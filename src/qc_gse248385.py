"""GSE248385：共同 balanced_v1 QC → 计算高置信度 ILC2 原始 counts。

运行 python src/qc_gse248385.py；参数只读 configs/1QC.json。
数据存 data/processed/GSE248385_qc/，报告与三类基础图存 results/GSE248385_qc/。
"""
import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ilc2_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ilc2_mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from threadpoolctl import threadpool_limits

from qc_utils import _raw_counts, read_gene_annotation, qc_sample
from ilc2_qc import add_evidence, apply_standard
from qc_gse153748 import cluster

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/GSE248385_RAW"
DATA = ROOT / "data/processed/GSE248385_qc"
OUT = ROOT / "results/GSE248385_qc"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy = config["basic_qc"], config["identity"]
    reference = config["dataset_overrides"]["GSE248385"]
    annotation = read_gene_annotation(ROOT / reference["ensembl_annotation"])
    raw = sc.read_10x_mtx(RAW / "SITTB12_filtered_feature_bc_matrix",
                          var_names="gene_ids", make_unique=False, cache=False)
    raw.X = _raw_counts(raw.X)
    if not raw.obs_names.is_unique or not raw.var_names.is_unique:
        raise ValueError("Input gene IDs and barcodes must be unique")
    raw.var.rename(columns={"gene_symbols": "gene_symbol"}, inplace=True)
    raw.var.index.name = "gene_id"
    raw.obs["barcode"] = raw.obs_names
    raw.obs_names = "GSE248385:GSM7912851:" + raw.obs_names
    raw.obs.index.name = "cell_id"
    metadata = dict(study="GSE248385", sample="GSM7912851", tissue="Pancreas", species="Mouse",
        strain="C57BL6/J", condition="naive_steady_state", role="unassigned", split="unassigned",
        animal_id="not_available", biological_rep="not_available", pool_size="not_available",
        sex="not_available", age="not_available", genome_build="GRCm38.p6",
        annotation_release="not_available", qc_annotation_release=reference["ensembl_release"],
        library_protocol="10x_3prime_v3", cellranger_version="3.1.0",
        sort_gate="Live CD45+ CD3- B220- Lin- CD127+ NK1.1-",
        source_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248385")
    for key, value in metadata.items():
        raw.obs[key] = value
    mapped = annotation.reindex(raw.var_names)
    if mapped.gene_biotype.isna().any():
        raise ValueError("QC annotation does not cover all input genes; do not guess biotypes")
    raw.var["mt"] = mapped.is_mt.to_numpy(bool)
    sc.pp.calculate_qc_metrics(raw, qc_vars=["mt"], percent_top=None, inplace=True)
    candidate = ((raw.obs.n_genes_by_counts >= q["min_detected_genes"])
                 & (raw.obs.total_counts >= policy["min_candidate_UMI"]))
    print(f"Input {raw.shape}; candidates {candidate.sum()}", flush=True)
    result = qc_sample(raw[candidate].copy(), annotation, min_genes=q["min_detected_genes"],
        min_cells=q["min_cells_per_sample"], min_coding_mirna=q["min_detected_protein_coding_or_miRNA"],
        max_mt_pct=q["max_mt_pct_inclusive"], n_genes_upper_sd=q["n_genes_upper_sd"])
    a = result["core"]
    print(f"Basic QC {a.n_obs}; core genes {a.n_vars}; Scrublet", flush=True)
    sc.pp.scrublet(a, expected_doublet_rate=policy["scrublet_expected_rate"],
        threshold=policy["scrublet_review_threshold"], n_prin_comps=30,
        use_approx_neighbors=True, random_state=0, verbose=False)
    if not np.isfinite(a.obs.doublet_score).all():
        raise ValueError("Nonfinite Scrublet scores")
    # 完整counts中的TCR等feature不能因模型core的biotype限制丢失身份证据。
    evidence = result["full"]
    evidence.obs = a.obs.copy()
    missing_markers = add_evidence(evidence)
    print("Clustering and shared identity selection", flush=True)
    cluster(a)
    evidence.obs["leiden"] = a.obs.leiden
    evidence.obs["library_cluster"] = "GSM7912851:" + a.obs.leiden.astype(str)
    apply_standard(evidence, policy)
    a.obs = evidence.obs.copy()
    a.obs["qc_version"] = policy["preset"]
    a.obs["cell_type"] = np.where(a.obs.selected_ILC2, "ILC2", "review_or_excluded")
    a.obs["identity_confidence"] = np.where(a.obs.selected_ILC2, "high", "review")
    a.obs["identity_confidence_method"] = "computational_" + policy["preset"] + "_not_flow_validated"
    a.obs["training_tier"] = np.where(a.obs.selected_ILC2, policy["main_training_tier"], "excluded")
    a.obs["low_RNA_review"] = ~a.obs.additional_quality_pass
    a.obs["selection_reason"] = np.select(
        [~a.obs.identity_positive, a.obs.alternative_program],
        ["weak_identity", "alternative_program"], default="retained")
    decisions = raw.obs.copy()
    decisions["candidate_pass"] = candidate
    for column in result["decisions"]:
        if column not in decisions:
            decisions[column] = result["decisions"][column].reindex(decisions.index)
    for column in a.obs:
        if column not in decisions:
            decisions[column] = a.obs[column].reindex(decisions.index)
    decisions["basic_qc_pass"] = decisions.index.isin(a.obs_names)
    decisions["selected_ILC2"] = decisions.index.isin(a.obs_names[a.obs.selected_ILC2])
    decisions["final_reason"] = np.where(~candidate, "below_candidate_RNA",
        np.where(~decisions.basic_qc_pass, decisions.qc_reason, decisions.selection_reason))
    decisions.to_csv(OUT / "cell_decisions.tsv.gz", sep="\t")
    review = a.obs.groupby("leiden", observed=True).agg(n_cells=("sample", "size"),
        retained=("selected_ILC2", "sum"), alternative_fraction=("alternative_program", "mean"),
        doublet_review=("doublet_review", "sum"), median_UMI=("total_counts", "median"))
    review.to_csv(OUT / "cluster_review.tsv", sep="\t")
    sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    markers = sc.get.rank_genes_groups_df(a, group=None)
    markers["gene_symbol"] = markers.names.map(a.var.gene_symbol)
    markers.to_csv(OUT / "cluster_markers.tsv", sep="\t", index=False)

    # 最终对象保留所有输入基因；core仅用于聚类和Scrublet。
    selected = result["full"][a.obs_names[a.obs.selected_ILC2]].copy()
    selected.obs = a.obs.loc[selected.obs_names].copy()
    selected.uns.clear()
    provenance = dict(dataset="GSE248385", version=policy["preset"], basic_qc=q,
        identity_policy=policy, qc_reference=reference, missing_identity_markers=missing_markers,
        identity_interpretation="computational_high_confidence_not_experimentally_validated",
        gene_space="all input genes retained; var.is_core defines the analysis core",
        identity_gene_space="full_input_counts", basic_qc_summary=result["summary"],
        source_metadata=metadata)
    selected.uns["metadata"] = provenance
    selected.write_h5ad(DATA / "GSE248385_ILC2.h5ad", compression="gzip")
    a.uns["qc_provenance"] = provenance
    sc.tl.umap(a, min_dist=.5, random_state=0)
    a.write_h5ad(OUT / "GSE248385_qc_annotated.h5ad", compression="gzip")

    # 仅三类基本图：QC、基础UMAP、marker dotplot。
    sc.settings.set_figure_params(dpi=110, frameon=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(axes, ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
                             ["UMI per cell", "Detected genes per cell", "Mitochondrial UMI (%)"]):
        bins = np.histogram_bin_edges(raw.obs[key], bins=50)
        ax.hist(raw.obs[key], bins=bins, color="#3B75AF", alpha=.5, label="Input")
        ax.hist(selected.obs[key], bins=bins, color="#DB8B35", alpha=.6, label="ILC2 retained")
        ax.set(xlabel=label, ylabel="Cells")
        ax.legend(fontsize=8)
    fig.suptitle(f"GSE248385 | {raw.n_obs:,} input -> {selected.n_obs:,} ILC2")
    fig.tight_layout()
    fig.savefig(OUT / "qc.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    a.uns["leiden_colors"] = ["#3B75AF", "#DB8B35", "#499870", "#B76D9F", "#A38B37", "#8B6553", "#678B9B", "#777777"][:a.obs.leiden.nunique()]
    a.obs["cell_type"] = pd.Categorical(a.obs.cell_type, categories=["ILC2", "review_or_excluded"])
    a.uns["cell_type_colors"] = ["#3B75AF", "#DB8B35"]
    sc.pl.umap(a, color=["leiden", "cell_type"], ncols=2, size=10, show=False)
    plt.savefig(OUT / "umap.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    panel = {"ILC2": ["Gata3", "Rora", "Il7r", "Id2", "Il1rl1", "Il17rb"],
             "Effector / repair": ["Il5", "Il13", "Areg"],
             "Alternative": ["Cd3d", "Cd3e", "Cd3g", "Ncr1", "Klrd1", "Rorc", "Il23r", "Lyz2", "Prss2"]}
    panel = {key: [g for g in genes if a.var.gene_symbol.eq(g).any()] for key, genes in panel.items()}
    dp = sc.pl.dotplot(a, panel, groupby="leiden", gene_symbols="gene_symbol", use_raw=False,
                      return_fig=True, cmap="Blues", figsize=(12, 4.5))
    dp.savefig(OUT / "marker_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    summary = dict(input_cells=raw.n_obs, candidate_cells=int(candidate.sum()),
        basic_qc_cells=a.n_obs, high_confidence_ILC2=selected.n_obs, input_genes=raw.n_vars,
        core_genes=a.n_vars, final_genes=selected.n_vars, basic_qc=result["summary"],
        alternative_program=int(a.obs.alternative_program.sum()),
        weak_identity=int((~a.obs.identity_positive).sum()),
        doublet_review_retained=int(selected.obs.doublet_review.sum()), provenance=provenance,
        versions={p: importlib.metadata.version(p) for p in ["scanpy", "anndata", "numpy", "scipy"]})
    files = [Path(__file__), ROOT / "src/qc_utils.py", ROOT / "src/ilc2_qc.py",
             ROOT / "src/qc_gse153748.py", ROOT / "configs/1QC.json", ROOT / reference["ensembl_annotation"]]
    summary["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = f"""# GSE248385 QC

统一预设 `{policy['preset']}`，复用 `qc_sample`、`apply_standard`；未加严数值阈值。
输入 {raw.n_obs:,} → RNA候选 {candidate.sum():,} → 基础QC {a.n_obs:,} → 计算高置信度ILC2 **{selected.n_obs:,}**。
最终对象保留全部 {selected.n_vars:,} 个输入基因，X 与 counts 均为原始整数 UMI。

QC注释采用 GRCm38 Ensembl {reference['ensembl_release']}，全部输入ID覆盖；作者原始注释release仍未知。
默认release84缺少3495个输入ID，故仅覆盖本研究注释资产，不改变QC门槛；配置在 configs/1QC.json。
基础QC包含 coding/miRNA 检出数、注释定义的线粒体比例及候选文库均值+3SD上限
（本次上限 {result['summary']['n_genes_upper_limit']:.2f} genes）。

高置信度指多基因计算支持，并非流式验证；Scrublet高分、低RNA、混合群和状态不独立删除。
身份规则不变，在完整counts上调用，避免core的biotype限制丢失Trac/Trdc等已有谱系marker；
core视图只用于聚类和Scrublet。旧研究在core上调用身份规则，合并比较时需记录此输入视图差异。
每个细胞依自身程序判定，不整群删除；替代程序 {summary['alternative_program']}、弱身份 {summary['weak_identity']}，可重叠。
完整输入缺失marker：{', '.join(missing_markers) or '无'}；缺失feature不解释为生物学阴性。
一个GSM，缺少独立生物重复；role/split均为unassigned，未加入训练。

- 数据：data/processed/GSE248385_qc/GSE248385_ILC2.h5ad。
- 当前图仅 qc.png、umap.png、marker_dotplot.png。
- 去留：cell_decisions.tsv.gz；群复核：cluster_review.tsv、cluster_markers.tsv。
- 参数/来源：summary.json；分析对象：GSE248385_qc_annotated.h5ad（X为log1p，counts为原始core UMI）。
- 旧exploration已改为_qc，旧初步结果归档于 archive/initial_inspection，不代表当前QC。
- 复现：python src/qc_gse248385.py。
"""
    (OUT / "QC_report.md").write_text(report)
    print(f"Saved {selected.shape} high-confidence ILC2 counts to {DATA}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
