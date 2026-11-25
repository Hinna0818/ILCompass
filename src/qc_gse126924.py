"""GSE126924 Lung: shared balanced_v1 QC and stage-specific Il13 detection.

Run: python src/qc_gse126924.py. Skin is intentionally not included.
"""
import os

for key, value in {"NUMBA_CACHE_DIR": "/tmp/ilc2_numba", "MPLCONFIGDIR": "/tmp/ilc2_mpl",
                   "NUMBA_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                   "OMP_NUM_THREADS": "4"}.items():
    os.environ.setdefault(key, value)

import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from threadpoolctl import threadpool_limits

from qc_utils import _gene_ids, _raw_counts, read_gene_annotation, qc_sample
from ilc2_qc import add_evidence, apply_standard, marker_values
from qc_gse153748 import cluster

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/GSE126924_qc"
OUT = ROOT / "results/GSE126924_qc"
SOURCE = ROOT / "data/raw/GSE126924/GSE126924_RAW.tar"
SAMPLE = "GSM3618237"


def read_lung():
    prefix = SAMPLE + "_Lung"
    with tarfile.open(SOURCE) as archive:
        with gzip.open(archive.extractfile(prefix + "_genes.tsv.gz"), "rt") as f:
            genes = pd.read_csv(f, sep="\t", header=None, dtype=str)
        with gzip.open(archive.extractfile(prefix + "_barcodes.tsv.gz"), "rt") as f:
            barcodes = pd.read_csv(f, header=None, dtype=str)[0]
        with gzip.open(archive.extractfile(prefix + "_matrix.mtx.gz"), "rb") as f:
            x = _raw_counts(mmread(f).T)
    ids = _gene_ids(genes[0])
    if x.shape != (len(barcodes), len(ids)) or not ids.is_unique or not barcodes.is_unique:
        raise ValueError("Invalid matrix dimensions or nonunique gene/cell IDs")
    var = pd.DataFrame({"gene_id_original": genes[0].values, "gene_symbol": genes[1].values}, index=ids)
    var.index.name = "gene_id"
    obs = pd.DataFrame({"barcode": barcodes.values}, index="GSE126924:" + SAMPLE + ":" + barcodes)
    obs.index.name = "cell_id"
    # Age/condition are inherited from the local source inventory, not inferred from expression.
    metadata = dict(study="GSE126924", sample=SAMPLE, library_id=SAMPLE, tissue="Lung",
                    species="Mouse", age="P14", condition="steady_state",
                    source_metadata="data/metadata/ILC2数据集汇总1015.csv",
                    animal_id="not_available", biological_rep="not_available", sex="not_available",
                    genotype="not_verified", split="unassigned", role="unassigned")
    for key, value in metadata.items():
        obs[key] = value
    raw = ad.AnnData(x, obs=obs, var=var)
    raw.layers["counts"] = x.copy()
    return raw


def detection(a, stage):
    counts = marker_values(a, ["Il13"], "counts").ravel()
    total = np.asarray(a.layers["counts"].sum(axis=1)).ravel()
    cp10k = np.divide(counts * 10000., total, out=np.zeros(len(total)), where=total > 0)
    return dict(stage=stage, cells=a.n_obs, Il13_positive=int((counts > 0).sum()),
                Il13_detection_rate=float((counts > 0).mean()),
                Il13_mean_raw_counts=float(counts.mean()), Il13_mean_CP10K=float(cp10k.mean()),
                Il13_positive_median_counts=float(np.median(counts[counts > 0])),
                Il13_counts_1=int((counts == 1).sum()), Il13_counts_2=int((counts == 2).sum()),
                Il13_counts_ge3=int((counts >= 3).sum()))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy = config["basic_qc"], config["identity"]
    reference = ROOT / config["core_genes"]["ensembl_annotation"]
    annotation = read_gene_annotation(reference)
    raw = read_lung()
    if not raw.var.gene_symbol.eq("Il13").any():
        raise ValueError("Il13 is absent from input features")
    mapped = annotation.reindex(raw.var_names)
    if mapped.gene_biotype.isna().any():
        raise ValueError("Incomplete reference annotation; do not guess gene biotypes")
    raw.var["mt"] = mapped.is_mt.to_numpy(bool)
    sc.pp.calculate_qc_metrics(raw, qc_vars=["mt"], percent_top=None, inplace=True)
    candidate = ((raw.obs.n_genes_by_counts >= q["min_detected_genes"])
                 & (raw.obs.total_counts >= policy["min_candidate_UMI"]))
    result = qc_sample(raw[candidate].copy(), annotation,
                       min_genes=q["min_detected_genes"], min_cells=q["min_cells_per_sample"],
                       min_coding_mirna=q["min_detected_protein_coding_or_miRNA"],
                       max_mt_pct=q["max_mt_pct_inclusive"], n_genes_upper_sd=q["n_genes_upper_sd"])
    full, analysis = result["full"], result["core"]
    print(f"Raw={raw.n_obs}; RNA candidates={candidate.sum()}; basic QC={full.n_obs}", flush=True)
    sc.pp.scrublet(analysis, expected_doublet_rate=policy["scrublet_expected_rate"],
                   threshold=policy["scrublet_review_threshold"], n_prin_comps=30,
                   use_approx_neighbors=True, random_state=config["random_seed"], verbose=False)
    if not np.isfinite(analysis.obs.doublet_score).all():
        raise ValueError("Invalid Scrublet scores")
    cluster(analysis)
    full.obs = analysis.obs.copy()
    missing = add_evidence(full)
    full.obs["library_cluster"] = SAMPLE + ":" + full.obs.leiden.astype(str)
    apply_standard(full, policy)
    full.obs["qc_version"] = policy["preset"]
    full.obs["cell_type"] = np.where(full.obs.selected_ILC2, "ILC2_candidate", "review_or_excluded")
    full.obs["identity_confidence_method"] = "computational_balanced_v1_not_experimentally_validated"
    full.obs["low_RNA_review"] = ~full.obs.additional_quality_pass
    full.uns["identity_selection_status"] = "computational_balanced_v1"
    full.obs["selection_reason"] = np.select(
        [~full.obs.identity_positive, full.obs.alternative_program],
        ["weak_identity", "alternative_program"], default="retained")
    # Il13 is a support marker in the shared rule. Quantify, but do not alter, this potential selection bias.
    support_without_il13 = (full.obs.ILC2_identity_n_detected + full.obs.ILC2_support_n_detected
                           - (full.obs.counts_Il13 > 0).astype(int))
    full.obs["identity_pass_without_Il13_support"] = (
        (full.obs.ILC2_identity_n_detected >= policy["min_core_markers"])
        & (support_without_il13 >= policy["min_total_identity_support_markers"])
        & ~full.obs.alternative_program)
    selected = full[full.obs.selected_ILC2].copy()
    stages = [(raw, "input"), (raw[candidate], "RNA_candidates"), (full, "basic_QC"),
              (selected, "balanced_ILC2"),
              (full[full.obs.identity_pass_without_Il13_support], "ILC2_without_Il13_identity_support"),
              (raw[~raw.obs_names.isin(full.obs_names)], "removed_by_basic_QC"),
              (full[~full.obs.selected_ILC2], "removed_by_identity")]
    stats = pd.DataFrame([detection(a, name) for a, name in stages if a.n_obs])
    stats.to_csv(OUT / "Il13_detection_by_stage.csv", index=False)
    print(stats.to_string(index=False), flush=True)

    decisions = raw.obs.copy()
    decisions["Il13_counts"] = marker_values(raw, ["Il13"], "counts").ravel()
    decisions["candidate_pass"] = candidate
    for frame in [result["decisions"], full.obs]:
        for col in frame.columns.difference(decisions.columns):
            decisions[col] = frame[col].reindex(decisions.index)
    decisions["basic_qc_pass"] = decisions.index.isin(full.obs_names)
    decisions["selected_ILC2"] = decisions.index.isin(selected.obs_names)
    decisions["final_reason"] = np.where(~candidate, "below_candidate_RNA",
        np.where(~decisions.basic_qc_pass, decisions.qc_reason, decisions.selection_reason))
    decisions.to_csv(OUT / "cell_decisions.tsv.gz", sep="\t")
    full.obs.groupby("leiden", observed=True).agg(
        cells=("sample", "size"), retained=("selected_ILC2", "sum"),
        Il13_positive=("counts_Il13", lambda x: int((x > 0).sum())),
        alternative_fraction=("alternative_program", "mean"), doublet_review=("doublet_review", "sum")
    ).to_csv(OUT / "cluster_review.tsv", sep="\t")
    analysis.obs = full.obs.copy()
    sc.tl.rank_genes_groups(analysis, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    markers = sc.get.rank_genes_groups_df(analysis, group=None)
    markers["gene_symbol"] = markers.names.map(analysis.var.gene_symbol)
    markers.to_csv(OUT / "cluster_markers.tsv", sep="\t", index=False)
    sc.tl.umap(analysis, min_dist=.5, random_state=config["random_seed"])
    full.obsm["X_umap"] = analysis.obsm["X_umap"].copy()
    selected.obsm["X_umap"] = analysis[selected.obs_names].obsm["X_umap"].copy()
    provenance = dict(preset=policy["preset"], basic_qc=result["summary"], policy=policy,
                      qc_annotation=str(reference.relative_to(ROOT)), missing_identity_markers=missing,
                      author_annotation_release="not_verified", full_input_genes_retained=True,
                      versions={p: importlib.metadata.version(p) for p in ["scanpy", "anndata", "numpy", "scipy"]},
                      sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [SOURCE, Path(__file__), ROOT / "configs/1QC.json"]})
    for a in [full, selected]:
        if (a.X != raw[a.obs_names, a.var_names].X).nnz or (a.X != a.layers["counts"]).nnz:
            raise ValueError("Original counts were altered")
        a.uns["qc_provenance"] = provenance
    full.write_h5ad(DATA / "GSE126924_Lung_basicQC.h5ad", compression="gzip")
    selected.write_h5ad(DATA / "GSE126924_Lung_ILC2.h5ad", compression="gzip")
    (OUT / "summary.json").write_text(json.dumps(dict(provenance=provenance, stages=stats.to_dict("records")), indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, key, label in zip(axes, ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
                              ["UMI", "Detected genes", "Mitochondrial UMI (%)"]):
        bins = np.histogram_bin_edges(raw.obs[key], bins=50)
        ax.hist(raw.obs[key], bins=bins, alpha=.5, label="Input", color="#607D8B")
        ax.hist(selected.obs[key], bins=bins, alpha=.6, label="ILC2 retained", color="#2A9D8F")
        ax.set(xlabel=label, ylabel="Cells")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "qc.png", dpi=180)
    plt.close("all")
    sc.pl.umap(analysis, color=["leiden", "cell_type"], ncols=2, show=False, frameon=False)
    plt.savefig(OUT / "umap.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    panel = ["Gata3", "Rora", "Il7r", "Id2", "Il1rl1", "Il17rb", "Il5", "Il13", "Areg",
             "Cd3d", "Cd3e", "Ncr1", "Klrd1", "Rorc", "Il23r", "Lyz2"]
    panel = [g for g in panel if analysis.var.gene_symbol.eq(g).any()]
    dp = sc.pl.dotplot(analysis, panel, groupby="leiden", gene_symbols="gene_symbol", use_raw=False,
                      return_fig=True, cmap="Blues", figsize=(10, 4))
    dp.savefig(OUT / "marker_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    table = "\n".join(f"|{r.stage}|{r.cells:,}|{r.Il13_positive:,}|{r.Il13_detection_rate:.2%}|"
                      for r in stats.itertuples())
    report = f"""# GSE126924 Lung QC 与 Il13 检出

文库：{SAMPLE}；仅处理Lung，未处理Skin。按本地来源清单为P14稳态肺；动物/重复信息未提供，不能把细胞视为生物重复。
复现：`python src/qc_gse126924.py`。参数只读 `configs/1QC.json`，复用 `qc_sample`、`apply_standard`，无研究特异阈值改动。
参考注释GRCm38 release84覆盖全部{raw.n_vars:,}个输入ID；这是QC注释，作者具体参考release未独立确认。

|阶段|细胞数|Il13 counts>0|检出率|
|---|---:|---:|---:|
{table}

基础QC：≥{q['min_detected_genes']}检出基因、≥{policy['min_candidate_UMI']}UMI确定RNA候选；coding/miRNA检出≥{q['min_detected_protein_coding_or_miRNA']}，mt≤{q['max_mt_pct_inclusive']}%；候选内检出基因上界均值+3SD={result['summary']['n_genes_upper_limit']:.2f}。在完整基因空间计算；仅分析视图限制biotype。
ILC2身份使用共同多基因正向及替代谱系规则，不要求Il13阳性。不使用状态得分删细胞。
Il13属于默认支持面板，因此另报去掉该单一身份证据的敏感性结果，未改变正式筛选。
Scrublet仅标记复核，不独立删细胞；保留细胞中{int(selected.obs.doublet_review.sum())}个有双细胞复核标记。
缺失身份marker：{', '.join(missing) or '无'}。计算身份支持不是独立实验验证。

检出定义为原始UMI>0；CP10K仅为描述性统计，不改变counts。该检出率不等于GeneCompass输入Token覆盖率，也不证明IL-13蛋白分泌。
UMAP和dotplot仅使用归一化分析视图；两个h5ad保留所有输入基因、整数X/counts、来源和去留标签，split/role未分配。
未做FM推理、未加入预训练。后续作为功能读出参考还需检查候选靶基因与Il13的共同Token覆盖。

输出：`data/processed/GSE126924_qc/GSE126924_Lung_basicQC.h5ad`、`GSE126924_Lung_ILC2.h5ad`；
本目录包含`Il13_detection_by_stage.csv`、逐细胞去留、群marker、summary及QC/UMAP/dotplot三类基础图。
"""
    (OUT / "QC_report.md").write_text(report)
    print(f"Completed: {OUT / 'QC_report.md'}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
