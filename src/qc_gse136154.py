"""GSE136154 四组肺 ILC 文库：统一 balanced_v1 QC，保留 downstream_only counts。

运行 python src/qc_gse136154.py。联合 UMAP 仅作 QC 展示，不输出下游训练特征。
"""
import os

for key, value in {
    "NUMBA_CACHE_DIR": "/tmp/ilc2_numba", "MPLCONFIGDIR": "/tmp/ilc2_mpl",
    "NUMBA_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
    "LOKY_MAX_CPU_COUNT": "4",
}.items():
    os.environ.setdefault(key, value)

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from threadpoolctl import threadpool_limits

from genecompass_coverage import map_genes, summarize_coverage
from ilc2_qc import add_evidence, apply_standard
from qc_gse153748 import cluster
from qc_utils import _gene_ids, _raw_counts, qc_sample, read_gene_annotation

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/downstream_data/GSE136154"
DATA = ROOT / "data/downstream_data/GSE136154_qc"
OUT = ROOT / "results/GSE136154_qc"


def plot_qc(a, decisions, selected, umap_colors=None):
    font_manager.fontManager.addfont(ROOT / "external/fonts/Arial.ttf")
    sc.settings.set_figure_params(dpi=110, frameon=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, key, label in zip(axes,
            ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
            ["UMI per cell", "Detected genes", "Mitochondrial UMI (%)"]):
        bins = np.histogram_bin_edges(decisions[key], bins=60)
        ax.hist(decisions[key], bins=bins, alpha=.5, color="#3B75AF", label="Input")
        ax.hist(selected.obs[key], bins=bins, alpha=.6, color="#DB8B35", label="Retained")
        ax.set(xlabel=label, ylabel="Cells")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "qc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    a.obs["qc_identity"] = pd.Categorical(
        np.where(a.obs.selected_ILC2, "ILC2 retained", "Review / excluded"),
        categories=["ILC2 retained", "Review / excluded"])
    a.uns["qc_identity_colors"] = ["#3B75AF", "#DB8B35"]
    sc.pl.umap(a, color=umap_colors or ["condition", "author_batch", "leiden", "qc_identity"],
               ncols=2, size=3, show=False)
    plt.savefig(OUT / "umap.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    panel = {
        "ILC2": ["Gata3", "Rora", "Il7r", "Id2", "Thy1", "Il1rl1", "Il17rb"],
        "State (not filtering)": ["Il5", "Il13", "Areg", "Mki67", "Crem"],
        "Alternative lineage": ["Cd3d", "Cd3e", "Trac", "Ncr1", "Klrd1", "Rorc", "Il23r", "Lyz2", "Epcam"],
    }
    panel = {k: [g for g in genes if a.var.gene_symbol.eq(g).any()] for k, genes in panel.items()}
    dp = sc.pl.dotplot(a, panel, groupby="leiden", gene_symbols="gene_symbol", use_raw=False,
                      return_fig=True, cmap="Blues", figsize=(12, max(4, a.obs.leiden.nunique() * .24)))
    dp.savefig(OUT / "marker_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy = cfg["basic_qc"], cfg["identity"]
    reference = cfg["core_genes"]
    annotation = read_gene_annotation(ROOT / reference["ensembl_annotation"])
    prep = json.loads((ROOT / "configs/2prepare_traindata.json").read_text())
    tokens = pickle.loads((ROOT / prep["token_dictionary"]).read_bytes())
    median = pickle.loads((ROOT / prep["gene_median_reference"]).read_bytes())
    libraries = pd.read_csv(RAW / "library_metadata.tsv", sep="\t", keep_default_na=False)
    geo = pd.read_csv(RAW / "geo_samples.tsv", sep="\t", keep_default_na=False).set_index("gsm")
    sources = {r["filename"]: r for r in json.loads((RAW / "source_manifest.json").read_text())}
    expected = {f"GSM40417{i}" for i in range(54, 62)}
    if set(libraries.gsm) != expected or libraries.gsm.duplicated().any():
        raise ValueError("Expected the eight audited scRNA-seq libraries, without duplicate GSMs")
    parts, all_decisions, summaries, inputs = [], [], [], []
    for row in libraries.to_dict("records"):
        path = RAW / row["file"]
        source = sources[path.name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Input differs from download manifest: {path}")
        raw = sc.read_10x_h5(path, gex_only=True)
        raw.X = _raw_counts(raw.X)
        raw.var["gene_symbol"] = raw.var_names.to_numpy()
        raw.var["gene_id_original"] = raw.var.gene_ids.astype(str).to_numpy()
        raw.var_names = _gene_ids(raw.var.gene_id_original)
        raw.var.index.name = "gene_id"
        if not raw.var_names.is_unique or not raw.obs_names.is_unique:
            raise ValueError(f"Duplicate gene IDs/barcodes: {row['gsm']}")
        if parts and not parts[0].var_names.equals(raw.var_names):
            raise ValueError("Input feature IDs/order differ; do not silently intersect genes")
        mapped = annotation.reindex(raw.var_names)
        if mapped.gene_biotype.isna().any():
            raise ValueError(f"QC annotation does not cover all input genes: {row['gsm']}")
        raw.var["mt"] = mapped.is_mt.to_numpy(bool)
        raw.var["reference_gene_symbol"] = mapped.gene_symbol.to_numpy()
        raw.var["genecompass_token"], _, raw.var["has_median_reference"] = map_genes(raw.var_names, tokens, median)
        raw.var["in_genecompass_vocab"] = raw.var.genecompass_token.ge(0)
        with h5py.File(path) as f:
            genomes = sorted(set(f["matrix/features/genome"].asstr()[:]))
            chemistry = str(f.attrs.get("chemistry_description", "not_available"))
        if genomes != ["mm10"]:
            raise ValueError(f"Unexpected input genome: {genomes}")
        raw.obs["barcode"] = raw.obs_names
        raw.obs_names = "GSE136154:" + row["gsm"] + ":" + raw.obs_names
        raw.obs.index.name = "cell_id"
        fields = dict(study="GSE136154", sample=row["gsm"], gsm=row["gsm"],
            library_id=row["library_id"], capture_library=row["gsm"], condition=row["condition"],
            author_replicate=str(row["author_replicate"]), author_batch=row["author_batch"],
            tissue="Lung", species="Mouse", strain="C57BL/6", animal_id="not_available",
            sex="not_available", age="not_available", genotype="not_available",
            experimental_context="in_vivo", author_timepoint="3 days",
            treatment_protocol=geo.loc[row["gsm"], "treatment"],
            author_characteristics=geo.loc[row["gsm"], "characteristics"],
            genome_build="mm10", qc_annotation_release=str(reference["ensembl_release"]),
            h5_chemistry=chemistry, source_file=path.name, source_url=source["url"],
            role="downstream_only", split="unassigned")
        for key, value in fields.items():
            raw.obs[key] = value
        sc.pp.calculate_qc_metrics(raw, qc_vars=["mt"], percent_top=None, inplace=True)
        candidate = ((raw.obs.n_genes_by_counts >= q["min_detected_genes"])
                     & (raw.obs.total_counts >= policy["min_candidate_UMI"]))
        d = raw.obs.copy()
        d["candidate_pass"] = candidate
        result = qc_sample(raw[candidate].copy(), annotation,
            min_genes=q["min_detected_genes"], min_cells=q["min_cells_per_sample"],
            min_coding_mirna=q["min_detected_protein_coding_or_miRNA"],
            max_mt_pct=q["max_mt_pct_inclusive"], n_genes_upper_sd=q["n_genes_upper_sd"])
        full, core = result["full"], result["core"]
        print(f"{row['library_id']}: input {raw.n_obs:,} -> candidates {int(candidate.sum()):,} "
              f"-> basic QC {full.n_obs:,}; Scrublet", flush=True)
        sc.pp.scrublet(core, expected_doublet_rate=policy["scrublet_expected_rate"],
            threshold=policy["scrublet_review_threshold"], n_prin_comps=30,
            use_approx_neighbors=True, random_state=cfg["random_seed"], verbose=False)
        if not np.isfinite(core.obs.doublet_score).all():
            raise ValueError("Scrublet returned nonfinite scores")
        full.obs = core.obs.copy()
        parts.append(full)
        d = d.join(result["decisions"].drop(columns=d.columns.intersection(result["decisions"].columns)))
        d["basic_qc_pass"] = d.index.isin(full.obs_names)
        all_decisions.append(d)
        summaries.append(dict(sample=row["gsm"], library_id=row["library_id"], condition=row["condition"],
            author_batch=row["author_batch"], input_cells=raw.n_obs, candidate_cells=int(candidate.sum()),
            basic_qc_cells=full.n_obs, basic_qc=result["summary"]))
        inputs.append(dict(sample=row["gsm"], **source))
        del raw, core, result
        gc.collect()

    full = ad.concat(parts, join="inner", merge="same")
    del parts
    missing_markers = add_evidence(full)
    analysis = full[:, full.var.is_core].copy()
    print(f"Joint QC visualization: {analysis.shape}; normalization, PCA, neighbors, Leiden", flush=True)
    cluster(analysis)
    full.obs["leiden"] = analysis.obs.leiden
    full.obs["library_cluster"] = full.obs["sample"].astype(str) + ":" + full.obs.leiden.astype(str)
    apply_standard(full, policy)
    full.obs["qc_version"] = policy["preset"]
    full.obs["identity_confidence"] = np.where(full.obs.selected_ILC2, "high", "review")
    full.obs["identity_confidence_method"] = "computational_ilc2_balanced_v1_not_experimentally_validated"
    full.obs["cell_type"] = np.where(full.obs.selected_ILC2, "ILC2", "review_or_excluded")
    full.obs["low_RNA_review"] = ~full.obs.additional_quality_pass
    full.obs["selection_reason"] = np.select(
        [~full.obs.identity_positive, full.obs.alternative_program],
        ["weak_identity", "alternative_program"], default="retained")
    analysis.obs = full.obs.copy()
    selected = full[full.obs.selected_ILC2].copy()
    coverage = summarize_coverage(selected, selected.var.genecompass_token.to_numpy(),
                                  selected.var.has_median_reference.to_numpy())
    decisions = pd.concat(all_decisions)
    decisions = decisions.join(full.obs.drop(columns=decisions.columns.intersection(full.obs.columns)))
    decisions["selected_ILC2"] = decisions.index.isin(selected.obs_names)
    decisions["final_reason"] = np.where(~decisions.candidate_pass, "below_candidate_RNA",
        np.where(~decisions.basic_qc_pass, decisions.qc_reason, decisions.selection_reason))
    for s in summaries:
        obs = full.obs.loc[full.obs["sample"].eq(s["sample"])]
        keep = obs.loc[obs.selected_ILC2]
        s.update(selected_ILC2=len(keep), weak_identity=int((~obs.identity_positive).sum()),
            alternative_program=int(obs.alternative_program.sum()),
            doublet_review_retained=int(keep.doublet_review.sum()), low_RNA_retained=int(keep.low_RNA_review.sum()))
    table = pd.DataFrame([{k: v for k, v in s.items() if k != "basic_qc"} for s in summaries])
    table["retention_pct"] = table.selected_ILC2 / table.input_cells * 100
    table.to_csv(OUT / "sample_summary.tsv", sep="\t", index=False)
    condition = table.groupby("condition", sort=False)[
        ["input_cells", "candidate_cells", "basic_qc_cells", "selected_ILC2"]].sum()
    condition["retention_pct"] = condition.selected_ILC2 / condition.input_cells * 100
    condition.to_csv(OUT / "condition_summary.tsv", sep="\t")
    selected.obs.groupby("library_id", observed=True)[
        ["total_counts", "n_genes_by_counts", "pct_counts_mt"]].median().to_csv(
            OUT / "retained_qc_medians.tsv", sep="\t")
    decisions.groupby(["library_id", "final_reason"], observed=True).size().rename("cells").to_csv(
        OUT / "exclusion_reasons.tsv", sep="\t")
    full.obs.groupby(["leiden", "condition"], observed=True).agg(
        cells=("sample", "size"), retained=("selected_ILC2", "sum"),
        alternative_fraction=("alternative_program", "mean")).to_csv(OUT / "cluster_review.tsv", sep="\t")
    provenance = dict(dataset="GSE136154", qc_version=policy["preset"], basic_qc=q,
        identity_policy=policy, qc_reference=reference, input_cells=len(decisions),
        basic_qc_cells=full.n_obs, selected_ILC2=selected.n_obs, input_genes=full.n_vars,
        analysis_core_genes=analysis.n_vars, token_coverage=coverage, missing_identity_markers=missing_markers,
        role="downstream_only", split="unassigned", source_files=inputs, samples=summaries,
        identity_interpretation="computational_support_not_experimental_validation",
        gene_space="all_input_GEX_genes; integer_raw_counts_in_X_and_counts_layer",
        annotation_interpretation="GRCm38 release 84 covers all IDs; author annotation release not verified",
        author_prefilter_status="Cell Ranger filtered matrix; not assumed to include authors' downstream QC",
        visualization_scope="All basic-QC cells; joint HVG/PCA/UMAP only for QC; refit training features without held-out conditions",
        software={p: importlib.metadata.version(p) for p in ["scanpy", "anndata", "numpy", "scipy", "scrublet"]})
    selected.uns.clear()
    selected.uns["qc_provenance_json"] = json.dumps(
        {k: v for k, v in provenance.items() if k not in {"source_files", "samples"}}, ensure_ascii=False)
    selected.uns["source_files"] = pd.DataFrame(inputs)
    selected.uns["qc_version"] = policy["preset"]
    selected.write_h5ad(DATA / "GSE136154_ILC2.h5ad", compression="gzip")
    selected.obs.to_csv(DATA / "GSE136154_ILC2_metadata.tsv.gz", sep="\t", index_label="cell_id")
    sample_metadata = pd.DataFrame([dict(sample=s["sample"], retained_cells=s["selected_ILC2"],
        **{k: full.obs.loc[full.obs["sample"].eq(s["sample"]), k].iloc[0]
           for k in fields if k != "sample"}) for s in summaries])
    sample_metadata.to_csv(DATA / "GSE136154_sample_metadata.tsv", sep="\t", index=False)
    selected.var.to_csv(OUT / "gene_token_mapping.tsv.gz", sep="\t")

    print(f"Retained {selected.n_obs:,} ILC2; UMAP and marker review", flush=True)
    sc.tl.umap(analysis, min_dist=.5, random_state=cfg["random_seed"])
    decisions = decisions.join(pd.DataFrame(analysis.obsm["X_umap"], index=analysis.obs_names,
                                           columns=["qc_umap_1", "qc_umap_2"]))
    decisions.to_csv(OUT / "cell_decisions.tsv.gz", sep="\t")
    sc.tl.rank_genes_groups(analysis, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    markers = sc.get.rank_genes_groups_df(analysis, group=None)
    markers["gene_symbol"] = markers.names.map(analysis.var.gene_symbol)
    markers.to_csv(OUT / "cluster_markers.tsv", sep="\t", index=False)
    plot_qc(analysis, decisions, selected)
    provenance["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "src/qc_utils.py", ROOT / "src/ilc2_qc.py",
                  ROOT / "src/qc_gse153748.py", ROOT / "configs/1QC.json"]}
    (OUT / "summary.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    report = ["# GSE136154 四组肺 ILC 文库 QC", "",
        f"输入 {len(decisions):,} → 基础QC {full.n_obs:,} → 计算高置信度ILC2 **{selected.n_obs:,}**。",
        "```tsv\n" + table.to_csv(sep="\t", index=False) + "```",
        "条件汇总：\n```tsv\n" + condition.to_csv(sep="\t") + "```",
        f"统一 {policy['preset']}，阈值未加严：genes≥{q['min_detected_genes']}、UMI≥{policy['min_candidate_UMI']}、"
        f"protein_coding/miRNA检出≥{q['min_detected_protein_coding_or_miRNA']}、mt≤{q['max_mt_pct_inclusive']}%；"
        f"按文库的候选细胞计算检出基因数均值+{q['n_genes_upper_sd']}SD（ddof=0）上限，样本过滤前后≥{q['min_cells_per_sample']}细胞。",
        "基础QC与Scrublet按GSM独立计算。核心身份≥1基因、核心+支持≥2；替代谱系按共享多基因规则排除。",
        "双细胞高分、低RNA、增殖/效应/干扰素及混合群只标记，不独立删除。高置信度为计算支持，并非实验验证。",
        f"最终 {selected.n_vars:,} 个原始输入基因全部保留；分析core为 {analysis.n_vars:,} 基因。"
        "X与layers['counts']均为相同整数原始UMI；唯一稳定Ensembl ID及完整来源metadata保留。",
        f"QC注释为{reference['ensembl_annotation']}，覆盖全部输入ID；H5标记mm10，作者注释release未经独立确认。",
        "输入为作者提供的Cell Ranger filtered矩阵，不等于复现作者论文最终QC；本项目不套用作者分条件的不同基因数阈值。",
        f"GeneCompass精确ID覆盖 {coverage['genes_in_vocab']:,} 基因、{coverage['umi_fraction_in_vocab']:.2%} UMI；"
        f"有median的词表内基因覆盖 {coverage['umi_fraction_with_median']:.2%} UMI。词表外基因未删除，median未重估。",
        "缺失身份面板feature：" + (", ".join(missing_markers) or "无") + "。",
        "C/D为作者实验batch，每批四组。author_replicate保留作者编号；单只动物/混池对应、逐文库年龄性别未知，不推断。",
        "四组均为downstream_only，split=unassigned；未纳入posttrain，未训练预测器或计算组合预测性能。",
        "联合HVG/PCA/UMAP仅用于QC展示；最终counts对象不附带这些拟合特征。下游组合预测须仅用训练条件重新拟合特征与模型。",
        "仅生成qc.png、umap.png、marker_dotplot.png三类图；去留表、原因、文库/条件汇总及marker表在本结果目录。",
        "数据与逐细胞/逐文库metadata：data/downstream_data/GSE136154_qc/。",
        "复现：python src/qc_gse136154.py。"]
    (OUT / "QC_report.md").write_text("\n\n".join(report) + "\n")
    print(table.to_string(index=False), flush=True)
    print(f"Saved {selected.shape} to {DATA}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
