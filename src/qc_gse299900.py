"""两个未处理肺捕获文库/四个hash样本：GRCm39注释、token覆盖和balanced_v1 QC。

运行 python src/qc_gse299900.py；不加载IL-25/感染样本，不启动后训练。
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
import pickle

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from threadpoolctl import threadpool_limits

from qc_utils import _raw_counts, _gene_ids, read_gene_annotation, qc_sample
from ilc2_qc import add_evidence, apply_standard
from qc_gse153748 import cluster
from genecompass_coverage import KEY_GENES, map_genes, summarize_coverage

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/GSE299900_RAW"
DATA = ROOT / "data/processed/GSE299900_qc"
OUT = ROOT / "results/GSE299900_qc"
LIBRARIES = {"A": "GSM9049537", "B": "GSM9049540"}


def export_metadata(a):
    """逐细胞表保持h5ad顺序；样本表仅使用已记录的来源字段。"""
    a.obs.to_csv(DATA / "GSE299900_ILC2_metadata.tsv.gz", sep="\t", index_label="cell_id")
    fields = ["study", "sample", "capture_library", "biological_rep", "hash_id", "processing_batch",
        "tissue", "species", "strain", "genotype", "condition", "sex", "age", "animal_id",
        "study_age_range", "study_sexes", "genome_build", "cellranger_reference", "annotation_release",
        "library_protocol", "cellranger_version", "source_file", "source_url", "role", "split", "qc_version"]
    samples = a.obs[fields].drop_duplicates().set_index("biological_rep")
    samples["retained_cells"] = a.obs.groupby("biological_rep", observed=True).size()
    samples.to_csv(DATA / "GSE299900_sample_metadata.tsv", sep="\t")


def read_library(day):
    parts, seen, inputs = [], set(), []
    for tag in ["h1", "h2"]:
        path = RAW / f"GSE299900_AK54_Uninfected_{day}_{tag}.h5"
        a = sc.read_10x_h5(path, gex_only=True)
        a.X = _raw_counts(a.X)
        symbols = a.var_names.to_numpy()
        original_ids = a.var.gene_ids.astype(str).to_numpy()
        a.var_names = _gene_ids(original_ids)
        a.var["gene_id_original"] = original_ids
        a.var["gene_symbol"] = symbols
        a.var.index.name = "gene_id"
        if not a.var_names.is_unique or not a.obs_names.is_unique or seen.intersection(a.obs_names):
            raise ValueError(f"Duplicate gene/cell or overlapping hash assignment in library {day}")
        seen.update(a.obs_names)
        if parts and not parts[0].var_names.equals(a.var_names):
            raise ValueError("Hash samples have inconsistent feature IDs/order")
        a.obs["barcode"] = a.obs_names
        a.obs_names = f"GSE299900:{LIBRARIES[day]}:{tag}:" + a.obs_names
        a.obs.index.name = "cell_id"
        values = dict(study="GSE299900", sample=LIBRARIES[day], capture_library=LIBRARIES[day],
            biological_rep=f"Uninfected_{day}_{tag}", hash_id=tag, processing_batch=day,
            tissue="Lung", species="Mouse", strain="C57BL/6", genotype="CD45.1",
            condition="naive_no_infection_no_cell_transfer", role="posttrain_candidate", split="unassigned",
            animal_id="not_available", sex="not_available", age="not_available",
            study_age_range="6-12 weeks at experiment start", study_sexes="both sexes",
            genome_build="GRCm39", cellranger_reference="GRCm39-2024-A", annotation_release="GENCODE_M33_Ensembl110",
            library_protocol="10x_3prime_v4", cellranger_version="9.0.0",
            sort_gate="live_CD45+_Lineage-_KLRG1+_with_CD90_or_YFP; exact gate per subset not independently reconstructed",
            source_file=path.name, source_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={LIBRARIES[day]}")
        for key, value in values.items():
            a.obs[key] = value
        with h5py.File(path) as f:
            types, numbers = np.unique(f["matrix/features/feature_type"][:], return_counts=True)
        inputs.append(dict(file=path.name, library=LIBRARIES[day], biological_rep=values["biological_rep"],
            cells=a.n_obs, gex_genes=a.n_vars, feature_types={t.decode(): int(n) for t, n in zip(types, numbers)},
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        parts.append(a)
    return ad.concat(parts, join="inner", merge="same"), inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy, reference = cfg["basic_qc"], cfg["identity"], cfg["dataset_overrides"]["GSE299900"]
    prep = json.loads((ROOT / "configs/2prepare_traindata.json").read_text())
    tokens = pickle.loads((ROOT / prep["token_dictionary"]).read_bytes())
    median = pickle.loads((ROOT / prep["gene_median_reference"]).read_bytes())
    annotation = read_gene_annotation(ROOT / reference["ensembl_annotation"])
    selected_parts, analyses, decisions_all, summaries, input_files, input_qc = [], [], [], [], [], []
    for day in LIBRARIES:
        raw, files = read_library(day)
        input_files.extend(files)
        mapped = annotation.reindex(raw.var_names)
        if mapped.gene_biotype.isna().any():
            raise ValueError(f"{int(mapped.gene_biotype.isna().sum())} GEX genes lack GRCm39-2024-A annotation")
        raw.var["mt"] = mapped.is_mt.to_numpy(bool)
        sc.pp.calculate_qc_metrics(raw, qc_vars=["mt"], percent_top=None, inplace=True)
        raw.var["genecompass_token"], _, raw.var["has_median_reference"] = map_genes(raw.var_names, tokens, median)
        raw.var["in_genecompass_vocab"] = raw.var.genecompass_token.ge(0)
        raw.var["reference_gene_symbol"] = mapped.gene_symbol.to_numpy()
        input_coverage = summarize_coverage(raw, raw.var.genecompass_token.to_numpy(), raw.var.has_median_reference.to_numpy())
        candidate = ((raw.obs.n_genes_by_counts >= q["min_detected_genes"])
                     & (raw.obs.total_counts >= policy["min_candidate_UMI"]))
        print(f"{LIBRARIES[day]} input {raw.shape}; candidates {candidate.sum()}; token UMI {input_coverage['umi_fraction_in_vocab']:.3%}", flush=True)
        result = qc_sample(raw[candidate].copy(), annotation, min_genes=q["min_detected_genes"],
            min_cells=q["min_cells_per_sample"], min_coding_mirna=q["min_detected_protein_coding_or_miRNA"],
            max_mt_pct=q["max_mt_pct_inclusive"], n_genes_upper_sd=q["n_genes_upper_sd"])
        a, evidence = result["core"], result["full"]
        print(f"{LIBRARIES[day]} basic QC {a.n_obs}; Scrublet and clustering", flush=True)
        sc.pp.scrublet(a, expected_doublet_rate=policy["scrublet_expected_rate"],
            threshold=policy["scrublet_review_threshold"], n_prin_comps=30,
            use_approx_neighbors=True, random_state=0, verbose=False)
        if not np.isfinite(a.obs.doublet_score).all():
            raise ValueError("Nonfinite Scrublet scores")
        evidence.obs = a.obs.copy()
        missing = add_evidence(evidence)
        cluster(a)
        evidence.obs["library_cluster"] = LIBRARIES[day] + ":" + a.obs.leiden.astype(str)
        apply_standard(evidence, policy)
        evidence.obs["qc_version"] = policy["preset"]
        evidence.obs["cell_type"] = np.where(evidence.obs.selected_ILC2, "ILC2", "review_or_excluded")
        evidence.obs["identity_confidence"] = np.where(evidence.obs.selected_ILC2, "high", "review")
        evidence.obs["identity_confidence_method"] = "computational_" + policy["preset"] + "_not_flow_validated"
        evidence.obs["training_tier"] = np.where(evidence.obs.selected_ILC2, policy["main_training_tier"], "excluded")
        evidence.obs["low_RNA_review"] = ~evidence.obs.additional_quality_pass
        evidence.obs["selection_reason"] = np.select([~evidence.obs.identity_positive, evidence.obs.alternative_program],
            ["weak_identity", "alternative_program"], default="retained")
        a.obs = evidence.obs.copy()
        d = raw.obs.copy()
        d["candidate_pass"] = candidate
        d = d.join(result["decisions"].drop(columns=d.columns.intersection(result["decisions"].columns)))
        d = d.join(a.obs.drop(columns=d.columns.intersection(a.obs.columns)))
        d["basic_qc_pass"] = d.index.isin(a.obs_names)
        d["selected_ILC2"] = d.index.isin(a.obs_names[a.obs.selected_ILC2])
        d["final_reason"] = np.where(~candidate, "below_candidate_RNA", np.where(~d.basic_qc_pass, d.qc_reason, d.selection_reason))
        decisions_all.append(d)
        sel = evidence[evidence.obs.selected_ILC2].copy()
        selected_parts.append(sel)
        analyses.append(a)
        input_qc.append(raw.obs[["sample", "total_counts", "n_genes_by_counts", "pct_counts_mt"]])
        token_col, _, has_median = map_genes(sel.var_names, tokens, median)
        selected_coverage = summarize_coverage(sel, token_col, has_median)
        summaries.append(dict(library=LIBRARIES[day], day=day, input_cells=raw.n_obs,
            candidate_cells=int(candidate.sum()), basic_qc_cells=a.n_obs, selected_ILC2=sel.n_obs,
            basic_qc=result["summary"], input_token_coverage=input_coverage, selected_token_coverage=selected_coverage,
            missing_identity_markers=missing, alternative_program=int(a.obs.alternative_program.sum()),
            weak_identity=int((~a.obs.identity_positive).sum()), doublet_review_retained=int(sel.obs.doublet_review.sum())))
        print(f"{LIBRARIES[day]} retained {sel.n_obs}", flush=True)

    if not selected_parts[0].var_names.equals(selected_parts[1].var_names):
        raise ValueError("Capture libraries have inconsistent gene IDs/order")
    selected = ad.concat(selected_parts, join="inner", merge="same")
    selected.uns.clear()
    provenance = dict(dataset="GSE299900", version=policy["preset"], basic_qc=q, identity_policy=policy,
        qc_reference=reference, libraries=summaries, source_files=input_files,
        identity_gene_space="full_input_GEX_counts", identity_interpretation="computational_high_confidence_not_experimentally_validated",
        sampling="two capture libraries; two hashed biological samples each; barcode-disjoint within each library",
        author_prefilter_status="per-hash Cell Ranger H5; exact preceding author filtering not established",
        token_mapping="exact stable Ensembl ID; no symbol remapping; original counts and all GEX genes retained",
        role="posttrain_candidate", split="unassigned")
    selected.uns["metadata"] = {k: v for k, v in provenance.items() if k not in {"libraries", "source_files"}}
    selected.uns["source_files"] = pd.DataFrame([{k: v for k, v in x.items() if k != "feature_types"} for x in input_files])
    selected.write_h5ad(DATA / "GSE299900_ILC2.h5ad", compression="gzip")
    export_metadata(selected)
    d = pd.concat(decisions_all)
    d.to_csv(OUT / "cell_decisions.tsv.gz", sep="\t")
    counts = d.groupby(["sample", "biological_rep"], observed=True).agg(input_cells=("candidate_pass", "size"),
        candidate_cells=("candidate_pass", "sum"), basic_qc_cells=("basic_qc_pass", "sum"), selected_ILC2=("selected_ILC2", "sum"))
    counts.to_csv(OUT / "sample_summary.tsv", sep="\t")
    selected.obs.groupby("biological_rep", observed=True)[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].median().to_csv(
        OUT / "retained_qc_medians.tsv", sep="\t")
    a = ad.concat(analyses, join="inner", merge="same")
    a.obs.groupby("library_cluster", observed=True).agg(n_cells=("sample", "size"), retained=("selected_ILC2", "sum"),
        alternative_fraction=("alternative_program", "mean")).to_csv(OUT / "cluster_review.tsv", sep="\t")
    # 联合UMAP只用于基本可视化；前面的QC/3SD/Scrublet均按真实捕获文库独立执行。
    a.X = a.layers["counts"].copy()
    cluster(a)
    sc.tl.umap(a, min_dist=.5, random_state=0)
    a.uns["qc_provenance"] = selected.uns["metadata"]
    a.write_h5ad(OUT / "GSE299900_qc_annotated.h5ad", compression="gzip")
    sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    markers = sc.get.rank_genes_groups_df(a, group=None)
    markers["gene_symbol"] = markers.names.map(a.var.gene_symbol)
    markers.to_csv(OUT / "cluster_markers.tsv", sep="\t", index=False)
    selected.var.to_csv(OUT / "gene_token_mapping.tsv.gz", sep="\t")
    selected.var[selected.var.gene_symbol.isin(KEY_GENES)].to_csv(OUT / "key_gene_token_mapping.tsv", sep="\t")
    token_col, _, has_median = map_genes(selected.var_names, tokens, median)
    total_coverage = summarize_coverage(selected, token_col, has_median)
    unmatched = selected.var.loc[~selected.var.in_genecompass_vocab, ["gene_symbol", "gene_biotype"]].copy()
    unmatched["total_UMI"] = np.asarray(selected.X[:, ~selected.var.in_genecompass_vocab.to_numpy()].sum(axis=0)).ravel()
    unmatched.sort_values("total_UMI", ascending=False).to_csv(OUT / "genes_not_in_vocab.tsv", sep="\t")

    sc.settings.set_figure_params(dpi=110, frameon=False)
    all_qc = pd.concat(input_qc)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(axes, ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
                             ["RNA UMI per cell", "Detected genes per cell", "Mitochondrial RNA UMI (%)"]):
        bins = np.histogram_bin_edges(all_qc[key], bins=50)
        ax.hist(all_qc[key], bins=bins, color="#3B75AF", alpha=.5, label="Input")
        ax.hist(selected.obs[key], bins=bins, color="#DB8B35", alpha=.6, label="ILC2 retained")
        ax.set(xlabel=label, ylabel="Cells")
        ax.legend(fontsize=8)
    fig.suptitle(f"GSE299900 uninfected lung | {len(d):,} input -> {selected.n_obs:,} ILC2")
    fig.tight_layout()
    fig.savefig(OUT / "qc.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    a.obs["cell_type"] = pd.Categorical(a.obs.cell_type, categories=["ILC2", "review_or_excluded"])
    a.uns["cell_type_colors"] = ["#3B75AF", "#DB8B35"]
    sc.pl.umap(a, color=["sample", "leiden", "cell_type"], ncols=3, size=5, show=False)
    plt.savefig(OUT / "umap.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    panel = {"ILC2": ["Gata3", "Rora", "Il7r", "Id2", "Il1rl1", "Il17rb"],
        "Effector / repair": ["Il5", "Il13", "Areg"],
        "Alternative": ["Cd3d", "Cd3e", "Ncr1", "Klrd1", "Rorc", "Il23r", "Lyz2"]}
    panel = {key: [g for g in genes if a.var.gene_symbol.eq(g).any()] for key, genes in panel.items()}
    dp = sc.pl.dotplot(a, panel, groupby="leiden", gene_symbols="gene_symbol", use_raw=False,
                      return_fig=True, cmap="Blues", figsize=(11, 4.5))
    dp.savefig(OUT / "marker_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    provenance.update(input_cells=len(d), basic_qc_cells=int(d.basic_qc_pass.sum()), selected_ILC2=selected.n_obs,
        input_genes=selected.n_vars, core_genes=a.n_vars, token_coverage=total_coverage,
        versions={p: importlib.metadata.version(p) for p in ["scanpy", "anndata", "numpy", "scipy"]})
    files = [Path(__file__), ROOT / "src/qc_utils.py", ROOT / "src/ilc2_qc.py", ROOT / "src/genecompass_coverage.py",
        ROOT / "src/qc_gse153748.py", ROOT / "configs/1QC.json", ROOT / reference["ensembl_annotation"],
        ROOT / prep["token_dictionary"], ROOT / prep["gene_median_reference"]]
    provenance["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT / "summary.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    lines = ["# GSE299900 未处理肺 ILC2 QC", "",
        f"输入 {len(d):,} → 基础QC {int(d.basic_qc_pass.sum()):,} → 计算高置信度ILC2 **{selected.n_obs:,}**。",
        f"两个GEX捕获文库，四个hash生物样本；最终保留全部 {selected.n_vars:,} 个RNA基因。", "",
        "GRCm39-2024-A 使用官方构建来源 GENCODE M33 / Ensembl110；全部输入GEX ID已核验注释覆盖。",
        "GeneCompass按稳定Ensembl ID精确匹配，不做symbol补猜、不删除词表外原始counts。",
        f"词表覆盖 {total_coverage['genes_in_vocab']:,} 基因、{total_coverage['umi_fraction_in_vocab']:.2%} 原始RNA UMI；",
        f"有median的词表内基因覆盖 {total_coverage['umi_fraction_with_median']:.2%} UMI。median仍为已有的人类同源代理，未重估。", "",
        "共同balanced_v1基础QC和身份规则均未加严；按捕获文库独立计算3SD及Scrublet。",
        "身份在完整RNA counts上判定；双细胞高分、低RNA和状态不独立删除。高置信度为计算支持，非实验验证。",
        "h5包含RNA/ADT/HTO三种feature，本次只将RNA用于QC和token统计；同捕获文库两个hash之间barcode无重叠。",
        "逐样本性别、年龄和animal_id未知；论文研究层面6–12周、两种性别不能当逐样本信息。",
        "作者h5是否经过进一步严格筛选未能仅凭文件确定；不声称重新获得了所有低RNA细胞。",
        "role=posttrain_candidate，split=unassigned；未改变现有训练数据或启动训练。", "",
        "数据：data/processed/GSE299900_qc/GSE299900_ILC2.h5ad，X=counts=原始整数UMI。",
        "基本图：qc.png、umap.png、marker_dotplot.png。",
        "数量：sample_summary.tsv；token表：gene_token_mapping.tsv.gz、key_gene_token_mapping.tsv。",
        "分析对象GSE299900_qc_annotated.h5ad仅为core视图，X为log1p，counts仍为原始UMI。",
        "复现：python src/qc_gse299900.py。"]
    (OUT / "QC_report.md").write_text("\n".join(lines) + "\n")
    print(counts.to_string(), flush=True)
    print(json.dumps(total_coverage, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
