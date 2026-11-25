"""五个WT稳态control文库：原始H5→balanced_v1→ILC2 counts与基础Scanpy图。

运行 python src/qc_gse301975_301976.py --study GSE301975（或GSE301976）。
只读取data/metadata/内已审核的control清单，不下载处理样本、不划分训练/测试。
"""
import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ilc2_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ilc2_mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
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
CONTROL_SAMPLES = {"GSE301975": {"GSM9093319", "GSM9093320"},
                   "GSE301976": {"GSM9093327", "GSM9093328", "GSM9093329"}}


def main(study):
    data = ROOT / f"data/processed/{study}_qc"
    out = ROOT / f"results/{study}_qc"
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy = cfg["basic_qc"], cfg["identity"]
    reference = cfg["dataset_overrides"][study]
    annotation = read_gene_annotation(ROOT / reference["ensembl_annotation"])
    prep = json.loads((ROOT / "configs/2prepare_traindata.json").read_text())
    tokens = pickle.loads((ROOT / prep["token_dictionary"]).read_bytes())
    median = pickle.loads((ROOT / prep["gene_median_reference"]).read_bytes())
    metadata = pd.read_csv(ROOT / f"data/metadata/{study}_sample_metadata.tsv", sep="\t", keep_default_na=False)
    if set(metadata["sample"]) != CONTROL_SAMPLES[study] or metadata["sample"].duplicated().any():
        raise ValueError("Input manifest must contain exactly the audited WT control libraries")
    parts, decisions, summaries, source_files = [], [], [], []
    for row in metadata.to_dict("records"):
        path = ROOT / row["relative_path"]
        raw = sc.read_10x_h5(path, gex_only=True)
        raw.X = _raw_counts(raw.X)
        raw.var["gene_symbol"] = raw.var_names.to_numpy()
        raw.var["gene_id_original"] = raw.var.gene_ids.astype(str).to_numpy()
        raw.var_names = _gene_ids(raw.var.gene_id_original)
        raw.var.index.name = "gene_id"
        if not raw.var_names.is_unique or not raw.obs_names.is_unique:
            raise ValueError(f"Duplicate input gene/barcode IDs: {row['sample']}")
        mapped = annotation.reindex(raw.var_names)
        if mapped.gene_biotype.isna().any():
            raise ValueError(f"QC annotation does not cover all input IDs: {row['sample']}")
        if parts and not parts[0].var_names.equals(raw.var_names):
            raise ValueError("Libraries differ in feature IDs/order; do not silently drop genes")
        raw.var["mt"] = mapped.is_mt.to_numpy(bool)
        raw.var["reference_gene_symbol"] = mapped.gene_symbol.to_numpy()
        raw.var["genecompass_token"], _, raw.var["has_median_reference"] = map_genes(raw.var_names, tokens, median)
        raw.var["in_genecompass_vocab"] = raw.var.genecompass_token.ge(0)
        raw.obs["barcode"] = raw.obs_names
        raw.obs_names = study + ":" + row["sample"] + ":" + raw.obs_names
        raw.obs.index.name = "cell_id"
        sc.pp.calculate_qc_metrics(raw, qc_vars=["mt"], percent_top=None, inplace=True)
        # 原始空液滴只计入barcode_positions；去留表追踪达到统一基因下限的位置。
        tracked = raw.obs.n_genes_by_counts >= q["min_detected_genes"]
        candidate = tracked & (raw.obs.total_counts >= policy["min_candidate_UMI"])
        d = raw.obs.loc[tracked, ["barcode", "total_counts", "n_genes_by_counts", "pct_counts_mt"]].copy()
        d["sample"], d["study"] = row["sample"], study
        d["candidate_pass"] = candidate.loc[tracked]
        a = raw[candidate].copy()
        for key, value in row.items():
            a.obs[key] = value
        with h5py.File(path) as f:
            a.obs["h5_software_version"] = str(f.attrs.get("software_version", "not_available"))
            a.obs["h5_chemistry"] = str(f.attrs.get("chemistry_description", "not_available"))
            genomes = sorted(set(f["matrix/features/genome"].asstr()[:]))
        a.obs["genome_build"] = ";".join(genomes)
        a.obs["qc_annotation_release"] = str(reference["ensembl_release"])
        source_files.append(dict(sample=row["sample"], relative_path=row["relative_path"],
            source_url=row["source_url"], bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            h5_software_version=a.obs.h5_software_version.iloc[0], h5_genomes=";".join(genomes)))
        summary = dict(sample=row["sample"], barcode_positions=raw.n_obs,
            positions_ge_min_genes=int(tracked.sum()), candidate_cells=a.n_obs)
        print(f"{row['sample']}: {raw.n_obs:,} barcode positions -> {a.n_obs:,} RNA candidates", flush=True)
        del raw
        gc.collect()
        result = qc_sample(a, annotation, min_genes=q["min_detected_genes"],
            min_cells=q["min_cells_per_sample"], min_coding_mirna=q["min_detected_protein_coding_or_miRNA"],
            max_mt_pct=q["max_mt_pct_inclusive"], n_genes_upper_sd=q["n_genes_upper_sd"])
        core = result["core"]
        sc.pp.scrublet(core, expected_doublet_rate=policy["scrublet_expected_rate"],
            threshold=policy["scrublet_review_threshold"], n_prin_comps=30,
            use_approx_neighbors=True, random_state=cfg["random_seed"], verbose=False)
        if not np.isfinite(core.obs.doublet_score).all():
            raise ValueError("Scrublet returned nonfinite scores")
        full = result["full"]
        full.obs = core.obs.copy()
        parts.append(full)
        d = d.join(result["decisions"].drop(columns=d.columns.intersection(result["decisions"].columns)))
        d["basic_qc_pass"] = d.index.isin(full.obs_names)
        decisions.append(d)
        summary["basic_qc_cells"] = full.n_obs
        summary["basic_qc"] = result["summary"]
        summaries.append(summary)
        print(f"{row['sample']}: basic QC {full.n_obs:,}; genes {full.n_vars:,}", flush=True)
        del a, result, core
        gc.collect()

    full = ad.concat(parts, join="inner", merge="same")
    del parts
    missing_markers = add_evidence(full)
    a = full[:, full.var.is_core].copy()
    print(f"{study}: joint Scanpy clustering {a.shape}", flush=True)
    cluster(a)
    full.obs["leiden"] = a.obs.leiden
    full.obs["library_cluster"] = full.obs["sample"].astype(str) + ":" + full.obs.leiden.astype(str)
    apply_standard(full, policy)
    full.obs["qc_version"] = policy["preset"]
    full.obs["cell_type"] = np.where(full.obs.selected_ILC2, "ILC2", "review_or_excluded")
    full.obs["identity_confidence"] = np.where(full.obs.selected_ILC2, "high", "review")
    full.obs["identity_confidence_method"] = "computational_" + policy["preset"] + "_not_flow_validated"
    full.obs["training_tier"] = np.where(full.obs.selected_ILC2, policy["main_training_tier"], "excluded")
    full.obs["low_RNA_review"] = ~full.obs.additional_quality_pass
    full.obs["selection_reason"] = np.select([~full.obs.identity_positive, full.obs.alternative_program],
        ["weak_identity", "alternative_program"], default="retained")
    a.obs = full.obs.copy()
    selected = full[full.obs.selected_ILC2].copy()
    coverage = summarize_coverage(selected, selected.var.genecompass_token.to_numpy(),
        selected.var.has_median_reference.to_numpy())
    for s in summaries:
        sample = selected[selected.obs["sample"].eq(s["sample"])]
        s["selected_ILC2"] = sample.n_obs
        s["token_coverage"] = summarize_coverage(sample, sample.var.genecompass_token.to_numpy(),
            sample.var.has_median_reference.to_numpy())
    provenance = dict(dataset=study, version=policy["preset"], basic_qc=q, identity_policy=policy,
        qc_reference=reference, source_files=source_files, samples=summaries, token_coverage=coverage,
        input_genes=full.n_vars, core_genes=a.n_vars, selected_ILC2=selected.n_obs,
        missing_identity_markers=missing_markers, role="posttrain_candidate", split="unassigned",
        identity_interpretation="computational_high_confidence_not_experimentally_validated",
        candidate_interpretation="fixed_RNA_threshold_on_raw_droplets_not_statistical_cell_calling",
        identity_gene_space="all_input_GEX_counts", final_gene_space="all_input_GEX_genes",
        annotation_interpretation="QC annotation chosen by input ID/assembly coverage; author release unknown",
        software={p: importlib.metadata.version(p) for p in ["scanpy", "anndata", "numpy", "scipy", "scrublet"]})
    selected.uns.clear()
    selected.uns["metadata"] = {k: v for k, v in provenance.items() if k not in {"source_files", "samples"}}
    selected.uns["source_files"] = pd.DataFrame(source_files)
    selected.write_h5ad(data / f"{study}_ILC2.h5ad", compression="gzip")
    selected.obs.to_csv(data / f"{study}_ILC2_metadata.tsv.gz", sep="\t", index_label="cell_id")
    samples = metadata.set_index("sample")
    samples["retained_cells"] = selected.obs.groupby("sample", observed=True).size()
    for key in ["genome_build", "h5_software_version", "h5_chemistry", "qc_annotation_release", "qc_version"]:
        samples[key] = selected.obs.groupby("sample", observed=True)[key].first()
    samples.to_csv(data / f"{study}_sample_metadata.tsv", sep="\t")
    selected.var.to_csv(out / "gene_token_mapping.tsv.gz", sep="\t")
    selected.var[selected.var.gene_symbol.isin(KEY_GENES)].to_csv(out / "key_gene_token_mapping.tsv", sep="\t")
    pd.DataFrame(source_files).to_csv(ROOT / f"data/metadata/{study}_raw_file_manifest.tsv", sep="\t", index=False)
    d = pd.concat(decisions)
    d = d.join(full.obs.drop(columns=d.columns.intersection(full.obs.columns)))
    d["selected_ILC2"] = d.index.isin(selected.obs_names)
    d["final_reason"] = np.where(~d.candidate_pass, "below_candidate_RNA",
        np.where(~d.basic_qc_pass, d.qc_reason, d.selection_reason))
    d.to_csv(out / "cell_decisions.tsv.gz", sep="\t")
    table = pd.DataFrame([{k: s[k] for k in ["sample", "barcode_positions", "positions_ge_min_genes",
        "candidate_cells", "basic_qc_cells", "selected_ILC2"]} for s in summaries])
    table.to_csv(out / "sample_summary.tsv", sep="\t", index=False)
    selected.obs.groupby("sample", observed=True)[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].median().to_csv(
        out / "retained_qc_medians.tsv", sep="\t")
    a.obs.groupby("library_cluster", observed=True).agg(n_cells=("sample", "size"),
        retained=("selected_ILC2", "sum"), alternative_fraction=("alternative_program", "mean")).to_csv(
        out / "cluster_review.tsv", sep="\t")
    sc.tl.umap(a, min_dist=.5, random_state=cfg["random_seed"])
    a.uns["qc_provenance"] = selected.uns["metadata"]
    a.write_h5ad(out / f"{study}_qc_annotated.h5ad", compression="gzip")
    sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    markers = sc.get.rank_genes_groups_df(a, group=None)
    markers["gene_symbol"] = markers.names.map(a.var.gene_symbol)
    markers.to_csv(out / "cluster_markers.tsv", sep="\t", index=False)

    # 仅三类基础图；QC图以RNA候选为分母，不把空液滴当输入细胞。
    sc.settings.set_figure_params(dpi=110, frameon=False)
    candidate_qc = d.loc[d.candidate_pass]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(axes, ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        ["UMI per candidate", "Detected genes", "Mitochondrial UMI (%)"]):
        bins = np.histogram_bin_edges(candidate_qc[key], bins=50)
        ax.hist(candidate_qc[key], bins=bins, alpha=.5, label="RNA candidates", color="#3B75AF")
        ax.hist(selected.obs[key], bins=bins, alpha=.6, label="ILC2 retained", color="#DB8B35")
        ax.set(xlabel=label, ylabel="Cells")
        ax.legend(fontsize=8)
    fig.suptitle(f"{study} controls | {len(candidate_qc):,} RNA candidates -> {selected.n_obs:,} ILC2")
    fig.tight_layout()
    fig.savefig(out / "qc.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    a.obs["cell_type"] = pd.Categorical(a.obs.cell_type, categories=["ILC2", "review_or_excluded"])
    a.uns["cell_type_colors"] = ["#3B75AF", "#DB8B35"]
    sc.pl.umap(a, color=["sample", "leiden", "cell_type"], ncols=3, size=5, show=False)
    plt.savefig(out / "umap.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    panel = {"ILC2": ["Gata3", "Rora", "Il7r", "Id2", "Il1rl1", "Il17rb"],
        "Effector / repair": ["Il5", "Il13", "Areg"],
        "Alternative": ["Cd3d", "Cd3e", "Trac", "Ncr1", "Klrd1", "Rorc", "Il23r", "Lyz2", "Epcam"]}
    panel = {k: [g for g in v if a.var.gene_symbol.eq(g).any()] for k, v in panel.items()}
    dp = sc.pl.dotplot(a, panel, groupby="leiden", gene_symbols="gene_symbol", use_raw=False,
        return_fig=True, cmap="Blues", figsize=(12, 4.5))
    dp.savefig(out / "marker_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    provenance["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "src/qc_utils.py", ROOT / "src/ilc2_qc.py",
                  ROOT / "configs/1QC.json", ROOT / reference["ensembl_annotation"]]}
    (out / "summary.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    limits = ("三个GEO文库按肠段命名，不视为三个独立动物；论文肠段合并与GEO文库的对应未厘清。"
        if study == "GSE301976" else "GEO有两个对照生物重复文库；论文有四只未感染动物，逐文库混池对应未知。")
    report = [f"# {study} 稳态control QC", "", "```tsv\n" + table.to_csv(index=False, sep="\t") + "```", "",
        f"统一预设{policy['preset']}，所有数值阈值未加严。候选{len(candidate_qc):,} → 基础QC{full.n_obs:,} → 计算支持ILC2 **{selected.n_obs:,}**。",
        f"最终保留全部{selected.n_vars:,}个输入RNA基因，X和layers['counts']均为原始整数UMI；分析core有{a.n_vars:,}基因。",
        "raw barcode位置不是细胞。先按统一genes/UMI门槛确定候选；未进行EmptyDrops统计cell calling。去留表追踪达到基因下限的位置。",
        "基础QC和Scrublet按GSM分别执行。身份在全基因counts调用apply_standard；低RNA、双细胞高分、状态及混合群不单独删除。",
        "高置信度是计算身份支持，不等于实验验证；残留ambient RNA/同型双细胞风险仍在。",
        f"QC参考：{reference['ensembl_annotation']}，全ID覆盖；作者注释release未知。注释选择原因：{reference['reason']}",
        "H5内软件/参考标签与GEO报告分别保留，不以覆盖率反推作者版本。",
        f"GeneCompass精确ID匹配：{coverage['genes_in_vocab']:,}基因，{coverage['umi_fraction_in_vocab']:.2%} UMI；有median覆盖{coverage['umi_fraction_with_median']:.2%} UMI。",
        "沿用既有median代理，不在本研究重估；词表外基因仍保留在最终counts。",
        f"输入缺失身份面板feature：{', '.join(missing_markers) or '无'}。", limits,
        "用途已确认为posttrain_candidate；split=unassigned，未重新划分、编码或训练。处理和KO样本不下载、不作为下游测试。",
        "最终对象和两类metadata见data/processed/对应_qc目录；qc_annotated对象保留全部基础QC细胞用于审查，X为log1p、counts为原始core计数。",
        "图仅qc.png、umap.png、marker_dotplot.png。逐细胞去留、逐样本统计、cluster marker和token表均在本目录。",
        f"复现：python src/qc_gse301975_301976.py --study {study}。"]
    (out / "QC_report.md").write_text("\n\n".join(report) + "\n")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(coverage, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {selected.shape} to {data}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=CONTROL_SAMPLES)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        main(args.study)
