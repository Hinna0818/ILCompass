"""GSE153748最终QC：原始tar→统一balanced_v1→外周counts与独立骨髓背景。

运行：conda activate ilc2_model; python src/qc_gse153748.py
规则唯一来源configs/1QC.json；不依赖旧中间对象，不在脚本内清理其他结果。
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/ilc2_model_numba")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ilc2_model_mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import tarfile
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from qc_utils import _raw_counts, read_gene_annotation, qc_sample
from ilc2_qc import PANELS, ACINAR, add_evidence, marker_values, apply_standard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/GSE153748_qc"
DATA = ROOT / "data/processed/GSE153748_qc"


def read_candidates(row: pd.Series, min_genes: int, min_umi: int) -> tuple:
    """保留宽松候选；液滴位点不当作已确认细胞。"""
    with tarfile.open(ROOT / row.input_file) as archive:
        genes = pd.read_csv(archive.extractfile("genes.tsv"), sep="\t", header=None,
                            names=["gene_id", "gene_symbol"], dtype=str)
        barcodes = pd.read_csv(archive.extractfile("barcodes.tsv"), header=None, dtype=str)[0]
        with threadpool_limits(limits=4):
            x = _raw_counts(mmread(archive.extractfile("matrix.mtx")).T)
    if x.shape != (len(barcodes), len(genes)) or not genes.gene_id.is_unique or not barcodes.is_unique:
        raise ValueError("输入矩阵维度或gene/barcode唯一性不符")
    umi = np.asarray(x.sum(axis=1)).ravel()
    detected = np.diff(x.indptr)
    candidate = (detected >= min_genes) & (umi >= min_umi)
    tracked = detected >= min_genes
    decisions = pd.DataFrame({"barcode": barcodes[tracked].to_numpy(), "sample_id": row.sample_id,
                              "tissue": row.tissue, "total_counts": umi[tracked],
                              "n_genes_by_counts": detected[tracked], "candidate_pass": candidate[tracked]},
                             index=row.sample_id + ":" + barcodes[tracked].to_numpy())
    obs = pd.DataFrame({"barcode": barcodes[candidate].to_numpy()},
                       index=row.sample_id + ":" + barcodes[candidate].to_numpy())
    for field, value in row.items():
        obs[field] = value
    obs["sample"] = row.sample_id
    obs["study"] = "GSE153748"
    obs["species"] = "Mouse"
    obs["genotype"] = "WT"
    obs["strain"] = "C57BL/6"
    obs["condition"] = "steady_state_no_exogenous_stimulation"
    obs["animal_id"] = "not_available"
    obs["biological_rep"] = "pooled_library_no_replicates"
    obs["pool_scope"] = "shared_tissue_cohort_per_batch"
    obs["sequencing_rounds"] = 2
    obs["sort_gate"] = "Lin-Thy1lowCD45+KLRG1high" if row.tissue == "SI" else "Lin-ST2+CD25+"
    obs["library_protocol"] = "10x_3prime_v2"
    obs["genome_build"] = "GRCm38_mm10"
    obs["split"] = "unassigned"
    obs.index.name = decisions.index.name = "cell_id"
    a = ad.AnnData(x[candidate], obs=obs, var=genes.set_index("gene_id"))
    # shortcut: raw液滴使用固定RNA候选门槛；需确认低RNA细胞时升级为统计cell calling。
    return a, decisions, {"barcode_positions": len(barcodes), "positions_ge200_genes": int(tracked.sum()),
                           "candidate_cells": a.n_obs, "candidate_UMI_fail": int((tracked & ~candidate).sum())}


def author_programs() -> dict[str, list[str]]:
    table = pd.read_csv(ROOT / "data/metadata/GSE153748_author_signatures.tsv", sep="\t", dtype=str)
    def genes(tissue: str, cluster: str) -> set[str]:
        return set(table.loc[(table.tissue == tissue) & (table.author_cluster == cluster) & (table.comparison == "all"), "gene"])
    stress = genes("LI", "5") | genes("SI", "1") | genes("SI", "6")
    return {"effector": sorted(genes("LI", "1") & genes("SI", "3")),
            "stress": sorted(g for g in stress if g.startswith(("Hsp", "Dnaj"))),
            "interferon": sorted(genes("BM", "6") & genes("LI", "7"))}


def annotate_states(a: ad.AnnData, programs: dict[str, list[str]]) -> None:
    """队列内相对高signature，不参与过滤，不能作为功能验证。"""
    cutoffs = {}
    for name, genes in programs.items():
        ids = a.var_names[a.var.gene_symbol.isin(genes)].tolist()
        if len(ids) < 3:
            raise ValueError(f"{name} signature映射不足")
        sc.tl.score_genes(a, ids, score_name=f"state_{name}_score", use_raw=False, random_state=0)
        detected = (marker_values(a, genes, "counts") > 0).sum(axis=1)
        a.obs[f"state_{name}_n_detected"] = detected
        cutoffs[name] = max(0.0, float(a.obs[f"state_{name}_score"].quantile(.75)))
        a.obs[f"state_{name}_flag"] = (a.obs[f"state_{name}_score"] > cutoffs[name]) & (detected >= 3)
    a.obs["state_effector_flag"] &= (a.obs.counts_Il5 > 0) | (a.obs.counts_Il13 > 0)
    flags = a.obs[[f"state_{name}_flag" for name in programs]]
    a.obs["state_signature_labels"] = [";".join(name+"_signature_high" for name, flag in zip(programs, values) if flag) or "no_strong_signature" for values in flags.itertuples(index=False, name=None)]
    a.uns["state_score_cutoffs"] = cutoffs


def cluster(a: ad.AnnData) -> None:
    sc.pp.normalize_total(a, target_sum=10000)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2500)
    sc.pp.pca(a, n_comps=30, mask_var="highly_variable", random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=0)
    sc.tl.leiden(a, resolution=.6, flavor="igraph", directed=False, n_iterations=2, random_state=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "run.log", mode="w"), logging.StreamHandler()])
    config = json.loads((ROOT / "configs/1QC.json").read_text())
    q, policy = config["basic_qc"], config["identity"]
    annotation = read_gene_annotation(ROOT / config["core_genes"]["ensembl_annotation"])
    metadata = pd.read_csv(ROOT / "data/metadata/GSE153748_sample_metadata.tsv", sep="\t")
    programs = author_programs()
    pd.DataFrame([(n, g) for n, genes in programs.items() for g in genes], columns=["program", "gene"]).to_csv(OUT / "state_signature_genes.tsv", sep="\t", index=False)
    parts, summaries, all_decisions, missing_markers, marker_tables = [], [], [], {}, []
    fig, axes = plt.subplots(1, len(metadata), figsize=(17, 3))
    for i, (_, row) in enumerate(metadata.iterrows()):
        logging.info("QC %s (%s)", row.sample_id, row.tissue)
        candidate, decisions, summary = read_candidates(row, q["min_detected_genes"], policy["min_candidate_UMI"])
        result = qc_sample(candidate, annotation, min_genes=q["min_detected_genes"],
                           min_cells=q["min_cells_per_sample"], min_coding_mirna=q["min_detected_protein_coding_or_miRNA"],
                           max_mt_pct=q["max_mt_pct_inclusive"], n_genes_upper_sd=q["n_genes_upper_sd"])
        a = result["core"]
        assert (a.layers["counts"] != candidate[a.obs_names, a.var_names].X).nnz == 0
        qc_decisions = result["decisions"].drop(columns=list(decisions.columns.intersection(result["decisions"].columns)))
        decisions = decisions.join(qc_decisions)
        summary.update(result["summary"])
        del candidate, result
        gc.collect()
        sc.pp.scrublet(a, expected_doublet_rate=policy["scrublet_expected_rate"],
                       threshold=policy["scrublet_review_threshold"], n_prin_comps=30,
                       use_approx_neighbors=True, random_state=0, verbose=False)
        if not np.isfinite(a.obs.doublet_score).all():
            raise ValueError("Scrublet分数非有限，不能默认singlet")
        missing_markers[row.sample_id] = add_evidence(a)
        missing_markers[row.sample_id] += [g for g in ACINAR if g not in set(a.var.gene_symbol)]
        cluster(a)
        a.obs["library_cluster"] = row.sample_id + ":" + a.obs.leiden.astype(str)
        apply_standard(a, policy)
        a.obs["scrublet_threshold"] = policy["scrublet_review_threshold"]
        a.obs["doublet_interpretation"] = "score_tail_flag_only_not_validated_doublet"
        a.obs["low_RNA_review"] = (a.obs.n_genes_by_counts < 500) | (a.obs.total_counts < 1000)
        failures = pd.DataFrame({"weak_identity": ~a.obs.identity_positive,
                                 "alternative_program": a.obs.alternative_program})
        a.obs["selection_reason"] = [";".join(n for n, f in zip(failures, values) if f) or "retained" for values in failures.itertuples(index=False, name=None)]
        a.obs["cell_type"] = np.where(a.obs.selected_ILC2, "ILC2_candidate", "review_or_excluded")
        a.obs["identity_confidence"] = np.where(a.obs.selected_ILC2, "supported", "review")
        a.obs["identity_confidence_method"] = policy["preset"] + "_computational_not_flow_validated"
        evidence = a.obs.drop(columns=list(decisions.columns.intersection(a.obs.columns)))
        decisions = decisions.join(evidence)
        decisions["selected_ILC2"] = decisions.selected_ILC2.eq(True)
        decisions["basic_qc_pass"] = decisions.index.isin(a.obs_names)
        decisions["final_reason"] = np.where(~decisions.candidate_pass, "below_candidate_UMI",
                                             np.where(~decisions.basic_qc_pass, decisions.qc_reason,
                                                      decisions.selection_reason))
        all_decisions.append(decisions)
        review = a.obs.groupby("library_cluster", observed=True).agg(
            n_cells=("sample", "size"), identity_fraction=("identity_positive", "mean"),
            alternative_fraction=("alternative_program", "mean"), doublet_flags=("doublet_review", "sum"),
            retained=("selected_ILC2", "sum"))
        review.to_csv(OUT / f"{row.sample_id}_cluster_review.tsv", sep="\t")
        sc.tl.rank_genes_groups(a, "leiden", method="t-test_overestim_var", n_genes=20, use_raw=False)
        markers = sc.get.rank_genes_groups_df(a, group=None)
        markers["gene_symbol"] = markers.names.map(a.var.gene_symbol)
        markers["sample_id"] = row.sample_id
        marker_tables.append(markers)
        axes[i].hist(a.obs.doublet_score, bins=40, density=True, alpha=.6, label="Observed")
        axes[i].hist(a.uns["scrublet"]["doublet_scores_sim"], bins=40, density=True, alpha=.4, label="Simulated")
        axes[i].axvline(policy["scrublet_review_threshold"], color="red", ls="--")
        axes[i].set(title=row.tissue, xlabel="Scrublet score (review only)")
        summary.update(sample_id=row.sample_id, tissue=row.tissue, identity_selected=int(a.obs.selected_ILC2.sum()),
                       doublet_flags=int(a.obs.doublet_review.sum()), retained_doublet_flags=int((a.obs.doublet_review & a.obs.selected_ILC2).sum()),
                       retained_low_RNA=int((a.obs.low_RNA_review & a.obs.selected_ILC2).sum()),
                       retained_mixed_cluster=int((a.obs.mixed_cluster_review & a.obs.selected_ILC2).sum()),
                       weak_identity=int((~a.obs.identity_positive).sum()), alternative_program=int(a.obs.alternative_program.sum()),
                       train_core_cells=int(a.obs.selected_ILC2.sum()) if row.tissue != "BM" else 0)
        summaries.append(summary)
        selected = a[a.obs.selected_ILC2].copy()
        selected.X = selected.layers["counts"].copy()
        selected.uns.clear()
        selected.obsm.clear()
        selected.obsp.clear()
        selected.varm.clear()
        selected.obs["training_tier"] = "BM_background_review" if row.tissue == "BM" else "peripheral_balanced"
        selected.obs["data_role"] = "background_candidate" if row.tissue == "BM" else "posttrain_candidate"
        parts.append(selected)
        logging.info("%s: candidates %s -> basic %s -> ILC2 %s", row.tissue, summary["candidate_cells"], a.n_obs, selected.n_obs)
        del a
        gc.collect()
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "doublet_scores.png", dpi=150)
    plt.close(fig)
    counts = ad.concat(parts, join="inner", merge="same")
    del parts
    assert counts.n_vars == 27998 and counts.obs_names.is_unique
    bm = counts[counts.obs.tissue.eq("BM")].copy()
    core = counts[counts.obs.tissue.ne("BM")].copy()
    del counts
    for a in [core, bm]:
        if a is core:
            cluster(a)
        else:
            sc.pp.normalize_total(a, target_sum=10000)
            sc.pp.log1p(a)
        annotate_states(a, programs)
        a.uns["metadata"] = {"dataset_id": "GSE153748", "version": policy["preset"],
                             "basic_qc": q, "identity_policy": policy, "samples": metadata,
                             "identity_panels": PANELS, "acinar_panel": ACINAR,
                             "missing_markers_json": json.dumps(missing_markers),
                             "state_signatures": programs, "batch_correction": False,
                             "state_interpretation": "cohort_relative_high_nonexclusive_not_functional_validation",
                             "identity_interpretation": "computational_supported_not_independently_validated",
                             "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153748"}
    sc.tl.umap(core, random_state=0)
    sc.pl.umap(core, color=["tissue", "leiden", "doublet_review", "low_RNA_review"], size=4, show=False, ncols=2)
    plt.savefig(OUT / "GSE153748_umap.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    sc.pl.umap(core, color=[f"state_{name}_score" for name in programs], cmap="viridis", size=4, show=False)
    plt.savefig(OUT / "GSE153748_states.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    target = ROOT / "data/train_data/GSE153748.h5ad"
    target.parent.mkdir(parents=True, exist_ok=True)
    for a, path in [(core, target), (bm, DATA / "GSE153748_BM_ILC2.h5ad")]:
        a.X = a.layers["counts"].copy()
        a.uns.pop("log1p", None)
        assert np.issubdtype(a.X.dtype, np.integer) and (a.X != a.layers["counts"]).nnz == 0
        assert (a.X.sum(axis=1).A1 == a.obs.total_counts.to_numpy()).all()
        assert a.obs.selected_ILC2.all() and not a.obs.alternative_program.any()
        temporary = path.with_suffix(".pending.h5ad")
        a.write_h5ad(temporary, compression="gzip")
        saved = sc.read_h5ad(temporary)
        assert saved.obs_names.equals(a.obs_names) and saved.var_names.equals(a.var_names)
        assert (saved.X != a.X).nnz == 0 and (saved.layers["counts"] != a.X).nnz == 0
        if "X_umap" in saved.obsm:
            assert np.isfinite(saved.obsm["X_umap"]).all()
        del saved
        temporary.replace(path)
    pd.DataFrame(summaries).to_csv(OUT / "sample_summary.tsv", sep="\t", index=False)
    pd.concat(all_decisions).to_csv(OUT / "cell_decisions.tsv.gz", sep="\t")
    pd.concat(marker_tables).to_csv(OUT / "cluster_markers.tsv", sep="\t", index=False)
    core.obs.groupby("tissue", observed=True)[[f"state_{n}_flag" for n in programs]].agg(["sum", "mean"]).to_csv(OUT / "state_summary.tsv", sep="\t")
    files = [Path(__file__), ROOT / "src/ilc2_qc.py", ROOT / "src/qc_utils.py", ROOT / "configs/1QC.json",
             ROOT / config["core_genes"]["ensembl_annotation"], ROOT / "data/metadata/GSE153748_author_signatures.tsv",
             ROOT / "data/GSE153748_RAW/GSE153748_Tissue_ILC2_sub-cluster_signature_genes.xlsx",
             ROOT / "data/metadata/GSE153748_sample_metadata.tsv", *[ROOT / p for p in metadata.input_file]]
    provenance = {"version": policy["preset"], "config": config, "sha256": {},
                  "software": {n: importlib.metadata.version(n) for n in ["scanpy", "anndata", "numpy", "scipy", "pandas"]},
                  "n_train_cells": core.n_obs, "n_BM_background_cells": bm.n_obs, "n_genes": core.n_vars,
                  "counts_preserved_against_raw": True, "output_reloaded_and_counts_compared": True,
                  "batch_correction": False, "seed": 0,
                  "calling": "genes>=200 & UMI>=500; heuristic_not_EmptyDrops",
                  "decisions_scope": "all raw barcode positions with >=200 detected genes"}
    for path in files:
        with path.open("rb") as handle:
            provenance["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(handle.read()).hexdigest()
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "validation.json").write_text(json.dumps({"passed": True, "unique_ids": True,
        "integer_counts_equal_raw_source": True, "saved_X_equals_counts": True,
        "saved_objects_reloaded_and_compared": True, "all_selected_pass_rules": True,
        "states_and_doublet_flags_not_used_as_veto": True, "no_BM_in_peripheral_object": bool(core.obs.tissue.ne("BM").all())}, indent=2) + "\n")
    logging.info("Saved %s: %s; BM background %s cells", target, core.shape, bm.n_obs)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
