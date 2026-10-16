"""阶段 1：按样本执行 GeneCompass paper-based 基础 QC。

--basic 运行新工具库；基础 QC 不自动判定 ILC2 身份。
--legacy-gse117567 调用既有初步分析（旧阈值 500 genes/1000 UMI/mt<15%）。
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_basic(config_path: Path) -> None:
    """根据样本账本逐个运行、保存及重读核验；不把五个组织混成一个 QC 样本。"""
    import anndata as ad
    import numpy as np
    import pandas as pd
    from qc_utils import read_10x_sample, read_gene_annotation, qc_sample

    config = json.loads(config_path.read_text())
    reference = config["core_genes"]["ensembl_annotation"]
    if not reference:
        raise ValueError("配置缺少经过核验的 Ensembl GTF 路径。")
    reference_path = ROOT / reference
    annotation = read_gene_annotation(reference_path)
    core_path = config["core_genes"].get("core_gene_list")
    whitelist = None
    if core_path:
        table = pd.read_csv(ROOT / core_path, sep="\t")
        whitelist = table["gene_id"].tolist()
    manifest = pd.read_csv(ROOT / config["sample_manifest"], sep="\t", dtype=str)
    if manifest["sample"].duplicated().any():
        raise ValueError("样本清单有重复 sample，请先明确文库和生物样本关系。")
    if not manifest["species"].eq(config["core_genes"]["species"]).all():
        raise ValueError("样本物种与该配置的 Ensembl 参考不一致。")
    qc = config["basic_qc"]
    if qc["sd_ddof"] != 0 or qc["sd_reference"] != "sample_nonempty_cells_before_basic_qc" or qc["metrics_gene_space"] != "original_all_gene_counts":
        raise ValueError("当前工具只实现配置中约定的全基因、样本非空细胞总体 SD。")
    rows = []
    checks = []
    # 参考解析一次；矩阵按样本依次载入/释放，减少内存峰值。
    for row in manifest.to_dict("records"):
        print(f"Reading and QC: {row['sample']} ({row['tissue']})", flush=True)
        a = read_10x_sample(ROOT / row["input_path"], row["file_prefix"], sample_id=row["sample"])
        for field in ["study", "species", "tissue", "condition", "biological_rep", "animal_or_pool"]:
            a.obs[field] = row[field]
        result = qc_sample(a, annotation,
            min_genes=qc["min_detected_genes"], min_cells=qc["min_cells_per_sample"],
            min_coding_mirna=qc["min_detected_protein_coding_or_miRNA"],
            max_mt_pct=qc["max_mt_pct_inclusive"], n_genes_upper_sd=qc["n_genes_upper_sd"],
            core_gene_ids=whitelist)
        destination = ROOT / config["output_root"] / f"{row['study']}_basic_qc" / row["sample"]
        destination.mkdir(parents=True, exist_ok=True)
        full, core = result["full"], result["core"]
        # 过滤前全体细胞仍在原始文件；去留清单记录所有输入细胞。
        result["decisions"].to_csv(destination / "cell_decisions.tsv.gz", sep="\t")
        full.write_h5ad(destination / "full_counts_qc.h5ad", compression="gzip")
        core.write_h5ad(destination / "core_counts_qc.h5ad", compression="gzip")
        full.var.to_csv(destination / "gene_annotation.tsv", sep="\t")
        summary = dict(result["summary"], tissue=row["tissue"], study=row["study"])
        rows.append(summary)
        # 核验真实 counts 与输入逐细胞一致，而不仅是核对矩阵形状。
        original_rows = a.obs_names.get_indexer(full.obs_names)
        assert (a.X[original_rows] != full.layers["counts"]).nnz == 0
        assert np.array_equal(full.obs_names, core.obs_names)
        assert core.var.biotype_group.isin(["protein_coding", "lncRNA", "miRNA"]).all()
        assert len(result["decisions"]) == a.n_obs
        assert full.obs.qc_pass.all()
        for name, expected in [("full_counts_qc.h5ad", full), ("core_counts_qc.h5ad", core)]:
            saved = ad.read_h5ad(destination / name, backed="r")
            assert saved.shape == expected.shape and saved.obs_names.equals(expected.obs_names)
            saved.file.close()
        checks.append({"sample": row["sample"], "counts_equal_input": True, "core_biotypes_valid": True, "saved_files_reloaded": True})
        print(f"  {a.n_obs} -> {full.n_obs} cells; {a.n_vars} -> {core.n_vars} core genes", flush=True)
        del a, result, full, core
        gc.collect()
    results = ROOT / config["report_root"]
    results.mkdir(parents=True, exist_ok=True)
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(results / "sample_summary.tsv", sep="\t", index=False)
    annotation.to_csv(results / "ensembl_gene_annotation.tsv", sep="\t")
    annotation[annotation.biotype_group.isin(["protein_coding", "lncRNA", "miRNA"])].to_csv(results / "ensembl_biotype_core_genes.tsv", sep="\t")
    digest = hashlib.sha256()
    with reference_path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    provenance = {"reference_path": reference, "reference_sha256": digest.hexdigest(), "reference_url": config["core_genes"]["annotation_url"], "ensembl_release": config["core_genes"]["ensembl_release"], "genome_build": config["core_genes"]["genome_build"], "biotype_definition_url": "https://www.gencodegenes.org/pages/biotypes.html", "config": config, "input_file_manifest": "data/metadata/raw_file_manifest.tsv", "identity_selection": "not_performed", "checkpoint_vocab_intersection": "not_performed"}
    (results / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (results / "validation.json").write_text(json.dumps({"passed": True, "samples": checks}, indent=2))
    print(summary_table[["sample", "tissue", "n_input_cells", "n_pass_cells", "n_removed_cells", "n_core_genes"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--basic", action="store_true", help="执行工具库基础 QC，不自动筛选 ILC2")
    mode.add_argument("--legacy-gse117567", action="store_true", help="使用已有初步 QC；其余参数传递给旧脚本")
    parser.add_argument("--config", default=str(ROOT / "configs/1QC.json"))
    args, remaining = parser.parse_known_args()
    if args.basic:
        if remaining:
            parser.error("基础 QC 未识别的参数：" + " ".join(remaining))
        run_basic(Path(args.config))
    else:
        script = ROOT / "main/qc_gse117567.py"
        sys.argv = [str(script), *remaining]
        runpy.run_path(str(script), run_name="__main__")
