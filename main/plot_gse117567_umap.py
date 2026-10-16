"""从筛选后的 GSE117567 原始 counts 计算联合 UMAP；不修改训练文件。

运行：conda activate ilc2_model; python main/plot_gse117567_umap.py
结果统一写入 results/GSE117567_train_umap/。
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/ilc2_model_numba")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ilc2_model_mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import hashlib
import json
from pathlib import Path
import importlib.metadata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/train_data/GSE117567.h5ad"
OUT = ROOT / "results/GSE117567_train_umap"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(SOURCE)
    a = sc.read_h5ad(SOURCE)
    assert a.shape == (25010, 27998) and a.obs_names.is_unique
    assert (a.X != a.layers["counts"]).nnz == 0
    markers = ["Gata3", "Rora", "Il7r", "Il1rl1", "Il13", "Areg"]
    for gene in markers:
        # 已保存的 marker UMI 合并同名 symbol；颜色为每万 UMI 归一化后 log1p。
        a.obs["expr_" + gene] = np.log1p(a.obs["counts_" + gene] * 1e4 / a.obs.total_counts)
    a.obs["tissue"] = pd.Categorical(a.obs.tissue, categories=["Lung", "Skin", "Gut", "Fat"])
    params = dict(target_sum=10000, hvg_flavor="seurat", n_top_genes=2500,
                  n_pcs=30, n_neighbors=15, min_dist=0.5, leiden_resolution=0.6,
                  seed=0, batch_correction="none_tissue_and_library_confounded",
                  input=str(SOURCE.relative_to(ROOT)), input_sha256=source_hash,
                  input_cells=a.n_obs, input_genes=a.n_vars)
    print("Normalizing and selecting HVGs", flush=True)
    sc.pp.normalize_total(a, target_sum=params["target_sum"])
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=2500)
    a.var.to_csv(OUT / "gene_hvg_selection.tsv", sep="\t")
    # 可视化对象只存 HVG，避免重复存一份完整大矩阵；训练对象仍保留全部基因。
    a = a[:, a.var.highly_variable].copy()
    sc.pp.pca(a, n_comps=30, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=0)
    print("Computing joint UMAP and exploratory Leiden clusters", flush=True)
    sc.tl.umap(a, min_dist=0.5, random_state=0)
    sc.tl.leiden(a, resolution=0.6, flavor="igraph", directed=False,
                 n_iterations=2, random_state=0, key_added="umap_leiden")
    assert np.isfinite(a.obsm["X_umap"]).all() and a.obsm["X_umap"].shape == (25010, 2)
    sc.settings.set_figure_params(dpi=120, dpi_save=200, frameon=False)
    sc.pl.umap(a, color="tissue", title="GSE117567 | QC + ILC2 selection | n=25,010",
               palette=["#3B75AF", "#DB8B35", "#499870", "#B76D9F"],
               size=5, show=False)
    plt.savefig(OUT / "GSE117567_umap_tissue.png", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "GSE117567_umap_tissue.pdf", bbox_inches="tight")
    plt.close("all")
    sc.pl.umap(a, color=["tissue", "umap_leiden", "n_genes_by_counts", "pct_counts_mt"],
               ncols=2, size=4, show=False)
    plt.savefig(OUT / "GSE117567_umap_overview.png", bbox_inches="tight", dpi=160)
    plt.close("all")
    sc.pl.umap(a, color=["expr_" + gene for gene in markers], title=markers,
               ncols=3, size=4, cmap="viridis", vmin=0, vmax="p99", show=False)
    plt.savefig(OUT / "GSE117567_umap_markers.png", bbox_inches="tight", dpi=160)
    plt.close("all")
    a.uns["visualization_provenance"] = params
    a.write_h5ad(OUT / "GSE117567_umap_hvg.h5ad", compression="gzip")
    coords = pd.DataFrame(a.obsm["X_umap"], index=a.obs_names, columns=["UMAP1", "UMAP2"])
    coords.join(a.obs[["sample_id", "tissue", "umap_leiden"]]).to_csv(OUT / "umap_coordinates.tsv.gz", sep="\t")
    assert file_hash(SOURCE) == source_hash, "训练 counts 文件不得改动"
    saved = sc.read_h5ad(OUT / "GSE117567_umap_hvg.h5ad", backed="r")
    assert saved.obs_names.equals(a.obs_names) and np.array_equal(saved.obsm["X_umap"], a.obsm["X_umap"])
    saved.file.close()
    params["scanpy_version"] = importlib.metadata.version("scanpy")
    params["n_exploratory_clusters"] = a.obs.umap_leiden.nunique()
    params["training_source_unchanged"] = True
    (OUT / "parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {OUT}; {a.shape}, {params['n_exploratory_clusters']} exploratory clusters", flush=True)


if __name__ == "__main__":
    main()
