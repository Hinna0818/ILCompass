"""将已保存的联合 UMAP 按真实 Leiden 标签着色，不重算坐标或聚类。

运行：python main/plot_saved_gse117567_clusters.py
"""
from plot_gse117567_umap import OUT
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    a = sc.read_h5ad(OUT / "GSE117567_umap_hvg.h5ad")
    assert a.n_obs == 25010 and a.obs.umap_leiden.notna().all()
    assert a.obs.umap_leiden.nunique() == 7
    coords = pd.read_csv(OUT / "umap_coordinates.tsv.gz", sep="\t", index_col=0)
    assert coords.index.equals(a.obs_names)
    assert np.allclose(coords[["UMAP1", "UMAP2"]], a.obsm["X_umap"])
    assert np.array_equal(coords.umap_leiden.astype(str), a.obs.umap_leiden.astype(str))
    sc.settings.set_figure_params(dpi=120, frameon=False)
    sc.pl.umap(a, color="umap_leiden", legend_loc="on data", legend_fontsize=12,
               legend_fontoutline=2, palette="tab10", size=5,
               title="GSE117567 | Leiden clusters (resolution=0.6) | n=25,010",
               show=False)
    plt.savefig(OUT / "GSE117567_umap_clusters.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / "GSE117567_umap_clusters.pdf", bbox_inches="tight")
    plt.close("all")
    table = pd.crosstab(a.obs.umap_leiden, a.obs.tissue)
    table["total"] = table.sum(axis=1)
    assert table.total.sum() == a.n_obs
    table.to_csv(OUT / "cluster_tissue_counts.tsv", sep="\t")
    print(table.to_string())


if __name__ == "__main__":
    main()
