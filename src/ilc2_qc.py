"""ILC2 多基因身份证据；数值门槛是项目规则，不是通用验证分类器。"""
import anndata as ad
import numpy as np
import pandas as pd

ACINAR = ["Cela2a", "Ctrb1", "Clps", "Prss2", "Cela1", "Cela3b", "Pnlip", "Cpa1", "Cpb1", "Reg1", "Cel", "Sycn"]

def marker_values(a, genes, layer=None):
    # 少数 symbol 重复时求和，但 h5ad 始终保留唯一 Ensembl ID。
    x = a.X if layer is None else a.layers[layer]
    return np.column_stack([np.asarray(x[:, a.var.gene_symbol.eq(g).values].sum(axis=1)).ravel() for g in genes])


PANELS = {
    "ILC2_identity": ["Gata3", "Rora", "Il7r", "Id2", "Thy1", "Bcl11b", "Gfi1"],
    "ILC2_support": ["Icos", "Klrg1", "Il1rl1", "Il17rb", "Il2ra", "Il5", "Il13", "Areg", "Arg1"],
    "ILC3": ["Rorc", "Il23r", "Il22", "Il17a", "Il17f", "Ccr6", "Kit", "Ahr"],
    "T": ["Cd3d", "Cd3e", "Cd3g", "Trac", "Trdc"],
    "NK_ILC1": ["Ncr1", "Klrd1", "Eomes", "Prf1", "Gzmb", "Tbx21", "Nkg7", "Xcl1"],
    "B": ["Cd79a", "Cd79b", "Ms4a1", "Cd19"],
    "Myeloid": ["Lyz2", "Adgre1", "Csf1r", "Itgam", "Lst1"],
    "Endothelial": ["Pecam1", "Cdh5", "Cldn5", "Kdr", "Esam"],
    "Stromal": ["Col1a1", "Col1a2", "Dcn", "Lum", "Pdgfra"],
    "Epithelial": ["Epcam", "Krt8", "Krt18", "Krt19"],
    "Erythroid": ["Alas2", "Bpgm", "Gypa", "Slc4a1", "Klf1"],
    "Precursor": ["Tcf7", "Zbtb16", "Kit", "Cd7", "Il18r1"],
    "State": ["Mki67", "Top2a", "Fos", "Jun", "Isg15", "Ifit1"],
}


def add_evidence(a: ad.AnnData) -> list[str]:
    """保留原始 UMI 的多基因证据；缺失 marker 单独记录，不解释成生物学阴性。"""
    all_genes = list(dict.fromkeys(g for panel in PANELS.values() for g in panel))
    counts = marker_values(a, all_genes, "counts")
    c = {g: counts[:, j] for j, g in enumerate(all_genes)}
    columns = {}
    for g in all_genes:
        columns["counts_" + g] = c[g].astype(int)
    for name, panel in PANELS.items():
        values = np.column_stack([c[g] for g in panel])
        columns[name + "_n_detected"] = (values > 0).sum(axis=1)
        columns[name + "_UMI"] = values.sum(axis=1).astype(int)
    a.obs = a.obs.drop(columns=list(columns), errors="ignore").join(pd.DataFrame(columns, index=a.obs_names))
    a.obs["identity_positive"] = (
        ((c["Gata3"] > 0) | (c["Rora"] > 0))
        & (a.obs.ILC2_identity_n_detected >= 3)
        & (a.obs.ILC2_support_n_detected >= 1)
    )
    # 只将一致的替代谱系程序作为强证据，不删单个 Nkg7/Ccr6/Il17a 阳性细胞。
    for name in ["T", "B", "Myeloid", "Endothelial", "Stromal", "Epithelial", "Erythroid"]:
        minimum_umi = 5 if name in {"Endothelial", "Stromal", "Epithelial", "Erythroid"} else 3
        a.obs["lineage_" + name] = (a.obs[name + "_n_detected"] >= 2) & (a.obs[name + "_UMI"] >= minimum_umi)
    nk_specific = np.column_stack([c[g] for g in ["Ncr1", "Klrd1", "Eomes", "Prf1", "Gzmb"]])
    a.obs["lineage_NK_ILC1"] = ((nk_specific > 0).sum(axis=1) >= 2) & (nk_specific.sum(axis=1) >= 3)
    a.obs["ilc3_program_review"] = ((c["Rorc"] > 0) & ((c["Il23r"] > 0) | (c["Il22"] >= 5))) | ((c["Il23r"] >= 2) & (c["Il22"] >= 5))
    a.obs["additional_quality_pass"] = (a.obs.n_genes_by_counts >= 500) & (a.obs.total_counts >= 1000)
    missing = [g for g in all_genes if g not in set(a.var.gene_symbol)]
    return missing


def apply_standard(a: ad.AnnData, policy: dict) -> None:
    """统一宽松筛选；add_evidence保留旧规则供历史脚本复现，本函数显式应用新预设。"""
    core = a.obs.ILC2_identity_n_detected
    total = core + a.obs.ILC2_support_n_detected
    a.obs["identity_positive"] = (core >= policy["min_core_markers"]) & (total >= policy["min_total_identity_support_markers"])
    for name in ["T", "B", "Myeloid", "Endothelial", "Stromal", "Epithelial", "Erythroid"]:
        a.obs["lineage_" + name] = (a.obs[name + "_n_detected"] >= policy["min_alternative_markers"]) & (a.obs[name + "_UMI"] >= policy["min_alternative_UMI"])
    nk = marker_values(a, ["Ncr1", "Klrd1", "Eomes", "Prf1", "Gzmb"], "counts")
    a.obs["lineage_NK_ILC1"] = ((nk > 0).sum(axis=1) >= policy["min_alternative_markers"]) & (nk.sum(axis=1) >= policy["min_alternative_UMI"])
    acinar = marker_values(a, ACINAR, "counts")
    a.obs["acinar_n_detected"] = (acinar > 0).sum(axis=1)
    a.obs["acinar_UMI_fraction"] = acinar.sum(axis=1) / a.obs.total_counts.to_numpy()
    a.obs["lineage_Acinar"] = (a.obs.acinar_n_detected >= policy["acinar_min_markers"]) & (a.obs.acinar_UMI_fraction >= policy["acinar_min_UMI_fraction"])
    a.obs["alternative_program"] = a.obs.filter(regex="^lineage_").any(axis=1) | a.obs.ilc3_program_review
    a.obs["identity_pass"] = a.obs.identity_positive & ~a.obs.alternative_program
    a.obs["selected_ILC2"] = a.obs.identity_pass & (a.obs.total_counts >= policy["min_candidate_UMI"])
    a.obs["doublet_review"] = a.obs.doublet_score > policy["scrublet_review_threshold"]
    a.obs["mixed_lineage_doublet_candidate"] = a.obs.doublet_review & a.obs.alternative_program
    fractions = a.obs.groupby("library_cluster", observed=True).alternative_program.mean()
    a.obs["mixed_cluster_review"] = a.obs.library_cluster.map(fractions).astype(float) >= policy["mixed_cluster_review_fraction"]
