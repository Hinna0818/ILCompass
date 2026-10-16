"""单样本原始 RNA counts 的基础 QC 工具。

顺序：读取/校验 → 在全基因 counts 上计算指标 → 过滤细胞 → 核心基因视图。
本模块执行 GeneCompass paper-based 基础规则，不自动判定 ILC2 身份。
所有过滤原因均保存，输入 AnnData 不会被原地修改。
"""
from pathlib import Path
import gzip
import re
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


# GENCODE/Ensembl 将这些旧 biotype 合并为 lncRNA；不包含 TEC 或假基因。
# 来源：https://www.gencodegenes.org/pages/biotypes.html
LNCRNA_BIOTYPES = frozenset({
    "lncRNA", "3prime_overlapping_ncRNA", "antisense", "bidirectional_promoter_lncRNA",
    "lincRNA", "macro_lncRNA", "non_coding", "processed_transcript",
    "sense_intronic", "sense_overlapping",
})


def _gene_ids(values: Iterable) -> pd.Index:
    """去除 Ensembl 版本后缀；重复 ID 必须由调用者明确解决，不能静默合并。"""
    return pd.Index([re.sub(r"\.\d+$", "", str(v)) for v in values])


def _raw_counts(matrix) -> sparse.csr_matrix:
    """校验有限、非负、整数计数；统一稀疏 CSR，避免构造细胞×全基因稠密矩阵。"""
    x = sparse.csr_matrix(matrix, copy=True)
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError("counts 必须是有限、非负的原始计数。")
    if not np.equal(x.data, np.floor(x.data)).all():
        raise ValueError("检测到非整数表达值；请传原始 counts，不能传 log1p/标准化矩阵。")
    if x.data.size and x.data.max() > np.iinfo(np.int64).max:
        raise ValueError("计数超出 int64 范围。")
    x = x.astype(np.int64)  # 聚合重复位置前扩展，防止 int32 求和溢出。
    x.sum_duplicates()
    x.eliminate_zeros()  # 稀疏存储的显式零不能当成检出基因。
    dtype = np.int32 if not x.data.size or x.data.max() <= np.iinfo(np.int32).max else np.int64
    return x.astype(dtype)


def read_10x_sample(directory: str | Path, prefix: str, *, sample_id: str | None = None) -> ad.AnnData:
    """读取一个旧版 10x 三件套，返回 cell×gene 的 raw-count AnnData。

    适用于 GSE117567 的 `<prefix>_matrix.mtx.gz/genes.tsv.gz/barcodes.tsv.gz`。
    var_names 使用唯一 Ensembl ID，gene_symbol 仅为注释，不因 symbol 重复改基因 ID。
    每个细胞命名为 sample:barcode，避免不同文库 barcode 相同导致 ID 冲突。
    """
    directory = Path(directory)
    genes = pd.read_csv(directory / f"{prefix}_genes.tsv.gz", sep="\t", header=None, dtype=str)
    if genes.shape[1] < 2:
        raise ValueError("genes.tsv 至少应包含 gene ID 和 symbol 两列。")
    barcodes = pd.read_csv(directory / f"{prefix}_barcodes.tsv.gz", sep="\t", header=None, dtype=str)[0]
    with gzip.open(directory / f"{prefix}_matrix.mtx.gz", "rb") as handle:
        x = _raw_counts(mmread(handle).T)
    ids = _gene_ids(genes[0])
    if x.shape != (len(barcodes), len(genes)) or not ids.is_unique or not barcodes.is_unique:
        raise ValueError("矩阵尺寸、gene ID 或 barcode 唯一性检查失败。")
    sample_id = sample_id or prefix.split("_")[0]
    var = pd.DataFrame({"gene_id_original": genes[0].to_numpy(), "gene_symbol": genes[1].to_numpy()}, index=ids)
    var.index.name = "gene_id"
    obs = pd.DataFrame({"barcode": barcodes.to_numpy(), "sample": sample_id}, index=sample_id + ":" + barcodes)
    obs.index.name = "cell_id"  # pandas Series 默认名为整数 0，h5ad 索引键必须是字符串。
    a = ad.AnnData(x, obs=obs, var=var)
    a.layers["counts"] = x.copy()
    return a


def read_gene_annotation(gtf_path: str | Path) -> pd.DataFrame:
    """逐行解析 Ensembl GTF 的 gene 记录，不把 exon/transcript 当成额外基因。

    输出以稳定 gene ID 为索引：gene_symbol、gene_biotype、biotype_group、is_mt。
    使用 GTF 染色体 MT/chrM 判断线粒体，避免仅靠 symbol 前缀。
    GTF 与待分析数据必须物种/版本匹配；本项目测试使用 GRCm38 release 84。
    """
    path = Path(gtf_path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise ValueError("GTF 记录不是 9 列。")
            if columns[2] != "gene":
                continue
            attributes = dict(re.findall(r'(\w+) "([^"]*)"', columns[8]))
            biotype = attributes.get("gene_biotype", attributes.get("gene_type"))
            if "gene_id" not in attributes or not biotype:
                raise ValueError("GTF gene 记录缺少 gene_id 或 gene_biotype。")
            rows.append([attributes["gene_id"], attributes.get("gene_name", attributes["gene_id"]), biotype, columns[0] in {"MT", "chrM", "M"}])
    if not rows:
        raise ValueError("GTF 没有 gene 记录；请检查下载文件与格式。")
    frame = pd.DataFrame(rows, columns=["gene_id", "gene_symbol", "gene_biotype", "is_mt"])
    frame["gene_id"] = _gene_ids(frame.gene_id)
    frame = frame.drop_duplicates()
    if frame.gene_id.duplicated().any():
        raise ValueError("GTF 同一稳定 gene ID 有冲突注释，不能静默取第一条。")
    frame["biotype_group"] = frame.gene_biotype.where(~frame.gene_biotype.isin(LNCRNA_BIOTYPES), "lncRNA")
    return frame.set_index("gene_id")


def qc_sample(
    adata: ad.AnnData,
    gene_annotation: pd.DataFrame,
    *,
    counts_layer: str | None = None,
    min_genes: int = 200,
    min_cells: int = 4,
    min_coding_mirna: int = 7,
    max_mt_pct: float = 15,
    n_genes_upper_sd: float = 3,
    core_gene_ids: Iterable | None = None,
) -> dict:
    """对一个样本做基础 QC，返回 full/core/decisions/summary 四项。

    参数
    ----
    counts_layer：指定原始计数层；默认优先 layers['counts']，否则使用 X。
    min_genes：至少检出 200 个基因（count>0），在原始全基因空间计算。
    min_cells：样本输入以及过滤后的可用细胞都至少 4 个，否则整个样本退出。
    min_coding_mirna：protein_coding 与 miRNA 并集至少检出 7 个，而非删除 >7 个。
    max_mt_pct：线粒体原始 counts/全部原始 counts×100；恰好 15% 保留。
    n_genes_upper_sd：样本所有非空细胞、过滤前 n_genes 的均值+3×总体 SD 上限。
        只过滤上侧；SD=0 不执行此项。它不等同于官方示例的 UMI/mt 双侧 z-score。
    core_gene_ids：可选的固定核心 gene ID 白名单，与允许 biotype 取交集。
        未提供时按 Ensembl biotype 定义本项目核心视图，不冒充 checkpoint 词表。

    返回
    ----
    full：过滤后全部原始基因；X 与 layers['counts'] 均为整数 raw counts。
    core：同一批细胞中 protein_coding/lncRNA/miRNA 的核心基因视图。
    decisions：所有输入细胞的 QC 指标、每条失败规则及合并原因。
    summary：样本级数量、阈值、基因覆盖、核心过滤损失等统计。

    注意：本函数不会归一化、做 HVG/批次校正、检测双细胞或自动判定 ILC2。
    输入不原地修改；原来的归一化/邻居/UMAP 不继承到新的 raw-count 输出。
    """
    if min_genes < 0 or min_cells < 1 or min_coding_mirna < 0 or not 0 <= max_mt_pct <= 100 or not np.isfinite(n_genes_upper_sd) or n_genes_upper_sd < 0:
        raise ValueError("QC 参数范围不合法。")
    if adata.isbacked:
        raise ValueError("请按样本载入内存再调用，不直接对 backed AnnData 运行。")
    if not adata.obs_names.is_unique:
        raise ValueError("细胞 ID 必须唯一。")
    if "sample" in adata.obs and adata.obs["sample"].nunique(dropna=False) > 1:
        raise ValueError("qc_sample 必须接收单样本；请按 sample 拆分后调用。")
    if counts_layer is None:
        counts_layer = "counts" if "counts" in adata.layers else None
    if counts_layer is None and "log1p" in adata.uns:
        raise ValueError("X 标记为 log1p 且未提供 counts 层。")
    x = _raw_counts(adata.layers[counts_layer] if counts_layer is not None else adata.X)
    ids = _gene_ids(adata.var["gene_id"] if "gene_id" in adata.var else adata.var_names)
    if not ids.is_unique:
        raise ValueError("去除版本后缀后 gene ID 重复，请先明确聚合策略。")
    required = {"gene_symbol", "gene_biotype", "biotype_group", "is_mt"}
    if not required <= set(gene_annotation.columns) or not gene_annotation.index.is_unique:
        raise ValueError("gene_annotation 必须由唯一 gene ID 索引并包含规定注释列。")
    mapped = gene_annotation.reindex(ids)
    if mapped.gene_biotype.isna().any():
        missing = ids[mapped.gene_biotype.isna()]
        raise ValueError(f"{len(missing)} 个基因缺少匹配注释，如 {list(missing[:5])}；请核实 Ensembl 版本或 gene ID，不能猜 biotype。")
    mt = mapped.is_mt.to_numpy(dtype=bool)
    coding = mapped.biotype_group.isin(["protein_coding", "miRNA"]).to_numpy()
    core = mapped.biotype_group.isin(["protein_coding", "lncRNA", "miRNA"]).to_numpy()
    if core_gene_ids is not None:
        core &= ids.isin(_gene_ids(core_gene_ids))
    if not core.any() or not coding.any() or not mt.any():
        raise ValueError("核心基因/蛋白编码+miRNA/线粒体基因集合为空，请核实注释覆盖。")

    # CSR 已消除显式零，getnnz 精确表示检出基因数，不需要全矩阵 toarray。
    total = np.asarray(x.sum(axis=1, dtype=np.int64)).ravel()
    detected = np.asarray(x.getnnz(axis=1)).ravel()
    mt_counts = np.asarray(x[:, mt].sum(axis=1, dtype=np.int64)).ravel()
    coding_detected = np.asarray(x[:, coding].getnnz(axis=1)).ravel()
    mt_pct = np.divide(mt_counts * 100.0, total, out=np.zeros(len(total)), where=total > 0)
    valid = detected[total > 0]
    mean = float(valid.mean()) if len(valid) else 0.0
    std = float(valid.std(ddof=0)) if len(valid) else 0.0
    upper = mean + n_genes_upper_sd * std
    decisions = adata.obs.copy()
    decisions.index.name = "cell_id"
    decisions["total_counts"] = total
    decisions["n_genes_by_counts"] = detected
    decisions["pct_counts_mt"] = mt_pct
    decisions["n_coding_mirna_by_counts"] = coding_detected
    flags = {
        "zero_counts": total == 0,
        "low_genes": detected < min_genes,
        "low_coding_mirna": coding_detected < min_coding_mirna,
        "high_mt": mt_pct > max_mt_pct,
        "high_genes": (detected > upper) & (std > 0),
        "small_sample_input": np.full(adata.n_obs, adata.n_obs < min_cells),
    }
    for name, values in flags.items():
        decisions["qc_fail_" + name] = values
    passed = ~np.column_stack(list(flags.values())).any(axis=1)
    small_after = int(passed.sum()) < min_cells
    # 若过滤后样本不足 4 个，只对本来通过基础规则的细胞补记该原因。
    decisions["qc_fail_small_sample_after"] = passed & small_after
    decisions["qc_pass"] = passed & (not small_after)
    reason_columns = [c for c in decisions if c.startswith("qc_fail_")]
    decisions["qc_reason"] = [";".join(c.removeprefix("qc_fail_") for c, fail in zip(reason_columns, row) if fail) or "pass" for row in decisions[reason_columns].itertuples(index=False, name=None)]
    decisions["qc_core_counts"] = np.asarray(x[:, core].sum(axis=1, dtype=np.int64)).ravel()
    decisions["qc_core_detected_genes"] = np.asarray(x[:, core].getnnz(axis=1)).ravel()

    var = adata.var.copy()
    var.index = ids
    var.index.name = "gene_id"
    for column in required:
        if column != "gene_symbol" or column not in var:
            var[column] = mapped[column].to_numpy()
    var["is_core"] = core
    var["is_coding_or_miRNA"] = coding
    var["mt"] = mt
    keep = decisions.qc_pass.to_numpy()
    full = ad.AnnData(x[keep].copy(), obs=decisions.loc[keep].copy(), var=var)
    full.layers["counts"] = full.X.copy()
    parameters = {"min_genes": min_genes, "min_cells": min_cells, "min_coding_mirna": min_coding_mirna, "max_mt_pct_inclusive": max_mt_pct, "n_genes_upper_sd": n_genes_upper_sd, "sd_ddof": 0}
    summary = {
        "sample": str(adata.obs["sample"].iloc[0]) if adata.n_obs and "sample" in adata.obs else "not_available",
        "n_input_cells": adata.n_obs, "n_pass_cells": full.n_obs, "n_removed_cells": adata.n_obs - full.n_obs,
        "n_input_genes": adata.n_vars, "n_core_genes": int(core.sum()), "n_excluded_genes": int((~core).sum()),
        "n_genes_mean_before_qc": mean, "n_genes_sd_before_qc": std, "n_genes_upper_limit": upper,
        "n_UMI_retained_cells_all_genes": int(total[keep].sum()), "n_UMI_retained_cells_core_genes": int(decisions.loc[keep, "qc_core_counts"].sum()),
        "reference_gene_coverage": 1.0,
        "core_source": "provided_whitelist_and_biotypes" if core_gene_ids is not None else "ensembl_biotypes_only",
    }
    summary.update({"n_fail_" + name: int(decisions["qc_fail_" + name].sum()) for name in [*flags, "small_sample_after"]})
    full.uns["basic_qc_parameters"] = parameters
    full.uns["basic_qc_summary"] = summary
    full.uns["identity_selection_status"] = "not_performed_by_basic_qc"
    model_core = full[:, core].copy()
    return {"full": full, "core": model_core, "decisions": decisions, "summary": summary}


def select_reviewed_ilc2(
    adata: ad.AnnData,
    *,
    label_key: str = "cell_type",
    confidence_key: str = "identity_confidence",
    min_cells: int = 4,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """身份复核后选取 ILC2；不会根据单个 marker 自动生成标签。

    必须先由多基因/聚类/参考注释审查填写 obs：cell_type='ILC2'、
    identity_confidence='high'。返回筛选对象和所有输入细胞的身份去留表。
    ILC2 数量不足 min_cells 时该样本不进入训练，counts 与 gene ID 保持不变。
    """
    if min_cells < 1 or label_key not in adata.obs or confidence_key not in adata.obs:
        raise ValueError("需要有效 min_cells，以及已复核的细胞类型和身份置信度列。")
    if "sample" in adata.obs and adata.obs["sample"].nunique(dropna=False) > 1:
        raise ValueError("请按单样本进行 ILC2 选择。")
    decisions = adata.obs.copy()
    keep = decisions[label_key].eq("ILC2").fillna(False) & decisions[confidence_key].eq("high").fillna(False)
    decisions["ilc2_identity_pass"] = keep
    decisions["ilc2_small_sample"] = keep & (int(keep.sum()) < min_cells)
    decisions["ilc2_keep"] = keep & (int(keep.sum()) >= min_cells)
    selected = adata[decisions.ilc2_keep.to_numpy()].copy()
    selected.uns["identity_selection_status"] = "reviewed_ILC2_high_confidence"
    return selected, decisions
