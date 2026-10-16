# 基础 QC 函数调用

先激活 ilc2_model。qc_utils.py 接受 Scanpy 的 AnnData，内部用稀疏矩阵精确计算，输入保持不变。

| 函数 | 用途 |
|---|---|
| read_10x_sample | 读取一个有前缀的旧版10x三件套；gene ID 为 Ensembl，细胞名为 sample:barcode |
| read_gene_annotation | 从真实 GTF 的 gene 记录解析 biotype、symbol 与线粒体标记 |
| qc_sample | 对单样本 raw counts 计算指标并返回 full/core/decisions/summary |
| select_reviewed_ilc2 | 依据已完成的身份/置信度注释选取 ILC2，不生成自动 marker 标签 |

```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd()  # 在仓库根目录运行
sys.path.insert(0, str(ROOT / "src"))
from qc_utils import read_10x_sample, read_gene_annotation, qc_sample

# 参考只读取一次，五个组织分别 QC，不在合并矩阵上计算 3 SD 上限。
genes = read_gene_annotation(ROOT / "data/references/Mus_musculus.GRCm38.84.gtf.gz")
manifest = pd.read_csv(ROOT / "data/metadata/sample_manifest.tsv", sep="\t")
summaries = []
for row in manifest.itertuples(index=False):
    adata = read_10x_sample(ROOT / row.input_path, row.file_prefix, sample_id=row.sample)
    adata.obs["tissue"] = row.tissue
    adata.obs["study"] = row.study
    adata.obs["condition"] = row.condition

    result = qc_sample(adata, genes, min_genes=200, min_cells=4,
                       min_coding_mirna=7, max_mt_pct=15, n_genes_upper_sd=3)
    full = result["full"]       # 保留细胞，完整基因 raw counts
    core = result["core"]       # 同一批细胞的允许 biotype 视图
    decisions = result["decisions"]  # 所有原始细胞的 QC 原因

    out = ROOT / "data/processed/GSE117567_basic_qc" / row.sample
    out.mkdir(parents=True, exist_ok=True)
    full.write_h5ad(out / "full_counts_qc.h5ad", compression="gzip")
    core.write_h5ad(out / "core_counts_qc.h5ad", compression="gzip")
    decisions.to_csv(out / "cell_decisions.tsv.gz", sep="\t")
    summaries.append({"tissue": row.tissue, **result["summary"]})

summary = pd.DataFrame(summaries)
print(summary[["sample", "tissue", "n_input_cells", "n_pass_cells", "n_removed_cells"]])
```

qc_sample 默认优先使用 layers['counts']，否则用 X；也可 counts_layer='raw_counts'。若 X 已归一化，必须提供真正原始 counts。函数拒绝缺少注释的基因，不靠名称猜类型。

完成聚类与多基因审查后，可在 full.obs 中填写 cell_type/identity_confidence，再调用：

```python
from qc_utils import select_reviewed_ilc2
ilc2, identity_decisions = select_reviewed_ilc2(full)
```

不能把当前 ILC2_enriched 初步标签直接改成 high；需要先完成复核。模型输入还需 checkpoint 配套词表/归一化编码，full.X/core.X 此时都未标准化。

批处理等价入口：`python src/1QC.py --basic`。实测结果见 results/GSE117567_basic_qc/QC_report.md。

## 统一ILC2身份规则（2026-10-15）

后续小鼠ILC2富集文库默认遵循 [ilc2_balanced_v1](ilc2_qc_standard.md)。在原始counts经过候选/基础QC、每GSM Scrublet并完成文库聚类后：

```python
from ilc2_qc import add_evidence, apply_standard
policy = config["identity"]  # configs/1QC.json
missing = add_evidence(adata)
adata.obs["library_cluster"] = adata.obs["sample"].astype(str) + ":" + adata.obs["leiden"].astype(str)
apply_standard(adata, policy)
selected = adata[adata.obs.selected_ILC2].copy()
selected.X = selected.layers["counts"].copy()
```

add_evidence单独调用保留旧strict_v1证据逻辑，必须显式apply_standard才是新标准。状态分数、doublet_review和mixed_cluster_review不独立进入删除条件。具体GSE153748完整入口为src/qc_gse153748.py；GSE117567旧对象仍为历史严格版本。
