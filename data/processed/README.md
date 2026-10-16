# 阶段 1 输出

`GSE117567_basic_qc/<GSM>/` 是新的基础 QC 输出：
- full_counts_qc.h5ad：基础过滤后细胞、全部原始基因、整数 counts。
- core_counts_qc.h5ad：同一批细胞的允许 biotype 核心视图。
- cell_decisions.tsv.gz：全部输入细胞的指标与去留原因。
- gene_annotation.tsv：输入基因的类型和核心标记。

这一步不自动判定 ILC2。身份审查的已有探索对象保留在 results/GSE117567_qc；新基础 QC 汇总和核验在 results/GSE117567_basic_qc。不能把基础 QC 后细胞全部当作高可信 ILC2。

原始数据保持原路径，大型输出不纳入 Git。
