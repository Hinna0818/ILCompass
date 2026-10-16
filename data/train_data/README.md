# Post-training candidates

`GSE153748.h5ad`：2026-10-15统一 **ilc2_balanced_v1** 外周候选，22,956细胞×27,998基因，X和counts均为原始整数UMI。骨髓8,938个候选另存 `../processed/GSE153748_qc/GSE153748_BM_ILC2.h5ad`。正式入口 `../../src/qc_gse153748.py`，唯一当前报告 `../../results/GSE153748_qc/QC_report.md`，默认标准 `../../docs/ilc2_qc_standard.md`。保留doublet_review/low_RNA_review/mixed_cluster_review，计算支持不等于独立确认身份或singlet。

`GSE117567.h5ad`：2026-10-14 **strict_v1历史版本**，25,010细胞×27,998基因，原始UMI；此次未覆盖。其筛选报告 `../../results/GSE117567_ilc2_selection/selection_report.md`、脚本 `../../main/build_train_gse117567.py` 与旧配置 `../../configs/GSE117567_ilc2_selection.json` 仅用于复现旧版本。

**当前两个对象不是同一身份QC版本，正式合并前需把GSE117567从原始数据用balanced_v1完整重跑。** 本次新身份规则对照29,070外周候选不冒充已重跑的新对象。

细胞来源和筛选信息在obs/uns['metadata']。动物信息缺失不猜测，混样文库不当独立动物；split仍unassigned。候选原始counts尚未完成GeneCompass的词表匹配/表达编码，不能直接当已编码训练输入。H5AD受.gitignore忽略，不提交Git。
