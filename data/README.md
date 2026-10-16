# 数据目录

原始 counts 保持只读使用。`GSE117567_RAW/` 是已下载的五个文库，保持现有路径；后续研究放在 `raw/<study>/`。

- `metadata/`：样本来源、真实生物重复、条件及后续 split 清单，可提交。
- `references/`：版本固定的 Ensembl/core/token/参考统计，记录来源与 hash。
- `processed/`：阶段 1 输出，完整 counts 与模型核心视图分开。
- `prepared/`：阶段 2 输出，按角色划分的 Hugging Face dataset 和编码版本。

大矩阵与逐细胞压缩表不纳入 Git。未知 animal/pool、注释 release 保持未确定状态，不推测填值。
