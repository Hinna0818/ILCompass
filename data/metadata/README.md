# 样本与来源账本

`sample_manifest.tsv` 为已下载 GSE117567 的五个文库。role/split 尚未分配，proposed_role 只是候选；不能据此启动训练。raw_cells 是原始矩阵中的细胞数，不能替代动物/池数。

`raw_file_manifest.tsv` 列出原始三件套的相对路径、GSM 来源、SHA256；已与实际文件核验。

后续阶段 1 的逐细胞清单还需 identity_confidence、training_tier、QC/排除原因；阶段 2 生成互斥的 split_manifest。测试研究中的 control/case 必须成组留出。
