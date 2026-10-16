# 参考资产

已取得原研究所用 `Mus_musculus.GRCm38.84.gtf.gz`，来源：https://ftp.ensembl.org/pub/release-84/gtf/mus_musculus/Mus_musculus.GRCm38.84.gtf.gz 。使用前经过 gzip CRC 检查；版本依据原研究 Methods：https://pmc.ncbi.nlm.nih.gov/articles/PMC6202223/ 。本次 QC 的 SHA256 记录在 results/GSE117567_basic_qc/provenance.json。

lncRNA 旧 biotype 归并依据 GENCODE/Ensembl 官方定义：https://www.gencodegenes.org/pages/biotypes.html 。不按基因名称猜 biotype，未知 ID 会显式报错。

本次核心视图是本项目 Ensembl biotype 策略，尚未与 checkpoint token/先验集合取交集；这在阶段 2 完成。正式模型编码仍需配套 token 词表、同源映射和基因中位数参考。大参考文件不纳入 Git。
