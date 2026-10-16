# GSE117567 高可信 ILC2 筛选（2026-10-14）

- [x] 核对基础 QC 与已有探索结果，检索 ILC2 注释、组织差异和双细胞方法。
- [x] 按文库计算 Scrublet、按组织重新聚类，形成可复核的多基因证据。
- [x] 依据细胞群与细胞层面证据筛选，构建 obs/var/uns 元数据。
- [x] 保存 data/train_data/GSE117567.h5ad，统计并核验原始 counts/来源/身份。

主方案：已有 paper-based 基础 QC + ≥500 genes/≥1000 UMI 的质量层 + 按 GSM Scrublet + 多基因正/负身份程序 + 按组织聚类证据。默认四个外周组织作为首版主训练集；骨髓独立审查并计入排除统计，不自动称全部前体。

来源：
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7062532/ （Areg/Il7r/Rora/Gata3 与替代谱系排除，ILCP 与 ILC2 区别）
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117567 （组织、8–14周、稳态reporter背景）
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6202223/ （组织受体差异、分选、取样及测序）
- https://www.sc-best-practices.org/cellular-structure/annotation/ （群体与marker审查）
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6625319/ （Scrublet及局限）

阈值为本项目筛选规则，不宣称存在文献验证的统一 RNA marker 数值门槛。缺失 animal/sex/具体年龄/解剖亚区不填猜测值。主数据的 metadata 存于 obs，每样本来源与筛选方案存于 uns['metadata']，保留全基因原始整数 counts。

完成：25,010×27,998 H5AD及完整metadata、按样本统计、逐细胞去留表、doublet敏感性与counts一致性核验。

执行修正：Scrublet自动阈值缺失改为明确人工审查；JSON来源数组转为标准列表；报告生成改用项目Python环境。以上未改变最终原始counts。
