# 四阶段配置

`1QC.json` 已由 `src/1QC.py --basic` 使用：基因注释为 GRCm38/Ensembl 84，核心视图由允许 biotype 构建，身份筛选仍需审查。core_gene_list=null 表示没有额外白名单；不代表已核验 checkpoint token 覆盖。

阶段 2–4 的 JSON 仍为设计参数，尚未执行。null 表示所需资产或训练参数待核验，不能用随机初始化或缺省统计悄悄补齐。

QC ≥7 是对原论文文字与官方代码矛盾的明确处理，详见 CLAUDE.md。验证严格子集默认关闭，须预先确定并与共同基础 QC 结果同时报告。
