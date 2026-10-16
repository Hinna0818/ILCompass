# 小鼠 ILC2 统一 QC 标准：ilc2_balanced_v1

2026-10-15起的项目默认规则。数值唯一来源 `configs/1QC.json`；正负程序由 `src/ilc2_qc.py:apply_standard` 实现。适用于已分选/富集ILC2的小鼠全转录组scRNA-seq文库，不直接作为未分选全组织或其他物种的身份分类器。数值为项目工作预设，未经独立流式/参考标签验证。

## 1. 输入与固定基础QC

- X或counts必须是有限、非负整数UMI。按GSM/捕获文库独立计算，不将组织混合后确定3SD。
- 未调用的raw液滴：先用检出基因≥200、总UMI≥500确定候选。已调用细胞也使用同样RNA下限，但无需将未提供的空液滴补入。
- 该候选门槛替代了旧版500genes/1000UMI，并非EmptyDrops/Cell Ranger统计cell calling。低RNA保留结果必须结合下述身份与背景复核，不能因有counts就称为细胞。
- 保留既有GeneCompass paper-based corrected v1：≥200检出基因；protein_coding/miRNA检出≥7；mt≤15%；样本前后≥4细胞；检出基因数≤样本候选过滤前均值+3SD（ddof=0）。3SD数值随文库而变，不能强制不同GSE使用同一数值。
- 全基因空间计算QC，随后保留protein_coding/lncRNA/miRNA；不以HVG矩阵替代最终counts。这是项目对论文描述的明确实现，不声称逐行复刻官方代码；checkpoint词表/表达编码仍需阶段2核验。

## 2. 宽松正向身份：保留而非追求典型表达

核心面板：Gata3、Rora、Il7r、Id2、Thy1、Bcl11b、Gfi1。
支持面板：Icos、Klrg1、Il1rl1、Il17rb、Il2ra、Il5、Il13、Areg、Arg1。

通过需同时满足：核心检出≥1个；核心+支持合计检出≥2个。检出为原始UMI>0。可以是两个核心基因，也可以一个核心加一个支持基因；不强制Gata3/Rora/ST2/效应因子表达。仅一个marker或仅支持面板不足，记为weak_identity，不称为已证明非ILC2。

不对整个群作自动否决。混合群（替代程序占比≥20%）只标记 `mixed_cluster_review`；群内各细胞仍依据自身多基因证据判定。状态与组织来源不作为身份否决项。

## 3. 替代谱系：多基因证据排除，保留不确定性

以下各面板至少2基因检出且合计≥5 UMI，标记一致替代谱系，暂不进入ILC2候选对象：

| 程序 | 基因 |
|---|---|
| T | Cd3d/Cd3e/Cd3g/Trac/Trdc |
| B | Cd79a/Cd79b/Ms4a1/Cd19 |
| NK/ILC1 | Ncr1/Klrd1/Eomes/Prf1/Gzmb（Nkg7/Xcl1单独不参与排除） |
| 髓系 | Lyz2/Adgre1/Csf1r/Itgam/Lst1 |
| 内皮 | Pecam1/Cdh5/Cldn5/Kdr/Esam |
| 基质 | Col1a1/Col1a2/Dcn/Lum/Pdgfra |
| 上皮 | Epcam/Krt8/Krt18/Krt19 |
| 红系 | Alas2/Bpgm/Gypa/Slc4a1/Klf1（不因Hb单阳性排除） |

ILC3：Rorc>0且（Il23r>0或Il22≥5），或Il23r≥2且Il22≥5。称一致ILC3程序待审查，可能包含可塑性/混合身份，不称为实验确认的污染。

腺泡：Cela2a/Ctrb1/Clps/Prss2/Cela1/Cela3b/Pnlip/Cpa1/Cpb1/Reg1/Cel/Sycn中≥3个检出，且该面板占总UMI≥5%。低水平消化酶RNA仅记录。记录缺失feature，缺少Trac/Trdc不解释为蛋白阴性。

单Rorc/Ccr6/Il17a、MHC-II、单Cd3g、热休克、干扰素、增殖或效应程序不直接删细胞。多基因条件也不能排除ambient RNA造成的假阳性，仍须检查群marker与表达强度。

## 4. 双细胞、低RNA和状态

每GSM用原始counts运行Scrublet，expected_rate=0.05（假设）、30PC、seed=0。score>0.25标记 `doublet_review`；没有清晰校准依据时，不靠该分数独立删除。替代谱系阳性本来就按第3节排除，高分+替代程序另记mixed_lineage_doublet_candidate；没有新增一项隐藏阈值。

保留细胞可能包括同型双细胞。后续任务需要高纯singlet时，须另行校准、报告敏感性，不能把本版候选集称为全部真实singlet。

基因<500或UMI<1000但满足本版RNA下限者标记 `low_RNA_review`；仅作质量分层。骨髓保留独立背景对象，不等于低质量，也不全部解释为前体。

效应/热休克/干扰素只在筛选后评分，不作为删除规则。GSE153748的作者signature是本研究内部状态参考，非独立身份证据；flag为队列相对高值，队列改变时阈值改变，不能跨对象直接比较比例。

## 5. 输出与跨研究约定

最终对象X和layers['counts']保留相同原始整数UMI、唯一Ensembl ID、GSE/GSM/barcode、组织、真实来源metadata、QC/身份/复核标记和未分配split。归一化只用于聚类/图/评分。保留去留表、逐样本数量、群marker、软件/输入hash及验证结果；无需保留所有重复矩阵。

GSE153748入口：`python src/qc_gse153748.py`，从原tar运行；输出 `data/train_data/GSE153748.h5ad`、`data/processed/GSE153748_qc/GSE153748_BM_ILC2.h5ad`，结果只在 `results/GSE153748_qc/`。

GSE117567现有25,010细胞对象是strict_v1历史版本，此次不覆盖。`results/ILC2_qc_standard/GSE117567_identity_comparison.tsv`仅在既有基础QC通过细胞上应用新身份标准作对照，不是完整新QC对象。正式跨GSE合并前须把GSE117567从原始数据用同版本候选/基础QC重跑，不能混用旧严格版并声称标准完全相同。

## 依据与限制

- [GeneCompass论文](https://www.nature.com/articles/s41422-024-01034-y)、[官方仓库](https://github.com/xCompass-AI/GeneCompass)：保留项目已声明的基础QC实现及后续编码边界。
- [OSCA：Quality control, redux](https://bioconductor.org/books/release/OSCA.advanced/quality-control-redux.html)：QC指标受生物状态影响，按文库处理并审查丢失的细胞类型；统一算法不等于强制相同阈值数值或保留比例。
- [Scrublet原始实现与论文链接](https://github.com/swolock/scrublet)：双细胞检测依赖输入群体组成，不将未校准分数视为真实概率。
- [GSE117567原研究](https://doi.org/10.1038/s41590-018-0201-4)、[GSE153748原研究](https://www.nature.com/articles/s41385-021-00431-5)：组织与分选背景不同，不能用最终数量接近作为QC合理性的依据。

本版优先减少对典型marker和聚类边界的依赖。代价是候选集的残余背景/双细胞风险增加；是否改善模型结果需后续验证，不能从保留数变多直接下结论。
