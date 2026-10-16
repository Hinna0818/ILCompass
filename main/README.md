# 脚本用途索引

更新：2026-10-15。所有项目自写探索性/临时脚本及已运行 notebook 集中在本目录；正式工具在 `src/`，检查在 `tests/`。从 **ILC2_model 项目根目录**执行下面命令，先 `conda activate ilc2_model`。`main/` 只放代码与说明，矩阵、图、日志放在 `data/` 或按运行命名的 `results/`，禁止写入 main/。

## 探索与数据集专用脚本

| 文件 | 用途与状态 | 输入 → 输出 | 运行命令 |
|---|---|---|---|
| `qc_gse117567.py` | **历史初筛，已运行**；500 genes/1000 UMI/mt<15%，ILC3 候选与其他谱系审查；不是当前主训练集 | `data/GSE117567_RAW/` → `results/GSE117567_qc/` | `python main/qc_gse117567.py`；审查后加 `--finalize --review results/GSE117567_qc/ilc3_review.json` |
| `build_train_gse117567.py` | **历史strict_v1，已运行**；基础QC后按GSM Scrublet、按组织聚类、多marker身份筛选，构建metadata | `data/processed/GSE117567_basic_qc/` → `results/GSE117567_ilc2_selection/`；最终 `data/train_data/GSE117567.h5ad` | `python main/build_train_gse117567.py`；复核后 `python main/build_train_gse117567.py --finalize --review configs/GSE117567_ilc2_selection.json` |
| `plot_gse117567_umap.py` | **当前联合UMAP可视化**；不修改训练文件，不重新过滤；联合群仅作探索 | `data/train_data/GSE117567.h5ad` → `results/GSE117567_train_umap/` | `python main/plot_gse117567_umap.py` |
| `plot_ilc2_studies_umap.py` | 两研究单独及联合UMAP；使用当前外周对象、明确不同QC版本；无批次校正，不修改输入 | 两个train_data h5ad → `results/ILC2_studies_umap/`的PNG/PDF、坐标、HVG与参数 | `python main/plot_ilc2_studies_umap.py` |
| `plot_saved_gse117567_clusters.py` | 按已保存的真实Leiden标签重绘UMAP并统计cluster×组织，不重算聚类/坐标 | 同一结果目录的 `GSE117567_umap_hvg.h5ad` → 聚类PNG/PDF和 `cluster_tissue_counts.tsv` | `python main/plot_saved_gse117567_clusters.py` |
| `compare_ilc2_qc_standard.py` | 新共享身份规则在GSE117567既有基础QC对象上的数量对照；不覆盖历史训练对象，不冒充完整重跑 | 旧annotated h5ad → `results/ILC2_qc_standard/GSE117567_identity_comparison.tsv` | `python main/compare_ilc2_qc_standard.py` |
| `audit_gse153748_signatures.py` | 只读审查作者亚群差异基因Excel，整理比较列表并评估QC参考；不修改细胞 | 原始signature xlsx → `results/GSE153748_qc/signature_reference/` | `/opt/anaconda3/bin/python main/audit_gse153748_signatures.py`（已有openpyxl） |
| `merge_ilc2_dataset_table.py` | 以「数据更新」样本表为底，补入「研究汇总」中缺少的18个研究（研究级一行、描述以研究汇总为准），分为稳态/外源处理/遗传扰动三类，并标记是否为单细胞RNA | `data/metadata/ILC2_new_datasets_metadata_2026-10-13.xlsx` + `ILC2_单细胞数据与metadata汇总_2026-10-12.xlsx` → `data/metadata/ILC2_datasets_merged_2026-10-15.csv` | `python3 main/merge_ilc2_dataset_table.py` |
| `traindata_qc.ipynb` | **交互式基础QC示例**；不要把它与新身份筛选结果混淆 | raw + Ensembl GTF → basic_qc样本对象 | 在 `main/` 启动 Jupyter 并打开；会写基础QC输出 |

## 历史脚本归档

这些脚本原来散在项目旁的 `../ILC2_rotation_proposal/`，现集中到 `main/archive/`。历史结果保持原路径以避免重复复制；脚本输入和输出路径已修复，未为了整理而重跑旧分析。

| 文件 | 用途 | 输入与原结果位置 |
|---|---|---|
| `archive/inspect_counts.py` | 最初5个GEO矩阵只读摸底、文件校验、QC分布与marker检出 | 当前 `data/GSE117567_RAW/` → `../ILC2_rotation_proposal/GSE117567_initial_inspection/` |
| `archive/prepare_workbook_inputs.py` | scBaseCount/ENA审计表汇总成Excel中间JSON；不是联网抓取脚本 | `../ILC2_rotation_proposal/scbasecount_audit/` 的已有TSV/SOFT → 同目录 `workbook_inputs.json` |
| `archive/check_environment.py` | Scanpy真实骨髓读取＋随机小GeneCompass模型CPU/MPS冒烟检查；**没有验证正式checkpoint** | 当前raw、旁边 `software/GeneCompass` → `../ILC2_rotation_proposal/environment/` |

历史脚本可用 `python main/archive/<文件名>.py` 运行，但可能覆盖其旧统计/报告，普通工作不需要重跑。移动和路径修改的旧/新SHA256记录在 `../docs/script_relocation.json`；历史provenance中的源码hash仍指原运行版本，不静默替换。

## 正式流程与检查（不移动）

| 文件 | 用途与实现状态 |
|---|---|
| `../src/qc_utils.py` | 可复用读取、Ensembl注释、逐样本基础QC及已复核ILC2选择工具 |
| `../src/ilc2_qc.py` | 共享ILC2多基因证据及显式apply_standard统一预设；历史调用语义保留 |
| `../src/qc_gse153748.py` | GSE153748统一balanced_v1、Scrublet仅标记、细胞级身份筛选与作者状态；运行 `python src/qc_gse153748.py`，输出 `data/train_data/GSE153748.h5ad`、独立BM对象和 `results/GSE153748_qc/` |
| `../src/1QC.py` | 正式基础QC入口 `--basic`；`--legacy-gse117567`已改为调用main中的旧脚本 |
| `../src/2prepare_traindata.py` | 数据划分与GeneCompass编码；当前设计骨架，未实现 |
| `../src/3finetune_FM.py` | 官方权重继续训练；当前设计骨架，未实现 |
| `../src/4downstream.py` | 表征与扰动任务；当前设计骨架，未实现 |
| `../tests/check_scaffold.py` | 四阶段CLI、配置、旧入口转发、目录和忽略规则检查 |
| `../tests/test_qc_utils.py` | QC边界及计数保留检查 |
| `../tests/test_ilc2_selection.py` | 多marker身份边界：ST2阴性、Rorc单独表达、混合谱系等 |
| `../tests/test_gse153748_qc.py` | 统一宽松边界：dropout/低RNA/混合群/双细胞高分不独立删除，强替代谱系/腺泡/红系排除 |

## 目录规则

`src/`正式流程；`main/`探索和临时程序；`configs/`参数；`data/`原始/参考/处理/训练数据；`results/<研究_分析>/`图、汇总、报告、参数和日志；`model/`权重；`docs/`项目设计与任务记录；`tests/`验证。

第三方GeneCompass checkout与其他实验的 `../test1009/` 不属本项目自写探索脚本，保持原状。早期proposal资料与下载审计结果保留原目录；没有删除或复制原始矩阵。后续一次性命令只要涉及分析或生成交付数据，应先写成main中的脚本，写清输入/输出/用途，结果不得散落项目根目录。

## 当前 UMAP 约定

输入是QC和身份筛选后的25,010细胞，四个组织，无骨髓。每万UMI归一化→log1p→2500 HVG→30 PCA→15近邻→UMAP（seed=0、min_dist=0.5）；不做batch correction。每组织仅一个来源文库，组织与文库效应混杂，UMAP距离/分群不能直接当作功能差异或独立重复证据。

训练h5ad不改动；`GSE117567_umap_hvg.h5ad`只是HVG可视化对象，X为log归一化值、counts层为所选HVG的原始UMI，不能替换模型输入。图、坐标、HVG表、参数与源文件hash同目录保存。
