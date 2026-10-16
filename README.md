# ILCompass

基于 GeneCompass 探索小鼠 ILC2 表征学习与下游扰动分析的轮转项目。

目前包含单细胞数据 QC、ILC2 候选筛选及探索性分析代码。数据划分与编码、正式 post-training 和下游模型仍在设计中，尚未实现完整训练流程。

## 目录

- `src/`：正式流程入口与共享 QC 工具。
- `main/`：数据集专用、探索及历史脚本，见 [脚本索引](main/README.md)。
- `configs/`：QC 参数与后续阶段的设计配置。
- `data/metadata/`：样本清单、公开来源和文件校验记录。
- `docs/`：QC 说明、模型设计与分析计划。
- `tests/`：流程骨架和 QC 边界检查。

## 环境与检查

`environment.yml` 与 `requirements-lock.txt` 记录本地 macOS 环境；Linux/CUDA 服务器需另行核验平台依赖。

```bash
conda env create -f environment.yml
conda activate ilc2_model
python tests/check_scaffold.py
python src/1QC.py --help
```

已有同名环境时无需重复创建。处理真实数据前需准备原始矩阵、参考注释等输入；见 [QC 使用说明](docs/qc_usage.md)。GeneCompass 源码、checkpoint 和配套资产另行准备，见 [模型资产说明](model/README.md)。

## 数据与当前边界

首批提交代码、配置、必要文档及小型来源表，不包含原始/处理后矩阵、模型权重、分析结果、电子表格或 notebook。历史分析结果与部分探索输入仍在本地，克隆仓库不等于获得完整可运行的数据环境。

ILC2 候选身份需要结合注释复核；计算筛选不等于实验验证。模型设计见 [设计文档](docs/model_design.md)。

项目许可待确定；第三方代码、模型和数据遵循各自授权与引用要求。
