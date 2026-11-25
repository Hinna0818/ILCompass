# ILCompass

ILCompass 研究小鼠 2 型固有淋巴细胞（ILC2）在细胞因子刺激和基因扰动下的转录响应。项目比较三种细胞编码器，并用公开的 CRISPR 筛选数据评估模型能否预测扰动效应的方向。

## 三种模型

| 模型 | 训练方式与输入 | 实现 |
|---|---|---|
| **Raw GeneCompass** | 冻结原始人鼠 GeneCompass Base，作为基线。使用官方词表、固定的小鼠表达中位数代理和最多 2,048 个基因 token。 | [权重加载](src/genecompass_compat.py) · [共享编码入口](compare_pretrain/model_compare/shared.py) |
| **LoRA GeneCompass** | 冻结同一 Base，仅训练 12 层注意力的 query/value 低秩适配器（rank 8）。语料含 113,159 个细胞，覆盖六项研究、七种组织。推理时同时加载 Base 和适配器。 | [适配器](compare_pretrain/genepass_lora/model.py) · [训练](compare_pretrain/genepass_lora/train.py) · [配置](compare_pretrain/genepass_lora/config.json) |
| **Scratch v3** | 在小肠、胰腺、肺和骨髓的 96,426 个细胞上从头训练 8 层、384 维 Transformer，联合重建被遮蔽的基因身份和表达值。使用自己的词表、训练集表达中位数和细胞划分。 | [模型](scratch/src/model_v3.py) · [训练](scratch/src/train_v3.py) · [配置](scratch/configs/training_v3.json) |

## 三类扰动任务

| 任务 | 数据、比较与读出 | 主要脚本 |
|---|---|---|
| **细胞因子扰动（cytokine perturbation）** | 从对照细胞预测刺激后的群体表达变化。IL-33 实验比较批次、研究或组织之间的迁移；联合刺激实验在 GSE136154（IL-33 + CGRP）和 GSE102299（IL-25 + NMU）比较三种编码器及表达基线，分别评估 0% 和 5% 联合刺激监督。 | [IL-33 LoRA](compare_pretrain/model_compare/il33_prediction/run.py) · [IL-33 Scratch](src/downstream/drugpred/train_il33_scratch_v3_3seeds.py) · [联合刺激](src/downstream/drugpred/train_combination_representation_3seeds.py) |
| **基因扰动（in silico KO）** | GSE279338 肺 ILC2 的 Piezo1 条件敲除实验。在同一组 WT 细胞上模拟降低目标基因表达，与真实 KO 参考细胞及匹配的随机基因对照比较，报告细胞表征的余弦相似度变化。 | [准备配对与对照](compare_pretrain/model_compare/gene_ko/run.py) · [三模型评估](compare_pretrain/model_compare/gene_ko/compare_models.py) |
| **CRISPR screening** | 用 Raw 和 LoRA 在 GSE126924 的肺 ILC2 参考细胞上预测读出变化，再与 GSE262028 的分选筛选比较。Il13 使用 S17/S25 报告基因分选，Gata3 使用 S31/S35 谱系分选。结果包括逐基因方向准确率、上下调召回率和相关性。 | [整理筛选计数](src/downstream/geneko/prepare_gse262028_screen_inputs.py) · [Il13](compare_pretrain/model_compare/crispr_il13/direction_eval/run.py) · [Gata3](compare_pretrain/model_compare/crispr_gata3/direction_eval/run.py) |

GSE202186 是另一项小鼠免疫细胞因子基准。它把 PBS 细胞的 Raw GeneCompass 表征与细胞因子的 ESM2 表征输入 FiLM 预测头，输出相对 PBS 的逐基因表达变化。三模型 ILC2 刺激比较与这项多细胞类型评估使用各自的数据和划分。[FiLM 模型](src/immune_dictionary/film.py) · [训练与折外评估](src/immune_dictionary/train.py) · [按条件预测](src/immune_dictionary/predict_condition.py)

KO 任务衡量预测表征向真实 KO 表征移动的程度。CRISPR 任务以筛选重复中的相对富集方向为实验参照；Gata3 分选还反映谱系组成，解释时需与成熟 ILC2 的单细胞表达变化区分。

## 运行前准备

使用 Python 3.10。项目依赖见 [requirements.txt](requirements.txt)。在项目根目录将上游 GeneCompass 检出到 `external/GeneCompass/` 并加入 Python 搜索路径。固定版本和第三方来源见 [external/README.md](external/README.md)。

```bash
git clone https://github.com/xCompass-AI/GeneCompass.git external/GeneCompass
git -C external/GeneCompass checkout 59e5e485cb940038a7db040c2f31c8a2ce76ba53
export PYTHONPATH="$PWD/external/GeneCompass${PYTHONPATH:+:$PYTHONPATH}"
```

本仓库发布源码、配置及小型参考表。运行正式推理前，将对应权重和数据按下表恢复至项目根目录下。自行提取蛋白特征时还需安装 `fair-esm` 并准备 ESM2 权重。

| 资产 | 预期位置 |
|---|---|
| Raw GeneCompass Base | `model/pretrained/GeneCompass_base/{config.json,pytorch_model.bin}` |
| LoRA 适配器 | `results/ILC2_GeneCompass_LoRA_v1/r8_seed0/best.pt` |
| Scratch v3 权重与配套输入 | `scratch/results/ilc2_scratch_v3/E3_8L384_MLM_seed0/best.pt` 和 `scratch/data/prepared/ilc2_scratch_v3/` |
| GeneCompass 编码参考 | `external/GeneCompass/scdata/dict/human_mouse_tokens.pickle` 和 `data/references/mouse_gene_median.pickle`。小鼠中位数代理的生成逻辑见 [映射脚本](main/build_mouse_gene_median.py) |
| 研究输入 | 各研究带整数 counts、稳定基因 ID 和来源元数据的 QC 对象。下游配置另列固定 split、基因顺序、输入缓存及历史对照。QC 标准见 [ilc2_qc_standard.md](docs/ilc2_qc_standard.md) |

## 从哪里运行

查看下游入口和任一任务的参数：

```bash
python src/4downstream.py --help
python src/4downstream.py cytokine --help
python src/4downstream.py gene-ko --help
python src/4downstream.py screen-il13 --help
python src/4downstream.py screen-gata3 --help
```

准备好各自配置列出的输入后，可按任务运行：

- **联合刺激**从已锁定的细胞划分和单刺激任务缓存开始。显式选择含 Scratch v3 的[三种子配置](configs/combination_representation_scratch_v3_3seeds_v1.json)，先执行 `audit` 核对输入，再执行 `all`。
- **Piezo1 KO** 先用 `python compare_pretrain/model_compare/gene_ko/run.py prepare` 固定 WT、KO 参考和随机对照。再用 `gene-ko prepare` 建立三个模型的共同评估队列，逐模型 `extract` 后运行 `score`。
- **CRISPR** 先整理 GSE262028 的筛选计数，再生成 GSE126924 参考细胞及冻结对照清单；准备顺序见下方命令。两个方向评估均支持 `--check-only`。正式推理将各模型的逐基因预测与统计写入对应的 `results/` 子目录。

```bash
python src/4downstream.py cytokine audit --config configs/combination_representation_scratch_v3_3seeds_v1.json
python src/4downstream.py cytokine all --config configs/combination_representation_scratch_v3_3seeds_v1.json

python src/downstream/geneko/prepare_gse262028_screen_inputs.py
python src/downstream/geneko/audit_il13_screen_tf_coverage.py
python src/downstream/geneko/prepare_il13_screen_design.py
python src/4downstream.py screen-gata3 --prepare-only
python src/4downstream.py screen-il13 --check-only
python src/4downstream.py screen-gata3 --check-only
```

[固定评估脚本](compare_pretrain/model_compare/test_reconstruction/run.py)可比较相同遮蔽目标上的基因身份与表达重建。轻量测试覆盖 Scratch v3 的损失与恢复，以及 CRISPR 方向指标对零效应和不确定预测的处理：

```bash
python scratch/tests/test_v3_training.py
python compare_pretrain/model_compare/crispr_il13/direction_eval/test_direction_metrics.py
```
