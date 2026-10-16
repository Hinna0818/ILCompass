# ILC2基础模型的三个下游接口

设计日期：2026-10-16。假设已取得可用的小鼠ILC2 post-trained GeneCompass；本文为设计，不表示模型已训练或下游数据已QC。

## 推荐决策

共用冻结的ILC2编码器，独立训练状态头和条件响应头；基因token删除单独作为敏感性分析。先完成状态分类、embedding扰动及低维表达响应，再视条件和重复数决定是否做单细胞分布生成。无需同时训练三个任务或先建设复杂新骨干。

| 任务 | 新增模块 | 最少需要的数据 | 当前定位 |
|---|---|---|---|
| token删除后的embedding变化 | 无，可选已锁定状态分类器 | 正常ILC2；真实KO用于外部核验 | 可优先做，不能直接称真实KO预测 |
| KO后表达变化 | 基因条件编码器+响应解码器 | 多目标真实遗传扰动及control | ILC2数据暂不足以支持通用未知KO |
| 刺激/药物后表达变化 | 条件编码器+响应解码器 | 有control的多种真实处理、时间/剂量和重复 | 可先做神经肽/细胞因子小任务 |
| 状态分类/评分 | 线性或小MLP头 | 有来源和定义的状态标签 | 最先实施；区分处理标签与功能状态 |

这里的共享是编码接口复用。每项任务独立划分、训练、选模，避免一个任务的标签泄漏到另一项测试。

## 共同输入输出约定

- 原始输入：整数counts、稳定基因ID和metadata；GeneCompass使用匹配checkpoint的token及表达编码。decoder的表达目标与骨干编码单独保存，不直接解读token顺序为固定基因坐标。
- 编码器输出：cell embedding `z: [B,D]`；以及有效token的上下文gene embedding `H: [B,L,D]`，保留每个位置的gene ID、padding和截断信息。Small配置D=256，其他checkpoint以实际配置为准。
- cell pooling使用经过验证的CLS或非padding平均，训练/测试一致。token删除比较额外报告在两条序列共同保留的基因上pooling的结果，以检查被删除基因本身的平均值贡献。
- condition记录perturbation类型、target或compound/stimulus、剂量及单位、时间、组织和给药途径。缺失剂量/时间不是0；不把文库ID作为可迁移生物条件。
- `study/sample/animal_or_pool/replicate/control_id/role/split`必须可追溯。共享control导致的依赖不能当多个独立重复。
- 主输出先选训练来源确定的约1,000–2,000个基因或10–30个预先定义程序；范围为建议，随有效独立条件数缩小。不能在最终测试集选HVG、DEG或程序阈值。

## 1. 基因虚拟扰动

### 1A. 不训练新decoder：embedding敏感性

借鉴[Geneformer论文](https://www.nature.com/articles/s41586-023-06139-9)及[作者in silico接口](https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html)。

```text
原细胞token序列 ─────────→ 固定编码器 → z、H
同一序列删除目标gene token → 固定编码器 → z_del、H_del
                                         ↓
                        cell/gene偏移、目标状态相似度变化
```

第一版采用token deletion，不把mask token当永久敲除。删除后同步更新长度/mask，并固定是否补入被截断基因的策略；建议不补入，避免额外基因成为混杂。目标未进入原序列时记为not_testable，报告覆盖率，不能把无变化当目标不重要。

读出：`1-cos(z,z_del)`衡量表征敏感度；共同基因的上下文向量偏移用于候选相关程序排名；`cos(z_del,mu_goal)-cos(z,mu_goal)`观察是否靠近指定状态。目标状态中心仅在允许的训练/验证参考中建立，最终测试case不参与指导候选筛选。也可把z_del输入已锁定状态分类器，输出分数变化；这种变化仍是分类器在修改输入上的模型读出。

对照：按检出率、表达量和原始token排名匹配的随机基因删除；比较同一基因在不同组织/重复的稳定性。绝对位置变化、序列缩短和表达排序都会影响结果。Cell embedding偏移大不表示促炎、抑炎或必需，方向必须依赖预先定义的参照。

可先测试Gata3、Rora、Il1rl1、Nmur1、Dusp2等候选，但必须报告每个基因进入token序列的比例，不预设模型一定复现文献机制。预测新的调控候选需要独立扰动实验或数据支持，embedding关联不等于调控因果边。

### 1B. 真实KO后的表达响应

借鉴[GEARS](https://www.nature.com/articles/s41587-023-01905-6)的“细胞基线+目标基因信息→扰动响应”，不在小数据上直接复制整套图网络。

```text
control细胞/细胞池 → GeneCompass → z0
目标基因向量 + KO/CRISPRi/CRISPRa类型 → 小投影层 → p
z0、p、组织/时间 → 小响应网络 → 基因程序变化或固定基因面板变化
```

条件向量可来自冻结的gene token向量或预先锁定的外部基因功能特征，经小投影层生成32–64维p。这是新目标的输入先验，不足以独自提供其响应标签。若只用target one-hot，则未训练的新目标没有可学习的映射。

对独立条件均值，先预测训练集拟合的低秩响应系数（例如16–32维，上限不能超过实际训练响应的秩），再投影到固定基因面板。与Ridge、无变化、平均响应比较。几种KO只能做特定目标案例，数千细胞不等于数千种KO。

只有基线数据训练的重建头`D(E(x))≈x`学的是观测表达关系。`D(E(delete_g(x)))`可作为探索性表达敏感度，但未经过真实KO监督前不能解释为KO后的反事实表达谱。GeneCompass原masked-expression头也不自动具备这项能力。

KO、抑制和激活必须区分。靶基因失去蛋白功能时RNA可仍检出，不强制所有KO目标RNA=0；不把输入值设0等同永久遗传缺失。下游QC不能因被敲掉的身份marker变低而自动去掉真实受影响细胞；需审查选择偏差、存活偏差与细胞组成改变。

候选案例：[GSE272724](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272724)为WT与Pac1/Dusp2 KO，两个sc样本记录，原论文记载4–5只鼠混样；可检查Dusp2删除的程序一致性，不能作为未知KO泛化的充分训练集或独立重复检验。Pac1/Dusp2不是PACAP受体Adcyap1r1。[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11527444/)还提供bulk入口GSE272723，但需逐项核对刺激和时间后再配用。

## 2. 刺激/药物条件响应与生成

### 推荐第一版：条件均值响应

借鉴[scGen](https://www.nature.com/articles/s41592-019-0494-8)的潜空间变化和[CPA](https://doi.org/10.15252/msb.202211517)的可组合条件表示，保留GeneCompass作为基线编码器。

定义一个真实生物样本/混样池的control表示均值`zbar0`和control表达目标`y0`，小MLP接收`[zbar0,p,tissue,time,dose]`预测`delta_y`，输出`y0+delta_y`。第一版输出的是条件预期变化，不声称生成单细胞异质性。输入中的缺失协变量用明确缺失标记处理；对只有一个取样时间的研究不声称学得时间规律。

目标尺度可选sample/pool内原始counts聚合后标准化得到的log-pseudobulk，或预先定义的程序分数。不要把每细胞log表达均值写成真正的counts pseudobulk。小数据优先Huber/MSE回归delta；响应基底、基因选择及任何标准化拟合只使用训练来源。

### 若确需生成单细胞分布：条件VAE

```text
匹配control细胞池 → 固定GeneCompass → zbar0
真实处理细胞y ──→ 小posterior网络 q(u|y,zbar0,p) → u
训练：zbar0 + p + u → 小decoder → 表达分布参数
推理：zbar0 + p + u~N(0,I) → 生成处理细胞集合
```

训练时用处理细胞构造posterior；推理不使用处理细胞。以原始UMI为目标时decoder输出负二项分布的均值比例/离散度，均值包含明确的library-size offset；深度来自预先规定的参考分布或control，不从测试处理组偷取。生成counts不是绝对每细胞RNA含量或分子动力学模拟。若用log标准化表达则采用相应连续分布，不能对其套负二项似然。

基本loss为`negative log likelihood + beta*KL`。可以按真实样本条件增加生成与实测的均值/程序差异；有明确分布评价需求后再增加MMD等项，不一开始堆叠多个loss。模型学习的是条件人群分布：不同control和treated细胞未一一配对，不能随机配对后逐细胞MSE，并把它叫个体反事实监督。CVAE不保证识别同一个细胞的真实反事实轨迹。

### 条件编码与泛化

- 已见刺激：多热编码或小embedding，组合先用单条件向量相加；残差网络对基线依赖作调整。
- 未见组合：全部组合样本留出，但组分可在训练出现。只见单刺激时，非加和相互作用不可识别；只一个测试组合是案例，不是通用组合泛化结论。[GEARS官方说明](https://github.com/snap-stanford/GEARS)明确提示仅单基因训练不足以可靠预测组合。
- 小分子新药：参考[chemCPA](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html)，用固定分子指纹/结构表示或靶点先验取代纯drug ID，经小MLP编码。按药物身份乃至结构簇留出；这提供外推输入，不保证新药或新细胞类型的预测有效。
- 细胞因子/神经肽：IL-33、NMU、CGRP不按小分子Morgan指纹编码。当前条件少时用身份和组合编码即可，不额外训练蛋白语言模型。剂量需有真实范围和单位，时间/途径不能混成一个刺激ID。
- 外部细胞系Perturb-seq或药物屏可预训练响应模块，但必须映射基因/物种并保留来源；不能当已验证的ILC2响应，更不能用测试ILC2条件微调后声称零样本迁移。

### 可以落地的数据和任务

[GSE136154](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136154)：PBS、CGRP、IL33、IL33+CGRP，每组两个sc生物重复，共8个filtered H5；需从ILC中确认ILC2。可做已知刺激跨重复预测；或锁定全部组合样本，用PBS/两个单刺激拟合后预测组合。只有两种刺激、一个组合，第一版以均值/程序响应和相加基线为主，CVAE仅探索，不能包装成广泛药物生成模型。

[GSE102299](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE102299)：PBS、IL25、IL33、NMU、NMU+IL25及HDM相关实验；10x与SMART-seq2并存，优先10x。可扩展已知刺激预测、留出NMU+IL25，或研究间IL33迁移。不同protocol、平台、时间和给药途径先核实，不能简单拼接条件名。

GSE136154 bulk体外有32条记录，含2生物重复×2技术重复×4条件×2时间；sc是体内单时间。合并技术重复后也不代表与sc完全匹配，主要作背景受限的响应方向/程序核验，不直接报告为同条件外部表达真值。纯化ILC2 bulk可用于均值响应监督/验证，组织bulk不能直接替代；bulk不能验证生成细胞分布的真实性。

## 3. 状态分类与功能评分

参考[scGPT](https://www.nature.com/articles/s41592-024-02201-0)的任务微调范式，第一版只训练线性头或`D→64→K`的MLP，先冻结骨干；少量解冻是否有效由同一验证划分比较。

```text
当前细胞表达 → 固定GeneCompass → z → 分类/评分头
                                  ├─处理类别概率
                                  ├─非互斥功能标签概率
                                  └─连续程序分数
```

- 互斥处理类别：softmax+交叉熵；研究的是样本处理标签能否迁移，不等于每个细胞都实际响应。
- 非互斥效应/修复/IFN/增殖等状态：多sigmoid+BCE；一个细胞可有多个状态，不强迫归为一个类。
- 连续程序活性：Huber/MSE回归，有噪声的signature分数作为计算标签。
- tissue分类可作独立辅助任务/混杂诊断，不能代替功能状态，不把组织、平台和文库差异一概叫状态。

GSE153748作者signature提供内部弱标签，训练后复现同套marker只能证明规则拟合。GeneCompass输入也含这些基因；必须比较直接signature分数，不能宣称独立发现功能。跨研究迁移、不同基因集评价或隐去定义标签基因可诊断marker捷径，但都不自动产生实验真值。处理数据提供条件标签，蛋白/功能实验或独立作者注释提供更强证据，仍需说明标签定义。

GSE102299与GSE136154共同IL33/PBS条件可做跨研究处理状态识别；其他条件用作预先指定的未知输入检查。未知类别不能强制解释为最接近的已知功能类；以验证数据校准拒判阈值或明确无校准置信度，softmax高分不是生物学确定性。

分类当前细胞与预测处理后状态是两件事：后者需经过已验证条件响应模型，再把生成表达重新按GeneCompass编码后送入同一分类头。直接对未验证的合成latent分类，只能报模型推测。

## 4. 训练、划分与评价

1. 锁定最终测试研究及其case/control；不得用于post-training、下游调参、特征拟合或候选基因排序。测试control允许在锁定后的推理阶段作为基线输入。
2. 按研究、真实animal/pool和扰动身份划分；跨重复、未见组合、未见靶点/新药分别评价。少于所需独立来源时直接标明内部探索。仅两重复的任务无法同时提供可靠train/val/test三套生物来源。
3. 先做PCA+线性/Ridge、原始GeneCompass+同一任务头、post-trained GeneCompass+同一头。响应另含零变化、训练平均响应、单刺激相加、Systema matching mean。训练预算及输入信息一致。
4. 分类：macro-F1、balanced accuracy、逐类混淆矩阵和按样本表现；多标签用逐类AUROC/AUPRC；signature弱标签另报直接规则基线。不要用同文库随机拆分的高准确率证明跨实验泛化。
5. 表达：逐条件delta RMSE/MAE、方向、预先固定程序；相关性仅辅助。报告靶基因以外的响应，避免目标置0造成虚假高分。测试DEG只在锁定后用于独立的描述性评价，不参与选模。
6. 生成分布：条件均值之外再比较方差、状态比例、固定空间的MMD/energy distance及重复间实测距离；低均值误差不等于生成异质性正确。
7. 不确定性按独立sample/condition汇总；cell bootstrap只表示池内抽样波动，不能制造生物重复。少量重复无法支撑精确群体置信区间。[Systema](https://www.nature.com/articles/s41587-025-02777-8)提示需防止把共同处理效应当扰动特异预测能力。

## 5. 实施优先级与交付

先提取并锁定embedding，做状态头和虚拟删除；再对GSE136154做条件均值响应与组合留出，GSE102299用于扩展/研究迁移。KO表达生成只在真实遗传扰动条件足够时开启；Dusp2案例用于机制一致性探索。小分子生成先查到足够ILC2药物标签再定义训练，不将神经肽数据改称小分子药物屏。

未来可由现有`src/4downstream.py`调用最少的共享头实现；本轮不新增空代码模块、CLI参数或训练配置。产出锁定split清单、embedding与条件ID、逐样本/逐扰动预测、基线表、原始/post-trained消融、机制候选与限制。checkpoint保存task定义、骨干版本、gene顺序、条件词表、目标变换、split及优化状态。

本设计未运行模型或下载候选扰动矩阵。生物学机制已知证据、模型一致性结果与新假设必须分别记录。
