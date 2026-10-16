# 模型资产

- `pretrained/`：官方 checkpoint 与配套 config/资产来源。
- `posttrained/`：阶段 3 的 ILC2 权重、超参数、数据版本和日志。
- `downstream/`：有监督下游网络或 decoder。

本仓库尚未下载/加载实际官方 checkpoint，也未完成正式训练。当前 Mac 小型随机模型测试不证明正式模型训练可行。

来源：[GeneCompass 官方仓库](https://github.com/xCompass-AI/GeneCompass#download-checkpoints)，[官方 checkpoint 下载入口](https://www.scidb.cn/en/anonymous/SUZOdk1y)，[先验文件](https://drive.google.com/drive/folders/1a8jhpyw4jmAifK6Ui6u1RJtRFlbsdx08?usp=sharing)。

需检查先验 embedding、ID 映射是否为实际文件而非 Git LFS 指针，并核验 checkpoint、词表与 config 的一致性。权重不纳入 Git；发布模型前确认授权，保留官方归属与来源。
