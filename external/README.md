# 第三方源码

其他机器可将 [GeneCompass](https://github.com/xCompass-AI/GeneCompass) checkout 至 `external/GeneCompass`，固定提交 `59e5e485cb940038a7db040c2f31c8a2ce76ba53`，并将该目录加入 PYTHONPATH。checkout 内容不纳入本项目 Git。

本机现有 `ilc2_model` 环境已经配置了另一位置的 GeneCompass 导入，不需要再次下载。迁移项目后须更新导入路径，并保留第三方 LICENSE 与引用。普通 git clone 不保证取得 LFS 实际先验。
