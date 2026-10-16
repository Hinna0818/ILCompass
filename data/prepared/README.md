# 阶段 2 输出

按 `<dataset_version>/<split>/` 保存编码数据，另存行对应的 cell ID、sample、split、有效 length、coverage 与版本/hash。每条 length 必须和同一行细胞匹配。

该阶段待实现；词表和参考统计必须先匹配 checkpoint。大型结果不纳入 Git。
