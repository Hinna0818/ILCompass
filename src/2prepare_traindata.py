"""阶段 2：合并 ILC2 counts、锁定样本划分、按 checkpoint 编码。

设计骨架；输入输出见 CLAUDE.md 和 configs/2prepare_traindata.json。
"""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/2prepare_traindata.json", help="预留配置路径；尚未执行")
    parser.parse_args()
    parser.error("阶段 2 尚未实现；需先确认高可信 ILC2、数据划分及 checkpoint 配套词表/归一化统计。")
