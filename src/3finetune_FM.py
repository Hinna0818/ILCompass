"""阶段 3：从官方 GeneCompass checkpoint 继续 ILC2 自监督训练。

设计骨架；输入输出见 CLAUDE.md 和 configs/3finetune_FM.json。
"""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/3finetune_FM.json", help="预留配置路径；尚未执行")
    parser.parse_args()
    parser.error("阶段 3 尚未实现；需先准备实际预训练权重、匹配先验和阶段 2 的数据。")
