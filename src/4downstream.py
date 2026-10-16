"""阶段 4：表征评价、in silico 基因删除与有监督扰动任务。

设计骨架；基因删除的表征偏移不等于真实 KO 表达预测。
"""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/4downstream.json", help="预留配置路径；尚未执行")
    parser.parse_args()
    parser.error("阶段 4 尚未实现；监督预测需独立的匹配 control/perturbation 数据，详见 CLAUDE.md。")
