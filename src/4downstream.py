"""Run an ILCompass perturbation analysis from its existing task entry point."""
import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "cytokine": "src/downstream/drugpred/train_combination_representation_3seeds.py",
    "gene-ko": "compare_pretrain/model_compare/gene_ko/compare_models.py",
    "screen-il13": "compare_pretrain/model_compare/crispr_il13/direction_eval/run.py",
    "screen-gata3": "compare_pretrain/model_compare/crispr_gata3/direction_eval/run.py",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use TASK --help for task-specific arguments.")
    parser.add_argument("task", choices=TASKS)
    task = parser.parse_args(sys.argv[1:2]).task
    os.execv(sys.executable, [sys.executable, str(ROOT / TASKS[task]), *sys.argv[2:]])
