"""只检查入口/配置；不再执行默认命令，避免正式训练被旧骨架检查启动。"""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    scripts = ["1QC", "2prepare_traindata", "3finetune_FM", "4downstream", "evaluate_posttrain"]
    for script in scripts:
        path = ROOT / "src" / f"{script}.py"
        ast.parse(path.read_text())
        result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    prep = json.loads((ROOT / "configs/2prepare_traindata.json").read_text())
    train = json.loads((ROOT / "configs/3finetune_FM.json").read_text())
    assert prep["dataset_version"] == train["dataset_version"]
    assert prep["prepared_dataset_path"] == train["prepared_dataset_path"]
    assert prep["split_fractions"] == [.9, .05, .05]
    assert train["initialize_from_pretrained"]
    assert train["checkpoint_path"] == "model/pretrained/GeneCompass_base"
    assert train["loss_mode"] in {"normalized", "official"}
    print("PASS: CLI/配置一致；未运行 QC、准备、训练或 test。")


if __name__ == "__main__":
    main()
