"""检查流程骨架与入口，不读取矩阵、不下载资产、不启动训练。"""
import ast
import csv
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    stages = ["1QC", "2prepare_traindata", "3finetune_FM", "4downstream"]
    configs = {}
    for stage in stages:
        script = ROOT / "src" / f"{stage}.py"
        ast.parse(script.read_text())
        configs[stage] = json.loads((ROOT / "configs" / f"{stage}.json").read_text())
        assert configs[stage]["status"] == ("basic_qc_implemented_identity_pending" if stage == "1QC" else "design_only_not_executed")
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert help_result.returncode == 0, help_result.stderr
        default_result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert default_result.returncode == 2
        assert ("required" in default_result.stderr) if stage == "1QC" else ("尚未实现" in default_result.stderr)
    qc = configs["1QC"]["basic_qc"]
    assert qc["min_detected_genes"] == 200
    assert qc["min_cells_per_sample"] == 4
    assert qc["min_detected_protein_coding_or_miRNA"] == 7
    assert qc["max_mt_pct_inclusive"] == 15
    assert not configs["1QC"]["validation"]["strict_subset_enabled"]
    assert configs["2prepare_traindata"]["holdout_complete_test_study_including_control"]
    assert configs["3finetune_FM"]["initialize_from_pretrained"]
    assert not configs["4downstream"]["in_silico_deletion_is_validated_KO_prediction"]

    # 验证旧流程的参数确实被传递，而非误触发默认真实分析。
    saved_runner = runpy.run_path
    legacy = ROOT / "main/qc_gse117567.py"
    forwarded = ["--input", "example", "--finalize", "--review", "review.json"]
    seen_argv = []
    with patch.object(sys, "argv", ["1QC.py", "--legacy-gse117567", *forwarded]), patch("runpy.run_path") as runner:
        runner.side_effect = lambda *args, **kwargs: seen_argv.append(sys.argv.copy())
        saved_runner(str(ROOT / "src/1QC.py"), run_name="__main__")
        runner.assert_called_once_with(str(legacy), run_name="__main__")
        assert seen_argv == [[str(legacy), *forwarded]]
    ast.parse(legacy.read_text())

    with (ROOT / "data/metadata/sample_manifest.tsv").open() as fh:
        samples = list(csv.DictReader(fh, delimiter="\t"))
    assert len(samples) == 5 and len({row["sample"] for row in samples}) == 5
    assert sum(int(row["raw_cells"]) for row in samples) == 35396
    assert all(row["split"] == "unassigned" and row["role"] == "unassigned" for row in samples)
    with (ROOT / "data/metadata/raw_file_manifest.tsv").open() as fh:
        files = list(csv.DictReader(fh, delimiter="\t"))
    assert len(files) == 15 and all(len(row["sha256"]) == 64 for row in files)
    for directory in ["src", "main", "main/archive", "configs", "data/raw", "data/metadata", "data/references", "data/processed", "data/prepared", "model/pretrained", "model/posttrained", "model/downstream", "external", "results", "notebooks", "docs", "tests"]:
        assert (ROOT / directory).is_dir(), directory
    for filename in ["environment.yml", "requirements.txt", "requirements-lock.txt", "README.md", "CLAUDE.md", ".gitignore"]:
        assert (ROOT / filename).is_file(), filename
    assert "-r requirements-lock.txt" in (ROOT / "environment.yml").read_text()

    # 使用临时 Git 元数据检查真正的忽略语义，不在项目目录初始化仓库。
    with tempfile.TemporaryDirectory(prefix="ilc2_scaffold_") as tmp:
        subprocess.run(["git", "init", "--quiet", tmp], check=True)
        command = ["git", f"--git-dir={Path(tmp) / '.git'}", f"--work-tree={ROOT}", "check-ignore", "--no-index", "--quiet"]
        ignored = ["data/GSE117567_RAW/test.mtx.gz", "data/processed/test.h5ad", "model/pretrained/weights.safetensors", "model/posttrained/run/weights.safetensors", "external/GeneCompass/README.md", "results/run/cell_embeddings.npy"]
        tracked = ["src/1QC.py", "main/build_train_gse117567.py", "main/README.md", "configs/1QC.json", "data/metadata/sample_manifest.tsv", "data/raw/README.md", "model/README.md", "model/posttrained/run/config.json", "results/GSE117567_qc/QC_report.md", "external/README.md"]
        for relative in ignored:
            result = subprocess.run([*command, relative], cwd=ROOT)
            assert result.returncode == 0, f"未忽略大型资产：{relative}"
        for relative in tracked:
            result = subprocess.run([*command, relative], cwd=ROOT)
            assert result.returncode == 1, f"误忽略可提交文件：{relative}"
    print("PASS: 四阶段 CLI/配置、legacy 参数传递、样本清单、目录与 Git 忽略规则；未执行真实 QC 或训练。")


if __name__ == "__main__":
    main()
