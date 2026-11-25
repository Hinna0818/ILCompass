"""Freeze target-cell pairs and rank-matched random controls; no model inference."""
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import anndata as ad
import numpy as np
import pandas as pd
from posttrain_data import file_hash

CONFIG = ROOT / "configs/gse262028_il13_insilico_design_v1.json"


def main():
    cfg = json.loads(CONFIG.read_text())
    ref, coverage, out, report = [ROOT / cfg[k] for k in
        ["reference_directory", "coverage_directory", "output_directory", "report_directory"]]
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (out / "design.json").exists():
        raise FileExistsError("Design already frozen; use the saved manifest, do not resample controls")
    a = ad.read_h5ad(ref / "GSE126924_Lung_IL13positive_ILC2_1864.h5ad", backed="r")
    cells = pd.read_csv(coverage / "cell_token_coverage.csv.gz")
    token_data = np.load(ref / "genecompass_inputs.npz")
    ids, lengths = token_data["input_ids"], token_data["lengths"]
    if not np.array_equal(a.obs_names, cells.cell_id) or not np.array_equal(lengths, cells.input_length):
        raise ValueError("Reference and input row order differs")
    if ids.shape != (len(cells), cfg["max_length"]):
        raise ValueError("Token shape mismatch")
    token_path = ROOT / "external/GeneCompass/scdata/dict/human_mouse_tokens.pickle"
    tokens = pickle.loads(token_path.read_bytes())
    token_gene = {tokens[g]: g for g in a.var_names if g in tokens}
    symbols = a.var.gene_symbol.to_dict()
    readout = a.var_names[a.var.gene_symbol.eq(cfg["readout_symbol"])]
    if len(readout) != 1:
        raise ValueError("Ambiguous readout")
    readout_token = tokens[readout[0]]
    reader_present = (ids == readout_token).any(axis=1)
    if not np.array_equal(reader_present, cells.Il13_in_top2048):
        raise ValueError("Readout coverage differs")
    targets = pd.read_csv(coverage / "TF_primary_candidates.csv").sort_values("ensembl_gene_id").reset_index(drop=True)
    if (len(targets) != cfg["expected_targets"] or not targets.ensembl_gene_id.is_unique
        or targets.ensembl_gene_id.isin(readout).any()
        or (targets[["S17_n_guides_author50", "S25_n_guides_author50"]] < cfg["minimum_guides_each_screen"]).any().any()
        or targets.joint_Il13_token_cells.lt(cfg["minimum_joint_cells"]).any()):
        raise ValueError("Target list differs from the accepted design")
    rng = np.random.default_rng(cfg["random_seed"])
    rows = []
    for target in targets.itertuples():
        token = tokens[target.ensembl_gene_id]
        matches = ids == token
        selected = np.flatnonzero(matches.any(axis=1) & reader_present)
        if len(selected) != target.joint_Il13_token_cells:
            raise ValueError(f"Target coverage mismatch: {target.author_symbol}")
        for r in selected:
            n = int(lengths[r])
            p = int(np.flatnonzero(matches[r, :n])[0])
            window = cfg["random_rank_window"]
            choices = np.arange(max(0, p-window), min(n, p+window+1))
            choices = choices[(ids[r, choices] != token) & (ids[r, choices] != readout_token)]
            if len(choices) < cfg["random_deletions_per_target_cell"]:
                raise ValueError("Insufficient distinct rank-matched random genes")
            controls = rng.choice(choices, cfg["random_deletions_per_target_cell"], replace=False)
            row = dict(pair_id=len(rows), target_gene_id=target.ensembl_gene_id,
                       target_symbol=target.author_symbol, cell_row=int(r), cell_id=cells.cell_id.iloc[r],
                       target_token_id=token, target_rank=p+1, input_length=n,
                       Il13_rank=int(cells.Il13_rank.iloc[r]),
                       target_in_token_top_half=p+1 <= n//2,
                       target_in_full_rank_top_half=p+1 <= int(cells.full_nonzero_length.iloc[r])//2)
            for j, pos in enumerate(controls):
                g = token_gene[int(ids[r, pos])]
                row.update({f"random{j}_gene_id": g, f"random{j}_symbol": symbols[g],
                            f"random{j}_token_id": int(ids[r, pos]), f"random{j}_rank": int(pos+1)})
            rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out / "target_cell_controls.tsv.gz", sep="\t", index=False)
    targets.to_csv(out / "targets.csv", index=False)
    cells.to_csv(out / "cells.csv.gz", index=False)
    half = pairs.groupby("target_gene_id").target_in_token_top_half.sum()
    inputs = [CONFIG, token_path, ref / "GSE126924_Lung_IL13positive_ILC2_1864.h5ad",
              ref / "genecompass_inputs.npz", coverage / "TF_primary_candidates.csv",
              coverage / "cell_token_coverage.csv.gz", coverage / "summary.json"]
    summary = dict(config=cfg, reference_cells=len(a), used_cells=int(pairs.cell_row.nunique()),
        target_genes=len(targets), target_cell_pairs=len(pairs), random_interventions=len(pairs)*3,
        top_half_pairs=int(pairs.target_in_token_top_half.sum()), top_half_targets_ge100=int((half>=100).sum()),
        median_cells_per_target=float(pairs.groupby("target_gene_id").size().median()),
        planned_forward_sequences_per_model=int(pairs.cell_row.nunique()+4*len(pairs)),
        random_pool="valid baseline Top2048 genes within +/-50 ranks; exclude target and Il13; 3 distinct per pair; may include other functional genes",
        score="per-target mean of [1-cos(h_Il13_after_target,h_Il13_before)] minus mean of 3 matched random distances",
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
        manifest_sha256=file_hash(out / "target_cell_controls.tsv.gz"),
        code_sha256=file_hash(Path(__file__)))
    (out / "design.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    a.file.close()
    text = f"""# IL-13 CRISPR候选验证：固定设计 v1

用户确认：按目标匹配细胞；229个候选为主；原始GeneCompass/LoRA加匹配随机扰动对照。

- 参考池1,864个Il13阳性ILC2，按已有balanced_v1保留，不重新筛选身份或状态。
- 229个候选：两screen各≥2合格guide，且与Il13共同进入Token≥100细胞。未按screen效应或模型效果挑选。
- 实际使用{summary['used_cells']}个不同细胞、{len(pairs):,}个目标—细胞组合；每个目标使用全部合格细胞，不均衡抽样。
- 两模型同一目标使用完全相同的细胞、原始输入、随机基因与推理精度。冻结encoder、eval模式、不训练。
- 每个目标—细胞组合固定3个不同随机基因：从目标前后50名的有效输入中抽取，排除目标自身及Il13。随机种子0；两个模型共用抽样结果。随机基因可能有功能，不等同实验NT sgRNA。
- 目标/随机基因分别在原始输入表达向量中降为0，再完整重新排序并截取Top2048；不是只删除现有2048序列中的一项。Il13保持原始表达，按gene ID重新定位末层embedding，不按旧位置取向量。若某变体丢失Il13或推理失败，显式报告并对两模型/全部对照采用共同有效组合。

## 读出与验证

主读出D=1-cos(扰动后Il13末层向量,原始Il13末层向量)。每细胞额外变化=D_target-mean(D_random1..3)，再对该目标的细胞求均值。
数值大表示相对随机扰动更强的Il13表示变化，不是更接近真实KO，也不能判断促进/抑制。
使用独立计算的S17/S25筛选效应，优先评估模型分数与筛选效应绝对值的秩相关、强效目标排序富集；两个screen分开报告并检查一致性后汇总。
当前仅准备候选设计，尚未完成筛选效应/显著性重分析，不赋予hit标签。遮蔽Il13后的表达重构不是本次已确认主任务。
同一细胞会用于多个基因，细胞数不是独立动物数；每个基因汇总后作为基因排序评估单位，不把全部组合混作独立重复。
其他随机种子如需加入应另建版本；当前3个随机基因不是3个训练seed。

## 敏感性与边界

前50%仅作敏感性：有效截断后长度为主分母，共{summary['top_half_pairs']:,}组合，{summary['top_half_targets_ge100']}个目标仍有≥100细胞。完整非零长度前50%另存标记，避免混用。
Gata3因S25只剩1合格guide，不在主229清单内；若补做应独立标为探索性对照。
14个primary目标缺少proxy median，继承既有流程的median=1；逐目标见targets.csv，解释时需单列敏感性。
这是新生肺Il13阳性状态下的无监督扰动敏感性，不等同体外分化CRISPR背景或真实表达预测。

## 文件与执行范围

清单：`{out.relative_to(ROOT)}/target_cell_controls.tsv.gz`、`targets.csv`、`cells.csv.gz`、`design.json`。
脚本：`python src/downstream/geneko/prepare_il13_screen_design.py`；已有design.json时拒绝覆盖，以免重抽对照。
规划每模型约{summary['planned_forward_sequences_per_model']:,}条序列前向（基线缓存、未去重随机变体的估计），不等于已运行数量。
本次未启动模型推理。后续先做少量基因的sham、读出对齐、随机对照与重复推理检查，再执行完整任务。
"""
    (out / "DESIGN.md").write_text(text)
    (report / "report.md").write_text(text)
    (report / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ["source_sha256", "config"]}, indent=2))


if __name__ == "__main__":
    main()
