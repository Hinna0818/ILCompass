"""Save Il13+ reference cells and audit screen TF targets using the existing GC KO tokenizer."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from qc_utils import _raw_counts
from posttrain_data import file_hash
from nmur1_genecompass_original import official_tokenizer

CONFIG = ROOT / "configs/gse126924_il13_tf_coverage_v1.json"


def main():
    cfg = json.loads(CONFIG.read_text())
    lora = json.loads((ROOT / cfg["lora_config"]).read_text())
    out, dest, screen = [ROOT / cfg[k] for k in ["output", "reference_directory", "screen_directory"]]
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(ROOT / cfg["source"])
    x = _raw_counts(a.layers["counts"])
    if (x != a.X).nnz or not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError("Counts or IDs invalid")
    if not a.obs.selected_ILC2.all() or not a.obs.qc_version.eq("ilc2_balanced_v1").all():
        raise ValueError("Expected balanced_v1 ILC2")
    readout_cols = np.flatnonzero(a.var.gene_symbol.eq(cfg["readout_symbol"]))
    if len(readout_cols) != 1:
        raise ValueError("Readout gene mapping is not unique")
    readout_col = int(readout_cols[0])
    readout_gene = a.var_names[readout_col]
    positive = x[:, readout_col].toarray().ravel() > 0
    a = a[positive].copy()
    if len(a) != cfg["expected_cells"]:
        raise ValueError("Reference selection differs from the audited 1864 cells")
    a.obs["reference_selection"] = "balanced_v1_ILC2_and_Il13_raw_counts_gt0"
    a.obs["role"] = "crispr_IL13_reference"
    a.obs["split"] = "unassigned"
    a.uns["reference_selection"] = dict(source=cfg["source"], readout_gene=readout_gene,
        definition="raw counts > 0; computational ILC2 identity, not experimentally confirmed singlets",
        source_sha256=file_hash(ROOT / cfg["source"]))
    reference_file = dest / "GSE126924_Lung_IL13positive_ILC2_1864.h5ad"
    a.write_h5ad(reference_file, compression="gzip")
    a.obs.to_csv(dest / "reference_cells.tsv.gz", sep="\t", index_label="cell_id")

    library = pd.read_csv(screen / "gene_screen_coverage.tsv", sep="\t", dtype={"author_gene_id": str})
    library = library[library.endpoint.eq("IL13")]
    genes = pd.read_csv(screen / "gene_map.tsv", sep="\t", dtype={"author_gene_id": str})
    for name in ["n_guides_original", "n_guides_main", "n_guides_author50"]:
        pivot = library.pivot(index="author_gene_id", columns="screen", values=name)
        pivot.columns = [str(c) + "_" + name for c in pivot.columns]
        genes = genes.merge(pivot, left_on="author_gene_id", right_index=True, validate="one_to_one")
    tokens = pickle.loads((ROOT / lora["token_dictionary"]).read_bytes())
    median = pickle.loads((ROOT / lora["median_reference"]).read_bytes())
    keep = a.var.is_core.to_numpy(bool) & a.var_names.isin(tokens)
    input_genes = a.var_names[keep].to_numpy()
    scale = np.array([median.get(g, 1.) for g in input_genes])
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid fixed median reference")
    input_token_ids = np.array([tokens[g] for g in input_genes])
    if len(set(input_token_ids)) != len(input_genes):
        raise ValueError("Nonunique gene-token mapping")
    full = a.layers["counts"].tocsr()
    model_x = full[:, keep].tocsr()
    n, m = len(a), len(genes)
    ranks = np.zeros((n, m), dtype=np.int32)
    readout_rank = np.zeros(n, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    max_length = int(lora["max_length"])
    target_index = {}
    for j, row in genes.iterrows():
        if row.mapping_status == "unique_exact_symbol" and row.ensembl_gene_id in input_genes:
            target_index.setdefault(tokens[row.ensembl_gene_id], []).append(j)
    tokenizer = official_tokenizer()
    readout_token = tokens[readout_gene]
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    input_values = np.zeros((n, max_length), dtype=np.float32)
    for r in range(n):
        ii, vv = tokenizer(np.log2(1 + model_x[r].toarray().ravel() / scale), input_genes, tokens)
        lengths[r] = len(ii)
        cut = min(len(ii), max_length)
        input_ids[r, :cut], input_values[r, :cut] = ii[:cut], vv[:cut]
        for rank, token in enumerate(ii, 1):
            if token == readout_token:
                readout_rank[r] = rank
            for j in target_index.get(token, []):
                ranks[r, j] = rank
    if not (readout_rank > 0).all():
        raise ValueError("Positive readout is absent from the full input ranking")
    np.savez_compressed(dest / "genecompass_inputs.npz", input_ids=input_ids, values=input_values,
                       lengths=np.minimum(lengths, max_length))
    sparse.save_npz(out / "target_ranks.npz", sparse.csr_matrix(ranks))
    cells = pd.DataFrame(dict(cell_id=a.obs_names, full_nonzero_length=lengths,
        input_length=np.minimum(lengths, max_length), Il13_rank=readout_rank,
        Il13_in_top2048=readout_rank <= max_length))
    cells.to_csv(out / "cell_token_coverage.csv.gz", index=False)
    reader_in = readout_rank <= max_length
    rows = []
    for j, row in genes.iterrows():
        gene = row.ensembl_gene_id
        measured = row.mapping_status == "unique_exact_symbol" and gene in a.var_names
        c = full[:, a.var_names.get_loc(gene)].toarray().ravel() if measured else None
        rr = ranks[:, j]
        detected = rr > 0
        included = detected & (rr <= max_length)
        joint = included & reader_in
        valid_half = detected & (rr <= np.minimum(lengths, max_length) // 2)
        full_half = detected & (rr <= lengths // 2)
        stats = dict(target_column=j, measured=measured,
            is_core=bool(a.var.loc[gene, "is_core"]) if measured else False,
            in_vocabulary=gene in tokens, in_model_gene_space=gene in input_genes,
            median_available=gene in median, cells=n,
            detected_cells=int((c > 0).sum()) if measured else None,
            detection_rate=float((c > 0).mean()) if measured else None,
            zero_cells=int((c == 0).sum()) if measured else None,
            mean_raw_counts=float(c.mean()) if measured else None,
            ranked_cells=int(detected.sum()), token_cells=int(included.sum()), token_rate=float(included.mean()),
            truncated_cells=int((detected & ~included).sum()),
            inclusion_given_detection=float(included.sum() / (c > 0).sum()) if measured and (c > 0).any() else None,
            joint_Il13_token_cells=int(joint.sum()), joint_Il13_token_rate=float(joint.mean()),
            token_top_half_cells=int(valid_half.sum()), full_rank_top_half_cells=int(full_half.sum()),
            joint_Il13_token_top_half_cells=int((valid_half & reader_in).sum()),
            joint_Il13_full_rank_top_half_cells=int((full_half & reader_in).sum()))
        for label, q in [("median", .5), ("p90", .9), ("p95", .95)]:
            stats["rank_" + label] = float(np.quantile(rr[detected], q)) if detected.any() else None
        stats["relative_rank_median"] = float(np.median(rr[detected] / lengths[detected])) if detected.any() else None
        rows.append(stats)
    result = pd.concat([genes.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    t = cfg["candidate_thresholds"]
    result["guide_supported_both_screens"] = (
        result.S17_n_guides_author50.ge(t["minimum_guides_author50_each_screen"])
        & result.S25_n_guides_author50.ge(t["minimum_guides_author50_each_screen"]))
    result["coverage_tier"] = np.select([
        ~result.guide_supported_both_screens,
        ~result.in_model_gene_space,
        result.joint_Il13_token_cells.ge(t["primary_joint_cells"]),
        result.joint_Il13_token_cells.ge(t["exploratory_joint_cells"])],
        ["guide_review", "mapping_or_input_unavailable", "primary_coverage", "exploratory_coverage"],
        default="low_joint_coverage")
    result.to_csv(out / "TF_coverage_all.csv", index=False)
    eligible = result[result.coverage_tier.eq("primary_coverage")].sort_values(
        ["joint_Il13_token_cells", "rank_median"], ascending=[False, True])
    eligible.to_csv(out / "TF_primary_candidates.csv", index=False)
    result[result.coverage_tier.eq("exploratory_coverage")].to_csv(out / "TF_exploratory_candidates.csv", index=False)
    top_half = result[result.guide_supported_both_screens
                      & result.joint_Il13_token_top_half_cells.ge(t["primary_joint_cells"])]
    top_half.to_csv(out / "TF_top_half_candidates.csv", index=False)
    main_targets = result[result.S17_n_guides_main.gt(0)]
    il13 = dict(cells=n, detected=n, input_cells=int(reader_in.sum()), truncated=int((~reader_in).sum()),
                rank_median=float(np.median(readout_rank)), rank_p90=float(np.quantile(readout_rank, .9)))
    summary = dict(config=cfg, readout=il13, tf_library_targets=len(result),
        targets_with_main_guides=int(result.S17_n_guides_main.gt(0).sum()),
        targets_measured=int(result.measured.sum()), targets_in_model_gene_space=int(result.in_model_gene_space.sum()),
        targets_detected=int(result.detected_cells.gt(0).sum()), targets_tokenized=int(result.token_cells.gt(0).sum()),
        targets_joint_with_Il13=int(result.joint_Il13_token_cells.gt(0).sum()),
        main_library=dict(targets=len(main_targets), measured=int(main_targets.measured.sum()),
            input_space=int(main_targets.in_model_gene_space.sum()),
            zero_in_all_cells=int((main_targets.measured & main_targets.detected_cells.eq(0)).sum())),
        top_half_primary_targets=len(top_half),
        tiers=result.coverage_tier.value_counts().to_dict(),
        selected_doublet_review=int(a.obs.doublet_review.sum()),
        input_genes=len(input_genes), input_genes_missing_median=int(sum(g not in median for g in input_genes)),
        tokenizer="Existing GC KO flow: core+vocab; log2(1+raw_counts/frozen_mouse_proxy_median); official argsort; Top2048; no CP10K",
        sha256={str(p.relative_to(ROOT)): file_hash(p) for p in
            [CONFIG, ROOT / cfg["source"], ROOT / lora["token_dictionary"], ROOT / lora["median_reference"],
             screen / "gene_map.tsv", screen / "gene_screen_coverage.tsv", Path(__file__)]})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    text = f"""# Il13阳性肺ILC2与CRISPR TF库覆盖度

固定参考：{n}个GSE126924 Lung balanced_v1计算身份支持、Il13原始counts>0细胞。
对象保存于`{reference_file.relative_to(ROOT)}`；全部原始基因与counts保留，未按TF覆盖或模型预测再次删细胞。
其中{summary['selected_doublet_review']}个带Scrublet复核标记；本版标记不独立删除，不称全部实验验证singlets。

## 排序与分母

复用既有GeneCompass/LoRA KO tokenizer，core基因∩词表，共{len(input_genes)}个输入候选。
raw counts除以固定mouse proxy median，再log2(1+x)，官方默认argsort并列规则；取Top-{max_length}；没有CP10K。
原始作者小鼠median不可得，当前proxy与既有任务一致，不冒充作者原始参考。
{summary['input_genes_missing_median']}个输入候选缺少median，按既有代码使用1，逐靶基因列出median_available。
两模型输入相同，故覆盖度相同；没有提取模型embedding或运行KO。

Il13：原始阳性{n}，实际输入{il13['input_cells']}（{il13['input_cells']/n:.2%}），截断{il13['truncated']}；非零输入全集中的rank中位数{il13['rank_median']:.1f}，P90={il13['rank_p90']:.1f}。
rank为1起始的基因位置，不含CLS。TF rank分布只计算有有效输入排序的阳性细胞；未测量/映射不明不计为零表达。
主要token率与joint率分母均为全部{n}个固定参考细胞，另给阳性条件下覆盖率。
top-half同时报告截断后有效长度与完整非零长度两种分母，不能互换。非零rank矩阵的行对应cell_token_coverage.csv.gz，列对应TF_coverage_all.csv的target_column，0表示无有效rank，原因需查看基因级映射/检出字段。

## sgRNA支持与候选

作者TF库共{len(result)}个符号（包括部分广义转录调控相关目标，未另认证每个都是序列特异性TF）。
主输入有guide的目标{summary['targets_with_main_guides']}；所有原始目标仍在结果表中，跨基因重复序列已由上游排除。
实测基因空间可唯一映射{summary['targets_measured']}；模型输入空间{summary['targets_in_model_gene_space']}；至少1细胞表达{summary['targets_detected']}；至少1细胞进入Token {summary['targets_tokenized']}。
仅计有主分析guide的1131个目标：{summary['main_library']['measured']}个在表达矩阵中，{summary['main_library']['input_space']}个可进入模型输入空间，{summary['main_library']['zero_in_all_cells']}个虽有测量但在本参考中全零。完整1133行表保留两个失去全部guide的目标以便追踪。
使用已经整理的S17/S25 author50资格，未重新筛guide、未计算screen效应或命中标签。
本次按覆盖度预设：两个screen各≥{t['minimum_guides_author50_each_screen']}合格guide且与Il13共同输入≥{t['primary_joint_cells']}细胞，列为primary；共同输入{t['exploratory_joint_cells']}–{t['primary_joint_cells']-1}列为exploratory。
这是工程可评估性分层，不是显著性或生物功能判定。若后续坚持原KO的top-half条件，应再使用对应joint_top_half列筛选，不能把当前primary数量视为最终top-half合格数。
截断后有效Token长度前50%且与Il13共同输入至少100细胞、两screen各≥2合格guide的目标共{len(top_half)}个，另存TF_top_half_candidates.csv。

分层计数：{json.dumps(summary['tiers'], ensure_ascii=False)}。
结果：TF_coverage_all.csv、TF_primary_candidates.csv、TF_exploratory_candidates.csv；没有按预期模型效果挑基因。
覆盖不足可能来自稳态/年龄/培养条件差异、零检测或排序截断，不能认定该基因不调控IL-13。
本参考为新生期肺ILC2，不等同筛选中的体外分化背景；结论限于该参考状态。

复现：`python src/downstream/geneko/audit_il13_screen_tf_coverage.py`。
"""
    (out / "report.md").write_text(text)
    (dest / "README.md").write_text(f"# IL-13 CRISPR参考细胞\n\n{text}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
