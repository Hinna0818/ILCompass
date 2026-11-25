"""Freeze all-corpus cell splits and fit Train-only preprocessing statistics.

From scratch/: python src/prepare.py
Keeps the source count h5ad unchanged; manifests map to its original row order.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from posttrain_data import assign_splits, encode, file_hash
from qc_utils import _gene_ids, _raw_counts

UNKNOWN = {"", "unknown", "not_available", "nan", "None", "<NA>"}

"""
calculate statistics of train corups
"""
def train_statistics(counts, library_size, cfg):
    """Input contains TRAIN rows and vocabulary columns, with full-GEX row sums."""
    if len(counts.shape) != 2 or counts.shape[0] != len(library_size) or (library_size <= 0).any():
        raise ValueError("Invalid training counts/library sizes")
    cp = counts.astype(np.float64).multiply((cfg["cp10k_target"] / library_size)[:, None]).tocsc()
    cp.eliminate_zeros()
    detected = np.diff(cp.indptr)
    med = np.full(cp.shape[1], cfg["zero_positive_median_fallback"], dtype=float)
    mean_cp, mean_v = np.full(cp.shape[1], np.nan), np.full(cp.shape[1], np.nan)
    for j in range(cp.shape[1]):
        values = cp.data[cp.indptr[j]:cp.indptr[j+1]]
        if len(values):
            med[j] = np.median(values)
            mean_cp[j] = values.mean()
            mean_v[j] = np.log2(1 + values / med[j]).mean()
    if not np.isfinite(med).all() or (med <= 0).any():
        raise ValueError("Invalid fitted medians")
    n = counts.shape[0]
    return pd.DataFrame({
        "training_cells": n,
        "training_positive_count": detected,
        "training_detection_rate": detected/n,
        "training_umi": np.asarray(counts.sum(axis=0)).ravel(),
        "median_positive_cp10k": med,
        "median_fallback": detected == 0,
        "median_low_support": detected < cfg["low_support_positive_cells"],
        "positive_statistics_available": detected > 0,
        "exposure_class": np.where(detected == 0, "unseen_positive", np.where(
            detected < cfg["low_exposure_positive_cells"], "low_exposure", "regular")),
        "mean_positive_cp10k": mean_cp,
        "mean_positive_normalized_value": mean_v,
        "mean_all_cells_normalized_value": np.where(detected > 0, mean_v*detected/n, 0),
        "idf_weight": np.clip(np.log1p(n/np.maximum(detected, 1)), cfg["idf_min"], cfg["idf_max"]),
        "expected_masks_per_epoch_all_nonzero": cfg["mask_probability"]*detected,
        "expected_masks_example_epochs_all_nonzero": cfg["mask_probability"]*cfg["example_epochs"]*detected,
    })


def known(value):
    return pd.notna(value) and str(value) not in UNKNOWN

"""
sample condition and evidence
"""
def load_condition_samples(obs, cfg):
    """Reuse the reviewed accession evidence; preserve unreported history as unknown."""
    evidence = json.loads((ROOT / cfg["condition_evidence"]).read_text())
    by_study = {r["study"]: r for r in evidence}
    originals = {}
    for study in ("GSE117567", "GSE153748", "GSE301975", "GSE301976"):
        frame = pd.read_csv(ROOT / f"data/metadata/{study}_sample_metadata.tsv", sep="\t").fillna("not_available")
        originals[study] = frame.set_index("sample_id" if "sample_id" in frame else "sample")
    hash_samples = pd.read_csv(ROOT / "data/processed/GSE299900_qc/GSE299900_sample_metadata.tsv", sep="\t")
    rows = []
    for (study, library, tissue, hash_id), sub in obs.groupby(
            ["study", "library_id", "tissue", "audit_hash_id"], observed=True, sort=True):
        ev = by_study[study]
        gsm = str(sub["sample"].iloc[0])
        if sub["sample"].nunique() != 1:
            raise ValueError("A source library maps to multiple GSMs")
        src = originals[study].loc[gsm].to_dict() if study in originals else {}
        if study == "GSE299900":
            hs = hash_samples[(hash_samples["sample"] == gsm) & (hash_samples.hash_id == hash_id)]
            if len(hs) != 1 or int(hs.retained_cells.iloc[0]) != len(sub):
                raise ValueError("GSE299900 hash/source mapping mismatch")
            src = hs.iloc[0].to_dict()
        def field(name, *aliases):
            for key in (name, *aliases):
                if known(src.get(key)): return str(src[key])
            for key in (name, *aliases):
                if key in sub:
                    values = [str(v) for v in sub[key].unique() if known(v)]
                    if values: return "|".join(values)
            return "not_available"
        source_file = ROOT / ev["source"]
        if source_file.suffix == ".txt":
            text = source_file.read_text()
            if f"^SAMPLE = {gsm}" not in text:
                raise ValueError(f"Missing original accession evidence: {gsm}")
        condition_class = "steady_vehicle_control" if ev["classification"] == "PBS_vehicle_control" else "steady_naive"
        if condition_class == "steady_vehicle_control" and not cfg["include_PBS_vehicle_controls"]:
            raise ValueError("All-cell policy requires explicitly including vehicle controls")
        row = dict(
            condition_unit_id="|".join((study, library, tissue, hash_id)), study=study, gsm_id=gsm,
            original_sample_id=field("biological_rep", "biological_replicate_id"), library_id=library,
            tissue=tissue, hto_id=hash_id, cells=len(sub), condition_original=str(sub.condition.iloc[0]),
            condition_class=condition_class, is_steady_pretrain=True,
            condition_confidence="sample_source_supported;not_proof_of_complete_lifetime_history",
            treatment=field("condition"), vehicle="PBS" if condition_class == "steady_vehicle_control" else "not_reported",
            dose="not_available", route="not_available",
            infection_status="no_infection_reported_for_selected_control",
            pathogen="not_applicable_to_selected_control", exposure_history="not_fully_reported",
            time_after_treatment="not_available", time_after_infection="not_applicable_to_selected_control",
            reported_study_timepoint=field("study_timepoint"),
            genotype=field("genotype"), sex=field("sex"), age=field("age", "age_weeks"),
            animal_id=field("animal_id"), biological_replicate_id=field("biological_rep", "biological_replicate_id"),
            pool_size=field("pool_size", "pool_size_reported"), sort_gate=field("sort_gate"),
            demultiplex_status="author_hash_labels_preserved;not_recomputed" if known(hash_id) else "not_available",
            source_file=field("source_file", "source_h5ad", "input_file"),
            source_metadata_key=f"{gsm};hash={hash_id}" if known(hash_id) else gsm,
            provenance_evidence=ev["source"], source_url=field("source_url"),
            evidence_limitation=ev["limitation"], qc_version=cfg["qc_version"],
        )
        if study == "GSE117567":
            row["age"] = f"{field('age_range_weeks_min')}-{field('age_range_weeks_max')} weeks;study_level_only"
            row["provenance_evidence"] += ";" + ev["original_source"]
            row["infection_status"] = "steady_state_no_exogenous_stimulation_annotation"
        if study == "GSE301976":
            # Reported study time is not a per-mouse exposure history guarantee.
            row["time_after_treatment"] = field("study_timepoint")
        rows.append(row)
    samples = pd.DataFrame(rows)
    if samples.condition_unit_id.duplicated().any() or samples.cells.sum() != len(obs):
        raise ValueError("Condition units lost or duplicated cells")
    return samples, evidence


def mdtable(df):
    rows = [list(df.columns), ["---"] * len(df.columns)]
    for line in df.itertuples(index=False, name=None):
        rows.append([f"{v:.4f}" if isinstance(v, float) else str(v) for v in line])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scratch/configs/preparation_v03.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.resolve().read_text())
    source, out, report_dir = [ROOT / cfg[k] for k in ("corpus", "output", "report")]
    staging = out.with_name(out.name + ".partial")
    if out.exists() or staging.exists():
        raise FileExistsError(f"Output already exists: {out} or {staging}")
    print("Reading corpus obs, features and integer raw counts", flush=True)
    with h5py.File(source, "r") as handle:
        obs, var = read_elem(handle["obs"]), read_elem(handle["var"])
        raw = read_elem(handle["X"])
        for key in ("data", "indices", "indptr"):
            expected = getattr(raw, key)
            saved = handle[f"layers/counts/{key}"]
            if saved.shape != expected.shape:
                raise ValueError("X/counts sparse structure mismatch")
            for start in range(0, len(expected), 2_000_000):
                np.testing.assert_array_equal(expected[start:start+2_000_000], saved[start:start+2_000_000])
    counts = _raw_counts(raw)
    del raw
    if not obs.index.is_unique or obs[["study", "sample", "barcode"]].duplicated().any():
        raise ValueError("Duplicate original cells")
    if not obs.qc_pass.all() or not obs.selected_ILC2.all() or set(obs.qc_version) != {cfg["qc_version"]}:
        raise ValueError("Unexpected QC/identity input")
    ids = _gene_ids(var.index)
    if not ids.equals(var.index) or not ids.is_unique or not ids.is_monotonic_increasing:
        raise ValueError("Require stable unique sorted gene IDs")
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    if (library_size <= 0).any():
        raise ValueError("Empty cells")
    presence_cols = [f"present_{s}" for s in sorted(obs.study.unique())]
    if not set(presence_cols) <= set(var.columns):
        raise ValueError("Missing study feature presence")
    presence = var[presence_cols]
    for study, idx in obs.groupby("study", observed=True).indices.items():
        if counts[idx][:, ~var[f"present_{study}"].to_numpy()].nnz:
            raise ValueError("Counts in unmeasured features")
    common = presence.all(axis=1).to_numpy()
    obs["audit_hash_id"] = obs.hash_id.astype(str).where(obs.hash_id.map(known), "not_available")
    samples, evidence = load_condition_samples(obs, cfg)
    obs["condition_unit_id"] = ["|".join(map(str, r)) for r in obs[
        ["study", "library_id", "tissue", "audit_hash_id"]].itertuples(index=False, name=None)]
    conditions = obs[["study", "sample", "barcode", "source_cell_id", "source_h5ad", "training_tier",
                      "condition_unit_id"]].copy()
    conditions["source_row"] = np.arange(len(obs))
    conditions["condition_class"] = obs.condition_unit_id.map(samples.set_index("condition_unit_id").condition_class)
    conditions["is_steady_pretrain"] = True
    conditions["evidence_key"] = conditions.condition_unit_id
    conditions.index.name = "cell_id"
    obs["split_stratum"] = obs.condition_unit_id
    split = assign_splits(obs, cfg["split_fractions"], cfg["split_seed"], labels=cfg["split_labels"])
    manifest = obs[["study", "sample", "library_id", "tissue", "audit_hash_id", "split_stratum"]].copy()
    manifest["source_row"] = np.arange(len(obs))
    manifest["split"] = split
    manifest["condition_class"] = conditions.condition_class
    manifest.index.name = "cell_id"
    train = split.eq("train").to_numpy()
    print(f"Retained {len(obs):,} cells; split: {split.value_counts().to_dict()}", flush=True)
    print(f"Fitting nonzero CP10K medians on {train.sum():,} TRAIN cells, {common.sum():,} genes", flush=True)
    stats = train_statistics(counts[train][:, common], library_size[train], cfg)
    stats.insert(0, "gene_id", var.index[common])
    stats.insert(1, "gene_symbol", var.gene_symbol.to_numpy()[common])
    panel_raw = pd.read_csv(ROOT / cfg["functional_panel"])
    panel = panel_raw.groupby(["gene_id", "gene_symbol"], sort=False).agg({c:lambda s: "|".join(dict.fromkeys(s))
        for c in ("pathway", "role", "source")}).reset_index()
    if len(panel) != 67 or not set(panel.gene_id) <= set(stats.gene_id):
        raise ValueError("67-gene functional panel missing from common space")
    vocab = stats[["gene_id", "gene_symbol", "training_positive_count", "training_detection_rate", "exposure_class",
                   "median_fallback", "median_low_support", "idf_weight"]].copy()
    vocab.insert(2, "token_id", np.arange(len(vocab)) + cfg["first_gene_token_id"])
    for col in presence:
        vocab[col] = var.loc[common, col].to_numpy()
    vocab["gene_symbol_conflict"] = var.loc[common, "gene_symbol_conflict"].to_numpy()
    roles = panel.set_index("gene_id").role
    vocab["functional_roles"] = vocab.gene_id.map(roles).fillna("")
    vocab["is_functional67"] = vocab.gene_id.isin(panel.gene_id)
    lengths = np.diff(counts[:, common].indptr)
    retained_umi = np.asarray(counts[:, common].sum(axis=1)).ravel()
    cell_stats = manifest[["source_row", "study", "tissue", "split"]].copy()
    cell_stats["library_size_full_GEX"] = library_size
    cell_stats["nonzero_genes_in_vocab"] = lengths
    cell_stats["UMI_in_vocab"] = retained_umi
    cell_stats["UMI_fraction_in_vocab"] = retained_umi/library_size
    # Selected real rows exercise the existing production encoder without Top-K.
    col = np.full(counts.shape[1], -1, dtype=int)
    col[common] = vocab.token_id.to_numpy()
    median = np.ones(counts.shape[1]); median[common] = stats.median_positive_cp10k
    chosen = np.unique(np.r_[np.flatnonzero(train)[:2], np.flatnonzero(~train)[:2], lengths.argmax()])
    encoded, values, encoded_lengths = encode(counts[chosen], col, median, int(lengths[chosen].max()), "cell_norm")
    np.testing.assert_array_equal(encoded_lengths, lengths[chosen])
    for row, source_row in enumerate(chosen):
        n = int(encoded_lengths[row]); genes = encoded[row, :n]
        if len(np.unique(genes)) != n:
            raise ValueError("Duplicate gene tokens")
        gene_cols = np.flatnonzero(common)[genes - cfg["first_gene_token_id"]]
        original = counts[source_row]
        raw_values = original.data[np.searchsorted(original.indices, gene_cols)]
        expected = np.log2(1 + cfg["cp10k_target"]*raw_values/library_size[source_row]/median[gene_cols])
        np.testing.assert_allclose(values[row, :n], expected, rtol=1e-6)
    staging.mkdir(parents=True)
    samples.to_csv(staging / "sample_condition_manifest.tsv", sep="\t", index=False)
    conditions.to_csv(staging / "condition_manifest.tsv.gz", sep="\t")
    manifest.to_csv(staging / "split_manifest.tsv.gz", sep="\t")
    vocab.to_csv(staging / "gene_vocab.tsv", sep="\t", index=False)
    stats.to_csv(staging / "train_preprocessing_statistics.tsv", sep="\t", index=False)
    stats[["gene_id", "gene_symbol", "median_positive_cp10k", "median_fallback", "median_low_support",
           "training_positive_count"]].to_csv(staging / "gene_medians_train.tsv", sep="\t", index=False)
    stats[["gene_id", "gene_symbol", "training_positive_count", "training_detection_rate", "exposure_class",
           "idf_weight"]].to_csv(staging / "train_gene_detection.tsv", sep="\t", index=False)
    stats.merge(panel[["gene_id", "pathway", "role"]], on="gene_id").to_csv(
        staging / "functional_gene_train_statistics.tsv", sep="\t", index=False)
    panel.to_csv(staging / "functional_gene_sets.csv", index=False)
    var.to_csv(staging / "source_gene_metadata.tsv.gz", sep="\t")
    cell_stats.to_csv(staging / "cell_input_statistics.tsv.gz", sep="\t")
    (staging / "config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    (staging / "condition_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    tissue = pd.crosstab(obs.tissue, split).reindex(columns=cfg["split_labels"])
    tissue["all"] = tissue.sum(axis=1)
    distribution = tissue.copy()
    for label in cfg["split_labels"] + ["all"]:
        distribution[f"{label}_percent"] = 100*tissue[label]/tissue[label].sum()
    deviation = max(float((distribution[f"{label}_percent"]-distribution.all_percent).abs().max())
                    for label in cfg["split_labels"])
    stratum = pd.crosstab(obs.split_stratum, split).reindex(columns=cfg["split_labels"])
    target = stratum.sum(axis=1).to_numpy()[:, None]*np.asarray(cfg["split_fractions"])
    assert (np.abs(stratum.to_numpy()-target) <= 1).all()
    assert split.notna().all() and len(manifest) == len(obs)
    assert stats.training_positive_count.sum() == lengths[train].sum()
    assert int(stats.training_umi.sum()) == int(retained_umi[train].sum())
    distribution.index.name = "tissue"
    distribution.to_csv(staging / "tissue_split_counts.tsv", sep="\t")
    stratum.to_csv(staging / "stratum_split_counts.tsv", sep="\t")
    pd.crosstab(obs.study, split).reindex(columns=cfg["split_labels"]).to_csv(staging / "study_split_counts.tsv", sep="\t")
    pd.crosstab(conditions.condition_class, split).reindex(columns=cfg["split_labels"]).to_csv(staging / "condition_split_counts.tsv", sep="\t")
    length_rows = []
    for label in ["all"] + cfg["split_labels"]:
        for tissue_name in ["all"] + sorted(obs.tissue.unique()):
            select = np.ones(len(obs), bool) if label == "all" else split.eq(label).to_numpy()
            if tissue_name != "all": select &= obs.tissue.eq(tissue_name).to_numpy()
            ls = lengths[select]
            length_rows.append(dict(split=label, tissue=tissue_name, cells=len(ls),mean=ls.mean(),
                median=np.median(ls),p90=np.quantile(ls,.9),p95=np.quantile(ls,.95),p99=np.quantile(ls,.99),
                max=int(ls.max()),UMI_vocab_fraction=float(retained_umi[select].sum()/library_size[select].sum())))
    lengths_table = pd.DataFrame(length_rows)
    lengths_table.to_csv(staging / "length_statistics.tsv", sep="\t", index=False)
    source_paths = {cfg["corpus"], cfg["functional_panel"], cfg["condition_evidence"]}
    for ev in evidence:
        source_paths.add(ev["source"])
        if "original_source" in ev: source_paths.add(ev["original_source"])
    source_paths.add("data/processed/GSE299900_qc/GSE299900_sample_metadata.tsv")
    source_manifest = [dict(path=p, bytes=(ROOT/p).stat().st_size, sha256=file_hash(ROOT/p)) for p in sorted(source_paths)]
    pd.DataFrame(source_manifest).to_csv(staging / "source_manifest.tsv", sep="\t", index=False)
    output_hashes = {p.name:file_hash(p) for p in staging.iterdir() if p.is_file()}
    checks = dict(passed=True, source_counts_integer_nonnegative=True, source_X_equals_counts_layer=True,
        no_duplicate_cells=True, all_corpus_cells_retained=True, source_roles_not_modified=True,
        condition_units_account_for_all_cells=True, source_feature_absence_checked=True,
        splits_disjoint_and_exhaustive=True, per_stratum_rounding_error_at_most_one_cell=True,
        train_detection_and_UMI_totals_reconciled=True, shared_encoder_normalization_verified=True,
        median_fit_uses_train_rows_only=True, all_67_functional_genes_in_vocab=True)
    report = dict(dataset_version=cfg["dataset_version"], source=cfg["corpus"], cells=len(obs),
        genes_union=counts.shape[1],genes_common=int(common.sum()),libraries=obs.library_id.nunique(),
        condition_units=len(samples),split_strata=len(stratum),split_counts=split.value_counts().to_dict(),
        split_type="within-study/library/tissue/known-hash stratified CELL holdout; not independent study/animal",
        maximum_tissue_fraction_deviation_pp=deviation,mask_probability=cfg["mask_probability"],
        train_median_fallback_genes=int(stats.median_fallback.sum()),
        exposure_classes=stats.exposure_class.value_counts().to_dict(),
        low_support_medians=int(stats.median_low_support.sum()),
        source_manifest="source_manifest.tsv", artifacts_sha256=output_hashes,
        preparation_code_sha256=file_hash(Path(__file__)),
        shared_encode_code_sha256=file_hash(ROOT/"src/posttrain_data.py"),
        software=dict(numpy=np.__version__,pandas=pd.__version__),
        data_policy="Retain all corpus; PBS and reporter-compatible baseline included explicitly; no new identity QC",
        limits=["Unknown ages/doses/lifetime histories remain not_available/not_fully_reported",
                "BM_background_review is retained provenance, not a new mature-ILC2 validation",
                "Train-only artifacts replace old study-holdout statistics",
                "No token shards or model training in this preparation stage"])
    (staging / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    (staging / "validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    staging.rename(out)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    (report_dir / "validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    description = f'''# Scratch v0.3：条件清单、分层划分与 Train-only 预处理

状态：已完成数据准备；未导出 Token 分片，未训练模型。工作目录为 `scratch/`。

源对象：`{cfg['corpus']}`，{len(obs):,} 细胞 × {counts.shape[1]:,} 并集基因。
源 h5ad 保持原始 counts，里面原有 split/role 仍为 unassigned；**本版本正式角色以 `{cfg['output']}/split_manifest.tsv.gz` 为准**。`source_row` 为源对象零起始行号，`cell_id` 为校验主键。

## 条件与来源

全部细胞保留，覆盖6研究、{obs.library_id.nunique()}文库、{len(samples)}个样本/hash条件单位。
逐细胞 `condition_manifest.tsv.gz` 通过 condition_unit_id 连接 `sample_condition_manifest.tsv` 的处理、时间、基因型与证据。
GSE301976 PBS 明确归类 steady_vehicle_control；其他来源 steady_naive 指来源注释的基线，不是严格无一生暴露史的证明。
GSE117567 reporter 背景从原始样本表补入，不以unknown视为普通WT。
未报告的dose/route/animal_id等不猜测；未用激活状态或BM标签删除细胞。

## 固定 split

比例 Train/Validation/Test=95/2.5/2.5，seed={cfg['split_seed']}。复用 `src/posttrain_data.py::assign_splits`，按 study/library/tissue 分层，并保留GSE299900已有hash分层。各层使用确定性shuffle及最大余数取整，数量不保证等于全局比例直接四舍五入。

{mdtable(split.value_counts().reindex(cfg['split_labels']).rename_axis('split').reset_index(name='cells'))}

{mdtable(distribution.reset_index())}

相对于全体语料的组织比例，各分区最大偏差 **{deviation:.4f} 个百分点**。它是内部细胞留出，不能作为独立研究或动物泛化证据；不另将 GSE299900/GSE248385当外部Val/Test。

## 冻结词表与训练统计

真实基因词表 {common.sum():,} 个，来自六研究 feature presence 交集；PAD=0、MASK=1，gene Token从2开始，保持稳定gene_id顺序。未测量特征未作为零值拟合，67功能候选全部保留。source_gene_metadata保留并集及presence供审计。

```text
s_ig = 10000 * raw_count_ig / full_original_GEX_library_size_i
m_g  = median(s_ig > 0 over TRAIN cells only)
v_ig = log2(1 + s_ig / m_g)
```

Train零阳性基因 **{stats.median_fallback.sum():,}** 个，按配置m_g=1兜底；1–4阳性基因 **{stats.exposure_class.eq('low_exposure').sum():,}** 个。阳性<20的中位数低支持标记共 **{stats.median_low_support.sum():,}** 个，包含零阳性；均不自动删除。
IDF使用 `clip(log(1 + N_train/max(n_positive,1)), 1, 4)`，全由Train拟合。统计含正表达CP10K均值、归一化v均值、UMI和检测率，可供简单基线读取。
Mask曝光字段为全非零输入下15%目标的理论期望，每epoch及20epoch示例；不是实际训练日志。

## 新版长度分布

{mdtable(lengths_table[lengths_table.tissue.eq('all')])}

非零长度统计不依赖表达排名；rank/masking仍需后续Token化。当前最长序列为{int(lengths.max()):,}，后续可据此设置容量报错保护；本阶段没有执行Top-K截断。

## 交付与读取

- `sample_condition_manifest.tsv`、`condition_manifest.tsv.gz`：条件白名单及逐细胞来源映射。
- `split_manifest.tsv.gz`：全部细胞唯一Train/Validation/Test角色。
- `gene_vocab.tsv`：真实基因Token、presence、曝光类别、功能候选与IDF。
- `gene_medians_train.tsv`、`train_gene_detection.tsv`：冻结预处理定义。
- `train_preprocessing_statistics.tsv`：训练统计完整表；`functional_gene_train_statistics.tsv`为67候选子集。
- `cell_input_statistics.tsv.gz`：逐细胞全GEX library size及词表内非零长度。
- 组织/研究/条件/分层计数、`source_manifest.tsv`、文件hash及`validation.json`：复现与完整性记录。

验证通过：counts层一致、稳定ID唯一、源cell无重复；所有细胞分区互斥且穷尽；训练检测和UMI总量核对；共享encoder在真实细胞上核验CP10K/median归一化与无截断长度。单元测试单独记录于报告目录。

从 `scratch/` 复现（已有版本拒绝覆盖）：

```bash
/root/.conda/envs/ilc2_model/bin/python src/prepare.py
```

下一步：读取固定manifest和冻结medians，导出All-nonzero变长Token分片及固定评估Mask；再实现E1b Cell-macro模型与完整训练benchmark。
'''
    (report_dir / "report.md").write_text(description)
    (out / "README.md").write_text(description)
    print(json.dumps({k:report[k] for k in ["cells","genes_common","split_counts","train_median_fallback_genes",
        "exposure_classes","maximum_tissue_fraction_deviation_pp"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
