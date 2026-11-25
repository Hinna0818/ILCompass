"""Generate and fully verify All-nonzero token shards from frozen preprocessing.

From scratch/: python -u src/tokenize_corpus.py
No masking, clipping, model fitting or parameter updates.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from token_data import pack_sequences, read_shard, verify_cell

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from posttrain_data import encode, file_hash
from qc_utils import _raw_counts


def mdtable(df):
    rows = [list(df.columns), ["---"] * len(df.columns)]
    for line in df.itertuples(index=False, name=None):
        rows.append([f"{v:.6g}" if isinstance(v, float) else str(v) for v in line])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scratch/configs/tokenization_v03.json")
    cfg = json.loads(parser.parse_args().config.resolve().read_text())
    prepared, out, report_dir = [ROOT / cfg[k] for k in ("prepared", "output", "report")]
    staging = out.with_name(out.name + ".partial")
    if out.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {out} or {staging}")
    prep = json.loads((prepared / "preparation_report.json").read_text())
    frozen_files = ["config.json", "split_manifest.tsv.gz", "gene_vocab.tsv", "gene_medians_train.tsv",
                    "train_gene_detection.tsv", "functional_gene_sets.csv", "cell_input_statistics.tsv.gz"]
    for name in frozen_files:
        if file_hash(prepared / name) != prep["artifacts_sha256"][name]:
            raise ValueError(f"Frozen preprocessing changed: {name}")
    norm = json.loads((prepared / "config.json").read_text())
    if norm["cp10k_target"] != 10000:
        raise ValueError("Shared cell_norm encoder implements CP10K=10000")
    source = ROOT / norm["corpus"]
    sources = pd.read_csv(prepared / "source_manifest.tsv", sep="\t").set_index("path")
    print("Verifying frozen count corpus SHA256", flush=True)
    if file_hash(source) != sources.loc[norm["corpus"], "sha256"]:
        raise ValueError("Raw corpus changed")
    manifest = pd.read_csv(prepared / "split_manifest.tsv.gz", sep="\t", index_col="cell_id")
    vocab = pd.read_csv(prepared / "gene_vocab.tsv", sep="\t")
    gene_med = pd.read_csv(prepared / "gene_medians_train.tsv", sep="\t")
    panel = pd.read_csv(prepared / "functional_gene_sets.csv")
    cell_stats = pd.read_csv(prepared / "cell_input_statistics.tsv.gz", sep="\t", index_col="cell_id")
    if not manifest.index.equals(cell_stats.index) or not vocab.gene_id.equals(gene_med.gene_id):
        raise ValueError("Frozen row/gene alignment mismatch")
    print("Loading raw sparse X only", flush=True)
    with h5py.File(source, "r") as handle:
        counts = _raw_counts(read_elem(handle["X"]))
        index_key = handle["obs"].attrs["_index"]
        source_ids = read_elem(handle["obs"][index_key]).astype(str)
        index_key = handle["var"].attrs["_index"]
        gene_ids = pd.Index(read_elem(handle["var"][index_key]).astype(str))
    if not np.array_equal(source_ids, manifest.index) or not np.array_equal(manifest.source_row, np.arange(len(manifest))):
        raise ValueError("Manifest does not align with raw corpus")
    gene_positions = gene_ids.get_indexer(vocab.gene_id)
    if (gene_positions < 0).any() or not np.all(np.diff(gene_positions) > 0):
        raise ValueError("Vocabulary must follow stable source gene-ID order")
    first = norm["first_gene_token_id"]
    if not np.array_equal(vocab.token_id, np.arange(len(vocab)) + first):
        raise ValueError("Unexpected frozen token IDs")
    token_col = np.full(len(gene_ids), -1, dtype=np.int32)
    token_col[gene_positions] = vocab.token_id.to_numpy()
    medians = np.ones(len(gene_ids), dtype=float)
    medians[gene_positions] = gene_med.median_positive_cp10k.to_numpy()
    expected_lengths = cell_stats.nonzero_genes_in_vocab.to_numpy()
    if expected_lengths.max() > cfg["max_encoder_length"]:
        raise ValueError("Capacity exceeded: report cell and increase capacity; do not truncate")
    np.testing.assert_array_equal(np.diff(counts[:, gene_positions].indptr), expected_lengths)
    np.testing.assert_array_equal(np.asarray(counts.sum(axis=1)).ravel(), cell_stats.library_size_full_GEX)
    staging.mkdir(parents=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["value_sample_seed"])
    sample_parts, examples, shard_records, cell_records = [], [], [], []
    split_summaries, exposure_rows, grouped_rows = [], [], []
    split_detection = {}
    panel_index = np.full(len(vocab), -1, dtype=int)
    panel_tokens = vocab.set_index("gene_id").loc[panel.gene_id, "token_id"].to_numpy()
    panel_index[panel_tokens-first] = np.arange(len(panel))
    functional_ranks = np.zeros((len(manifest), len(panel)), dtype=np.int16)
    max_value_error = 0.0
    for split in norm["split_labels"]:
        rows = np.flatnonzero(manifest.split.eq(split))
        detection = np.zeros(len(vocab), dtype=np.int64)
        all_classes = vocab.exposure_class.to_numpy()
        exact = {"tokens":0,"value_sum":0.0,"value_square_sum":0.0,"min":np.inf,"max":0.0,
                 "values_gt_5":0,"values_gt_10":0,"values_gt_20":0}
        for shard_no, begin in enumerate(range(0, len(rows), cfg["cells_per_shard"])):
            source_rows = rows[begin:begin+cfg["cells_per_shard"]]
            lengths = expected_lengths[source_rows]
            offsets = np.r_[0, lengths.cumsum()].astype(np.int64)
            flat_ids = np.empty(offsets[-1], dtype=np.int32)
            flat_values = np.empty(offsets[-1], dtype=np.float32)
            for start in range(0, len(source_rows), cfg["encoding_batch_cells"]):
                stop = min(start+cfg["encoding_batch_cells"], len(source_rows))
                block = counts[source_rows[start:stop]]
                ids, vals, encoded_lengths = encode(block, token_col, medians,
                    max_len=int(lengths[start:stop].max()), encoding="cell_norm")
                np.testing.assert_array_equal(encoded_lengths, lengths[start:stop])
                packed_ids, packed_values, _ = pack_sequences(ids, vals, encoded_lengths)
                flat_ids[offsets[start]:offsets[stop]] = packed_ids
                flat_values[offsets[start]:offsets[stop]] = packed_values
            name = f"{split}/shard_{shard_no:04d}"
            folder = staging / name
            folder.mkdir(parents=True)
            for key, array in [("gene_ids",flat_ids),("values",flat_values),("offsets",offsets),
                               ("source_rows",source_rows.astype(np.int64))]:
                np.save(folder/(key+".npy"),array,allow_pickle=False)
            saved = read_shard(folder)
            np.testing.assert_array_equal(saved["source_rows"], source_rows)
            np.testing.assert_array_equal(saved["offsets"], offsets)
            if saved["gene_ids"].dtype != np.int32 or saved["values"].dtype != np.float32:
                raise ValueError("Unexpected stored dtype")
            if saved["gene_ids"].shape != saved["values"].shape or len(saved["gene_ids"]) != offsets[-1]:
                raise ValueError("Stored token lengths disagree")
            for j, source_row in enumerate(source_rows):
                a,b = int(offsets[j]),int(offsets[j+1])
                ii,vv = saved["gene_ids"][a:b],saved["values"][a:b]
                max_value_error = max(max_value_error,verify_cell(ii,vv,counts[source_row],token_col,
                    medians,first,norm["cp10k_target"]))
                keys = panel_index[ii-first]
                good = keys >= 0
                functional_ranks[source_row,keys[good]] = np.flatnonzero(good)+1
                select = rng.random(b-a) < cfg["value_sample_probability"]
                if select.any():
                    sample_parts.append((vv[select].copy(),np.full(int(select.sum()),source_row,dtype=np.int32),ii[select].copy()))
                classes = all_classes[ii-first]
                cell_records.append(dict(cell_id=manifest.index[source_row],source_row=int(source_row),
                    split=split,shard=name,shard_row=j,offset_start=a,length=b-a,
                    low_exposure_tokens=int((classes=="low_exposure").sum()),
                    unseen_positive_tokens=int((classes=="unseen_positive").sum())))
            vv = saved["values"]
            exact["tokens"] += len(vv)
            exact["value_sum"] += vv.sum(dtype=np.float64)
            exact["value_square_sum"] += np.square(vv.astype(np.float64)).sum()
            exact["min"] = min(exact["min"],float(vv.min()))
            exact["max"] = max(exact["max"],float(vv.max()))
            for threshold in (5,10,20): exact[f"values_gt_{threshold}"] += int((vv>threshold).sum())
            detection += np.bincount(saved["gene_ids"]-first,minlength=len(vocab))
            n = min(cfg["top_value_examples"],len(vv))
            top = np.argpartition(vv,-n)[-n:]
            for pos in top:
                local = int(np.searchsorted(offsets,pos,side="right")-1)
                source_row = int(source_rows[local]); gene = int(saved["gene_ids"][pos]-first)
                examples.append(dict(split=split,cell_id=manifest.index[source_row],gene_id=vocab.gene_id.iloc[gene],
                    gene_symbol=vocab.gene_symbol.iloc[gene],value=float(vv[pos]),rank=int(pos-offsets[local]+1),
                    study=manifest.study.iloc[source_row],tissue=manifest.tissue.iloc[source_row],
                    training_positive_count=int(vocab.training_positive_count.iloc[gene]),
                    median_positive_cp10k=float(gene_med.median_positive_cp10k.iloc[gene])))
            files = list(folder.glob("*.npy"))
            shard_records.append(dict(split=split,shard=name,cells=len(source_rows),tokens=len(vv),
                min_length=int(lengths.min()),max_length=int(lengths.max()),bytes=sum(p.stat().st_size for p in files),
                **{p.stem+"_sha256":file_hash(p) for p in files}))
            print(f"Verified {name}: {len(source_rows):,} cells / {len(vv):,} tokens",flush=True)
        if split=="train":
            np.testing.assert_array_equal(detection,vocab.training_positive_count)
        split_detection[split] = detection
        exact.update(split=split,cells=len(rows),mean=exact["value_sum"]/exact["tokens"])
        exact["std"] = np.sqrt(max(0,exact["value_square_sum"]/exact["tokens"]-exact["mean"]**2))
        split_summaries.append(exact)
        for cls in ("regular","low_exposure","unseen_positive"):
            selected = vocab.exposure_class.eq(cls).to_numpy()
            exposure_rows.append(dict(split=split,exposure_class=cls,genes_in_vocab=int(selected.sum()),
                genes_observed=int((detection[selected]>0).sum()),token_events=int(detection[selected].sum()),
                token_fraction=float(detection[selected].sum()/detection.sum())))
    cells = pd.DataFrame(cell_records).sort_values("source_row")
    if not np.array_equal(cells.cell_id,manifest.index) or not np.array_equal(cells.source_row,np.arange(len(manifest))):
        raise ValueError("Output cell manifest lost or duplicated source cells")
    np.testing.assert_array_equal(cells.length,expected_lengths)
    table = cells.merge(manifest[["study","tissue","condition_class"]],left_on="cell_id",right_index=True,validate="one_to_one")
    table.to_csv(staging/"cell_manifest.tsv.gz",sep="\t",index=False)
    shards = pd.DataFrame(shard_records)
    shards.to_csv(staging/"shard_manifest.tsv",sep="\t",index=False)
    value_sample = np.concatenate([s[0] for s in sample_parts])
    sample_source = np.concatenate([s[1] for s in sample_parts])
    sample_ids = np.concatenate([s[2] for s in sample_parts])
    np.savez_compressed(report_dir/"value_sample.npz",values=value_sample,source_rows=sample_source,gene_ids=sample_ids)
    summary = pd.DataFrame(split_summaries)
    summary["sampled_tokens"] = [int(manifest.split.iloc[sample_source].eq(s).sum()) for s in summary.split]
    for q,label in [(0.5,"median"),(.9,"p90"),(.99,"p99"),(.999,"p999")]:
        summary["sample_"+label] = [np.quantile(value_sample[manifest.split.iloc[sample_source].eq(s).to_numpy()],q) for s in summary.split]
    summary.to_csv(report_dir/"normalized_value_statistics.tsv",sep="\t",index=False)
    exposure = pd.DataFrame(exposure_rows)
    exposure.to_csv(report_dir/"input_exposure_classes.tsv",sep="\t",index=False)
    genotype_counts = vocab[["gene_id","gene_symbol","training_positive_count","exposure_class"]].copy()
    for split,detection in split_detection.items(): genotype_counts[f"{split}_token_events"]=detection
    genotype_counts.to_csv(report_dir/"gene_token_events.tsv",sep="\t",index=False)
    pd.DataFrame(examples).sort_values("value",ascending=False).head(cfg["top_value_examples"]).to_csv(
        report_dir/"largest_normalized_values.tsv",sep="\t",index=False)
    frows=[]
    for split in norm["split_labels"]:
        selected = manifest.split.eq(split).to_numpy()
        for tissue in ["all"]+sorted(manifest.tissue.unique()):
            ix=selected if tissue=="all" else selected & manifest.tissue.eq(tissue).to_numpy()
            rr=functional_ranks[ix]
            for j,g in enumerate(panel.itertuples()):
                r=rr[:,j];positive=r[r>0]
                frows.append(dict(split=split,tissue=tissue,gene_id=g.gene_id,gene_symbol=g.gene_symbol,
                    cells=int(ix.sum()),positive_cells=len(positive),input_cells=len(positive),truncated_cells=0,
                    detection_rate=len(positive)/ix.sum(),rank_median=np.median(positive) if len(positive) else np.nan,
                    rank_p90=np.quantile(positive,.9) if len(positive) else np.nan,
                    rank_p95=np.quantile(positive,.95) if len(positive) else np.nan))
    pd.DataFrame(frows).to_csv(report_dir/"functional_gene_input_statistics.tsv",sep="\t",index=False)
    groupcols=[["split"],["split","tissue"],["split","study"],["split","study","tissue"]]
    for cols in groupcols:
        for key,part in table.groupby(cols,observed=True):
            names=key if isinstance(key,tuple) else (key,)
            scope=dict(zip(cols,names))
            grouped_rows.append(dict(group_type="|".join(cols),**scope,cells=len(part),tokens=int(part.length.sum()),
                mean_length=part.length.mean(),median_length=part.length.median(),p99_length=part.length.quantile(.99),
                max_length=int(part.length.max()),low_exposure_tokens=int(part.low_exposure_tokens.sum()),
                unseen_positive_tokens=int(part.unseen_positive_tokens.sum()),
                cells_with_low_exposure=int(part.low_exposure_tokens.gt(0).sum()),
                cells_with_unseen_positive=int(part.unseen_positive_tokens.gt(0).sum())))
    pd.DataFrame(grouped_rows).to_csv(report_dir/"group_input_statistics.tsv",sep="\t",index=False)
    assert int(shards.tokens.sum())==int(expected_lengths.sum())==int(summary.tokens.sum())
    checks=dict(passed=True,source_and_frozen_preprocessing_hashes_match=True,all_cells_once_in_correct_split=True,
        all_stored_shards_reopened=True,all_cells_full_gene_set_verified=True,all_cell_values_checked_against_raw_counts=True,
        descending_float64_ratio_and_stable_ties_verified=True,no_padding_mask_special_tokens_stored=True,
        train_gene_event_counts_equal_frozen_detection_counts=True,no_truncation=True,no_clipping=True,
        finite_positive_values=True,checked_cells=len(cells),checked_tokens=int(shards.tokens.sum()),
        max_absolute_value_error=max_value_error)
    details=dict(dataset_version=norm["dataset_version"],config=cfg,source=norm["corpus"],
        corpus_sha256=sources.loc[norm["corpus"],"sha256"],
        frozen_preprocessing_sha256={k:prep["artifacts_sha256"][k] for k in frozen_files},
        shared_encoder_sha256=file_hash(ROOT/"src/posttrain_data.py"),
        tokenizer_sha256=file_hash(Path(__file__)),token_data_sha256=file_hash(Path(__file__).with_name("token_data.py")),
        format="npy flat arrays, int32 gene_ids, float32 values, int64 offsets/source_rows; mmap read; unmasked",
        cells=len(cells),tokens=int(shards.tokens.sum()),shards=len(shards),
        array_bytes=int(shards.bytes.sum()),capacity=cfg["max_encoder_length"],actual_max_length=int(expected_lengths.max()),
        fit="none; reused frozen TRAIN medians",masking="none; training collator is a later stage",
        quantile_scope=f"Bernoulli p={cfg['value_sample_probability']} token sample, seed={cfg['value_sample_seed']}; counts/min/max/mean/std exact",
        limitations=["Held-out input diagnostics, not model test scores","All-nonzero limited to frozen common gene vocabulary",
                     "No PAD/Mask collation, PyTorch training loader or model tested yet"])
    (staging/"config.json").write_text(json.dumps(details,indent=2)+"\n")
    (staging/"validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    staging.rename(out)
    (report_dir/"tokenization_report.json").write_text(json.dumps(details,indent=2)+"\n")
    (report_dir/"validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    report=f'''# Scratch v0.3 All-nonzero Token 分片及验收

状态：生成及全量数据验收完成。没有模型训练、Mask或表达裁剪。

## 数据与编码

输入为冻结的113,159细胞含BM语料和95/2.5/2.5细胞划分，6研究、18文库、7组织。词表为已冻结的26,725个共同测量基因；复用 `src/posttrain_data.py::encode(encoding="cell_norm")`。

`CP10K = raw_count / full_original_GEX_library_size * 10000`，按 `CP10K / frozen_Train_positive_median` 降序，存储 `log2(1+ratio)`。未重新拟合任何中位数、检测率或IDF；没有使用Val/Test拟合。

## 分片结果

{mdtable(shards.groupby('split',sort=False).agg(shards=('shard','size'),cells=('cells','sum'),tokens=('tokens','sum'),max_length=('max_length','max'),bytes=('bytes','sum')).reset_index())}

合计 **{len(cells):,}细胞 / {int(shards.tokens.sum()):,} Token / {len(shards)}分片**，数组文件共 {shards.bytes.sum()/2**30:.3f} GiB（未压缩，可mmap）。每分片至多{cfg['cells_per_shard']}个细胞，编码微批次{cfg['encoding_batch_cells']}；只在编码期间临时PAD，落盘时全部去除。
实际最大长度 **{expected_lengths.max():,}**；容量保护{cfg['max_encoder_length']}，超长报错，不以此Top-K截断。

## 全量验收

所有分片保存后重新mmap读取，对 **每个细胞** 核对原counts中的完整词表内非零基因集合；同时检查source_row/cell_id/split、无重复ID、无PAD/MASK、长度、float64 ratio降序和稳定gene-ID并列次序、每个表达值的原始counts公式。
共核对{int(shards.tokens.sum()):,}个表达值；最大绝对浮点误差 **{max_value_error:.8g}**，符合float32存储容差。Train逐基因Token次数与冻结正检测次数逐项一致，67个功能基因的所有正检测事件都保留，没有因为低曝光删除基因。

## 归一化值与尾部

{mdtable(summary[['split','tokens','min','max','mean','std','values_gt_5','values_gt_10','values_gt_20','sampled_tokens','sample_median','sample_p99','sample_p999']])}

min/max/mean/std及阈值计数为全部Token精确统计；分位数来自固定seed、独立1% Token Bernoulli抽样。不能把抽样分位数称为精确全量分位数。原始样本保存在 `value_sample.npz`，极值对应细胞、基因、rank及中位数见 `largest_normalized_values.tsv`。
没有根据极值在测试集上拟合裁剪规则；当前仅作输入诊断。若未来引入裁剪，应在Train拟合阈值并重新版本化。

## 低曝光基因输入

{mdtable(exposure)}

类别由冻结Train正检测数定义：unseen_positive=0，low_exposure=1–4，regular≥5。Val/Test中出现unseen_positive并非泄漏：词表按feature兼容空间预先定义，这些基因使用明确的中位数兜底，但没有训练阳性上下文监督。Token保留不等于表达能力可靠；不要自动门控删除。
完整逐基因计数见 `gene_token_events.tsv`，按组织/研究的输入统计见 `group_input_statistics.tsv`；功能67基因的检测和rank见 `functional_gene_input_statistics.tsv`。

## 读取格式

数据目录：`{cfg['output']}`。

```text
cell_manifest.tsv.gz  # cell_id, source_row, split, shard, shard_row, offset_start, length
shard_manifest.tsv    # 每分片数量、长度、文件hash
train/shard_0000/
  gene_ids.npy        # int32，[total_tokens]，包含有效基因ID，不含0/1
  values.npy          # float32，[total_tokens]
  offsets.npy         # int64，[cells+1]，首项0，末项total_tokens
  source_rows.npy     # int64，[cells]，映射回原始counts
validation/...
test/...
```

从scratch工作目录读取：

```python
import sys
sys.path.insert(0, "src")
from token_data import read_shard
s = read_shard("data/prepared/ilc2_scratch_v03/tokenized_shards_v1/train/shard_0000")
j = 0
a, b = int(s["offsets"][j]), int(s["offsets"][j+1])
gene_ids, values = s["gene_ids"][a:b], s["values"][a:b]
source_row = int(s["source_rows"][j])
```

基因下标由冻结词表定义，不可按symbol另排序；该Token序列为未Mask数据。数据加载后才按batch动态PAD并执行训练Mask。

## 下一步

实现训练DataLoader的长度分桶与动态PAD、15%/80:10:10 collator、固定Val Mask，然后实现E1b双重构头/Cell-macro及完整forward/backward benchmark。当前验收只覆盖数据与数值，不能称为训练循环已验收或预训练性能已提高。Test文件检查只看输入完整性，没有模型预测或Test指标选模。

复现（从scratch/，已有输出拒绝覆盖）：`/root/.conda/envs/ilc2_model/bin/python -u src/tokenize_corpus.py`。
'''
    (report_dir/"report.md").write_text(report)
    (out/"README.md").write_text(report)
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":
    main()
