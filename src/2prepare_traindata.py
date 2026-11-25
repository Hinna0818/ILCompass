"""合并全部已审核稳态 ILC2、文库内 90/5/5 划分、导出 GeneCompass 数据。

python src/2prepare_traindata.py [--config configs/2prepare_traindata.json]
保留全基因整数 counts；不同参考按稳定 ID 取并集，未测基因的结构性零有显式标记。
不会重新 QC、估计 median、做批次校正或训练模型。已有输出不覆盖。
"""
import argparse
import gc
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetInfo, Features, Sequence, Value
import pyarrow as pa

from genecompass_coverage import map_genes, summarize_coverage
from posttrain_data import SPLITS, assign_splits, encode, file_hash
from qc_utils import _gene_ids, _raw_counts

ROOT = Path(__file__).resolve().parents[1]


def to_dataset(ids, vals, length):
    features = Features({"input_ids": Sequence(Value("int32")), "values": Sequence(Value("float32")),
                         "length": Value("int16"), "species": Sequence(Value("int16"))})
    # 直接从连续 numpy buffer 构造 Arrow，避免把数亿 token 转成 Python 标量。
    offsets = pa.array(np.arange(len(ids) + 1, dtype=np.int32) * ids.shape[1])
    table = pa.table({"input_ids": pa.ListArray.from_arrays(offsets, pa.array(ids.ravel())),
                      "values": pa.ListArray.from_arrays(offsets, pa.array(vals.ravel())),
                      "length": pa.array(length), "species": pa.array([[1]] * len(ids), type=pa.list_(pa.int16()))})
    return Dataset(table, info=DatasetInfo(features=features))


def merge_sources(cfg):
    parts, sources, symbols = [], {}, []
    for study, path in cfg["inputs"].items():
        a = ad.read_h5ad(ROOT / path)
        version = a.uns.get("metadata", {}).get("version")
        if version != cfg["qc_version"] or not a.obs_names.is_unique:
            raise ValueError(f"{study}: QC 版本不符或细胞 ID 重复")
        if not a.obs.selected_ILC2.all() or not a.obs.qc_pass.all():
            raise ValueError(f"{study}: 输入包含未通过 QC/身份筛选的细胞")
        genes = _gene_ids(a.var_names)
        if not genes.is_unique or not genes.str.startswith("ENSMUSG").all():
            raise ValueError(f"{study}: 需要唯一的小鼠稳定 Ensembl ID")
        counts = _raw_counts(a.X)
        if "counts" in a.layers and (counts != a.layers["counts"]).nnz:
            raise ValueError(f"{study}: X 与原始 counts 层不同")
        obs = a.obs.copy()
        for col in obs.select_dtypes("category").columns:
            obs[col] = obs[col].astype(object)
        obs["source_cell_id"] = a.obs_names.astype(str)
        for col in ("split", "role", "data_role"):
            if col in obs:
                obs[f"source_{col}"] = obs[col].astype(str)
        obs.index = [s if s.startswith(study + ":") else study + ":" + s for s in obs.source_cell_id]
        obs.index.name = "cell_id"
        obs["cell_uid"] = obs.index
        obs["study"] = study
        obs["source_h5ad"] = path
        obs["tissue_original"] = obs.tissue.astype(str)
        obs["tissue"] = obs.tissue_original.replace({"Gut": "SI"})
        if not set(obs.tissue) <= {"Lung", "SI", "LI", "Pancreas", "Skin", "Fat"}:
            raise ValueError(f"{study}: 未审核的组织")
        obs["library_id"] = study + ":" + obs["sample"].astype(str)
        obs["split_stratum"] = obs.library_id
        if "hash_id" in obs:
            valid_hash = obs.hash_id.notna() & ~obs.hash_id.astype(str).isin(["", "unknown", "unassigned"])
            obs.loc[valid_hash, "split_stratum"] += ":" + obs.loc[valid_hash, "hash_id"].astype(str)
        obs["qc_version"] = version
        symbols.append(pd.Series(a.var.gene_symbol.astype(str).to_numpy(), index=genes, name=study))
        parts.append(ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes)))
        sources[study] = {"path": path, "sha256": file_hash(ROOT / path), "cells": a.n_obs,
                          "genes": a.n_vars, "raw_umi": int(counts.sum(dtype=np.int64)), "qc_version": version}
        print(f"Loaded {study}: {a.shape}", flush=True)
        del a
        gc.collect()
    print("Merging gene union and source metadata", flush=True)
    merged = ad.concat(parts, join="outer", fill_value=0, merge=None)
    if not merged.obs_names.is_unique or merged.n_obs != sum(s["cells"] for s in sources.values()):
        raise ValueError("合并发生细胞丢失或 ID 重复")
    annotations = pd.concat(symbols, axis=1).reindex(merged.var_names)
    merged.var["gene_symbol"] = annotations.bfill(axis=1).iloc[:, 0]
    merged.var["gene_symbol_conflict"] = annotations.nunique(axis=1) > 1
    for study in cfg["inputs"]:
        merged.var[f"present_{study}"] = annotations[study].notna()
        merged.var[f"symbol_{study}"] = annotations[study].fillna("")
    for col in merged.obs.select_dtypes("object").columns:
        merged.obs[col] = merged.obs[col].fillna("unknown").astype(str)
    print("Assigning fixed within-library splits", flush=True)
    merged.obs["split"] = assign_splits(merged.obs, cfg["split_fractions"], cfg["random_seed"])
    merged.obs["role"] = merged.obs["split"]
    merged.obs["data_role"] = merged.obs["split"]
    merged.layers["counts"] = merged.X.copy()
    if int(merged.X.sum(dtype=np.int64)) != sum(s["raw_umi"] for s in sources.values()):
        raise ValueError("合并改变了 UMI 总数")
    return merged, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/2prepare_traindata.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    out, rep, combined = (ROOT / cfg[k] for k in ("prepared_dataset_path", "report_path", "combined_h5ad"))
    if out.exists() or combined.exists():
        raise FileExistsError("已有输出不覆盖；请检查后更改版本路径或明确删除待替代版本")
    rep.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    combined.parent.mkdir(parents=True, exist_ok=True)
    a, sources = merge_sources(cfg)
    tokens = pickle.load(open(ROOT / cfg["token_dictionary"], "rb"))
    medians = pickle.load(open(ROOT / cfg["gene_median_reference"], "rb"))
    col, med, has_med = map_genes(a.var_names, tokens, medians)
    a.var["genecompass_token"] = col
    a.var["has_median_reference"] = has_med
    a.var["genecompass_median"] = med
    coverage = summarize_coverage(a, col, has_med)
    report = {"config": cfg, "sources": sources, "cells": a.n_obs, "genes_union": a.n_vars,
              "genes_common": int(a.var[[f"present_{s}" for s in sources]].all(axis=1).sum()),
              "gene_symbol_conflicts": int(a.var.gene_symbol_conflict.sum()), "coverage": coverage,
              "token_dictionary_sha256": file_hash(ROOT / cfg["token_dictionary"]),
              "median_reference_sha256": file_hash(ROOT / cfg["gene_median_reference"]),
              "split_cells": a.obs.split.value_counts().to_dict(),
              "limitations": ["同文库细胞随机留出为内部评价，不代表新动物/新研究泛化",
                              "并集中的 absent genes 为结构性零，见 var.present_<study>；参考版本差异仍存在",
                              "身份为计算支持；小鼠 median 是既有同源映射代理，缺失参考使用 1"]}
    a.uns["metadata"] = {"version": cfg["qc_version"], "dataset_version": cfg["dataset_version"],
                          "counts": "X and layers/counts are unnormalized nonnegative integer UMI"}
    a.uns["posttrain_provenance_json"] = json.dumps(report, ensure_ascii=False)
    for split in SPLITS:
        idx = np.flatnonzero(a.obs.split.eq(split))
        print(f"Encoding {split}: {len(idx):,} cells", flush=True)
        ids, vals, lengths = encode(a.X[idx], col, med, cfg["max_gene_tokens"], cfg["encoding"])
        a.obs.loc[a.obs.index[idx], "genecompass_length"] = lengths
        ds = to_dataset(ids, vals, lengths)
        ds.save_to_disk(str(out / split))
        cellmap = a.obs.iloc[idx][["cell_uid", "study", "sample", "library_id", "split_stratum", "tissue", "split"]].copy()
        cellmap["length"] = lengths
        cellmap.to_csv(out / f"{split}_cell_map.tsv.gz", sep="\t", index=False)
        del ds, ids, vals
        gc.collect()
    a.obs["genecompass_length"] = a.obs.genecompass_length.astype("int16")
    a.obs.to_csv(combined.with_name(combined.stem + "_metadata.tsv.gz"), sep="\t")
    a.obs.groupby(["study", "library_id", "tissue", "split"], observed=True).size().unstack(fill_value=0).to_csv(rep / "library_split_counts.tsv", sep="\t")
    pd.crosstab(a.obs.tissue, a.obs.split).to_csv(rep / "tissue_split_counts.tsv", sep="\t")
    a.var.to_csv(rep / "gene_union_metadata.tsv.gz", sep="\t")
    partial = combined.with_suffix(".partial.h5ad")
    print(f"Saving full-count h5ad: {a.shape}", flush=True)
    a.write_h5ad(partial, compression="gzip")
    partial.rename(combined)
    report["combined_h5ad_sha256"] = file_hash(combined)
    report["cell_map_sha256"] = {s: file_hash(out / f"{s}_cell_map.tsv.gz") for s in SPLITS}
    (rep / "prepare_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("cells", "genes_union", "genes_common", "split_cells", "coverage")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
