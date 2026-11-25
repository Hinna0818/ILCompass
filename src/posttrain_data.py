"""Posttrain 共用的确定性细胞划分与 GeneCompass 编码。"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("posttrain_train", "posttrain_val", "posttrain_test")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_splits(obs, fractions, seed, labels=SPLITS):
    """文库内随机分细胞；已知 hash 再分层，不推断未知动物或 pool。"""
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != len(labels) or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("比例数量须与分区名称一致，比例为正且和为 1")
    result = pd.Series(index=obs.index, dtype=object)
    for stratum, rows in obs.groupby("split_stratum", sort=True, observed=True).groups.items():
        rows = np.asarray(sorted(rows))
        salt = int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "little")
        rows = np.random.default_rng(np.random.SeedSequence([seed, salt])).permutation(rows)
        target = len(rows) * fractions
        sizes = np.floor(target).astype(int)
        remainder = len(rows) - sizes.sum()
        sizes[np.argsort(-(target - sizes), kind="stable")[:remainder]] += 1
        if (sizes == 0).any():
            raise ValueError(f"{stratum}: 细胞过少，无法分为{len(labels)}个非空集合")
        for split, idx in zip(labels, np.split(rows, np.cumsum(sizes)[:-1])):
            result.loc[idx] = split
    return result


def encode(counts, token_col, median, max_len=2048, encoding="median"):
    """与上一版同一编码：counts / gene median → 排序 → log2(1+x)。"""
    if encoding not in {"median", "cell_norm", "proxy"}:
        raise ValueError(f"未知 encoding: {encoding}")
    keep = np.where(token_col >= 0)[0]
    if not np.isfinite(median[keep]).all() or (median[keep] <= 0).any():
        raise ValueError("gene median 必须有限且为正")
    x = counts[:, keep].tocsr().astype(np.float64)
    if encoding == "cell_norm":
        x = x.multiply(1e4 / np.maximum(np.asarray(counts.sum(1)), 1)).tocsr()
    if encoding != "proxy":
        x = x.multiply(1 / median[keep]).tocsr()
    x.eliminate_zeros()
    tok = token_col[keep]
    ids = np.zeros((x.shape[0], max_len), dtype=np.int32)
    vals = np.zeros((x.shape[0], max_len), dtype=np.float32)
    length = np.zeros(x.shape[0], dtype=np.int16)
    for i in range(x.shape[0]):
        r = x[i]
        order = np.argsort(-r.data, kind="stable")[:max_len]
        length[i] = len(order)
        ids[i, :len(order)] = tok[r.indices[order]]
        vals[i, :len(order)] = np.log2(1 + r.data[order])
    if (length == 0).any() or not np.isfinite(vals).all():
        raise ValueError("编码产生空序列或非有限表达值")
    return ids, vals, length
