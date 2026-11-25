"""Train-only output panel and resumable frozen embeddings; opens no combination files."""
import gc
import json
import pickle
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from models import ROOT, FrozenILC2Encoder
from posttrain_data import encode, file_hash
from genecompass_coverage import map_genes
from qc_utils import _raw_counts


def write_json(path, value):
    partial = path.with_suffix(".partial.json")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    partial.replace(path)


def prepare(cfg, device):
    directory = ROOT / cfg["data_directory"]
    cache = ROOT / cfg["cache_directory"]
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "development_manifest.json").read_text())
    if set(manifest) != {"train", "validation"}:
        raise ValueError("Development manifest must contain train and validation only.")
    paths = {s: directory / manifest[s]["path"] for s in manifest}
    for split, path in paths.items():
        if path.resolve() != (directory / split / f"GSE136154_{split}.h5ad").resolve():
            raise ValueError("Only explicit development files may be opened.")
    prep = json.loads((ROOT / "configs/2prepare_traindata.json").read_text())
    signature = {
        "inputs": {s: file_hash(p) for s, p in paths.items()},
        "checkpoint": file_hash(ROOT / cfg["checkpoint"] / "pytorch_model.bin"),
        "tokens": file_hash(ROOT / prep["token_dictionary"]),
        "medians": file_hash(ROOT / prep["gene_median_reference"]),
        "encoding": prep["encoding"], "max_gene_tokens": prep["max_gene_tokens"],
        "n_hvg": cfg["n_hvg"], "marker_genes": cfg["marker_genes"],
        "split_version": cfg["split_version"], "precision": "float32", "pooling": "masked_gene_mean",
    }
    stamp = cache / "signature.json"
    if stamp.exists() and json.loads(stamp.read_text()) != signature:
        raise ValueError("Cache input/config changed; use a new cache directory.")
    write_json(stamp, signature)
    if (cache / "complete.json").exists():
        print("PREPARE: verified input signature; using completed cache", flush=True)
        return cache

    tokens = pickle.loads((ROOT / prep["token_dictionary"]).read_bytes())
    medians = pickle.loads((ROOT / prep["gene_median_reference"]).read_bytes())
    encoder = FrozenILC2Encoder(ROOT / cfg["checkpoint"]).to(device)
    train_ids = None
    train_genes = None
    panel_ids = None
    totals = {}
    for split, batch in [("train", "C"), ("validation", "D")]:
        print(f"PREPARE: reading {split} ({batch} batch)", flush=True)
        a = ad.read_h5ad(paths[split])
        if (not a.obs.split.eq(split).all() or not a.obs.author_batch.eq(batch).all()
                or not a.obs.condition.isin(["PBS", "IL33", "CGRP"]).all()
                or not a.obs.study.eq("GSE136154").all()
                or not a.obs.qc_version.eq("ilc2_balanced_v1").all()
                or not a.obs_names.is_unique or not a.var_names.is_unique):
            raise ValueError(f"Invalid development role/QC/identifiers in {split}")
        if json.loads(a.uns["split_provenance_json"])["split_version"] != cfg["split_version"]:
            raise ValueError("Unexpected split version.")
        if set(a.obs.condition.astype(str)) != {"PBS", "IL33", "CGRP"}:
            raise ValueError("Each development batch needs PBS and both single conditions.")
        if split == "train":
            train_ids, train_genes = set(a.obs_names), a.var_names.copy()
        elif train_ids.intersection(a.obs_names) or not train_genes.equals(a.var_names):
            raise ValueError("Overlapping cells or inconsistent measured gene space.")
        counts = _raw_counts(a.layers["counts"])
        if (counts != a.X).nnz:
            raise ValueError("Source X does not equal raw counts.")
        col, med, _ = map_genes(a.var_names, tokens, medians)
        ids, vals, lengths = encode(counts, col, med, prep["max_gene_tokens"], prep["encoding"])
        y = ad.AnnData(counts.astype(np.float32), var=a.var.copy())
        sc.pp.normalize_total(y, target_sum=1e4)
        sc.pp.log1p(y)
        if split == "train":
            sc.pp.highly_variable_genes(y, flavor="seurat", n_top_genes=cfg["n_hvg"])
            marker = y.var.gene_symbol.isin(cfg["marker_genes"])
            panel = y.var.loc[y.var.highly_variable | marker].copy()
            panel["prespecified_marker"] = marker.loc[panel.index]
            panel_ids = panel.index.tolist()
            panel.to_csv(cache / "gene_panel.tsv", sep="\t", index_label="gene_id")
            (cache / "gene_ids.txt").write_text("\n".join(panel_ids) + "\n")
            missing = sorted(set(cfg["marker_genes"]) - set(y.var.gene_symbol))
            print(f"PREPARE: TRAIN-ONLY panel G={len(panel_ids)}, missing markers={missing}", flush=True)
        np.save(cache / f"{split}_y.npy", y[:, panel_ids].X.toarray().astype(np.float32))
        a.obs[["cell_uid", "gsm", "author_batch", "condition", "split", "protocol_id"]].to_csv(
            cache / f"{split}_metadata.tsv", sep="\t", index=False)
        totals[split] = {"cells": a.n_obs, "genes": len(panel_ids), "batch": batch}
        del counts, a, y
        gc.collect()

        destination = cache / f"{split}_h.npy"
        if destination.exists():
            print(f"ENCODER: {split} already complete", flush=True)
            continue
        partial = cache / f"{split}_h.partial.npy"
        progress = cache / f"{split}_progress.json"
        start = json.loads(progress.read_text())["completed_rows"] if progress.exists() else 0
        h = np.lib.format.open_memmap(partial, mode="r+" if start else "w+",
                                     dtype=np.float32, shape=(len(ids), encoder.hidden_size))
        t0 = time.monotonic()
        for left in range(start, len(ids), cfg["encoder_batch_size"]):
            right = min(left + cfg["encoder_batch_size"], len(ids))
            length = int(lengths[left:right].max())
            tid = torch.as_tensor(ids[left:right, :length], dtype=torch.long, device=device)
            tv = torch.as_tensor(vals[left:right, :length], device=device)
            mask = torch.arange(length, device=device)[None, :] < torch.as_tensor(lengths[left:right], device=device)[:, None]
            features = encoder(tid, tv, mask).cpu().numpy()
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite frozen embedding.")
            h[left:right] = features
            if (left // cfg["encoder_batch_size"] + 1) % 50 == 0 or right == len(ids):
                h.flush()
                write_json(progress, {"completed_rows": right})
                elapsed = time.monotonic() - t0
                print(f"ENCODER {split} {right}/{len(ids)} cells; {elapsed:.1f}s since resume; "
                      f"GPU peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        h.flush()
        del h, ids, vals, lengths
        partial.replace(destination)
    write_json(cache / "complete.json", {"splits": totals, "signature": signature})
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return cache
