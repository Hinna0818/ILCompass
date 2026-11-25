"""Prepare GSE202186 integer mouse-by-cell-type counts and frozen PBS GeneCompass features."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gse202186 import read_annotated_rna
from genecompass_coverage import map_genes
from immune_dictionary.assemble import CELL_TO_GROUP, MIN_CELLS
from posttrain_data import encode, file_hash

SOURCE = ROOT / "immune-dict/data"
OVERVIEW = ROOT / "results/GSE202186_overview"
OUT = ROOT / "data/processed/GSE202186_perturbation_v1/rna"
CHECKPOINT = ROOT / "model/pretrained/GeneCompass_base"
GROUP = ["cyt_display", "rep", "celltype"]


def write_h5ad(adata, filename):
    with tempfile.TemporaryDirectory(dir=OUT) as directory:
        temp = Path(directory) / filename
        adata.write_h5ad(temp, compression="lzf")
        temp.replace(OUT / filename)


def prepare_counts():
    libraries = pd.read_csv(OVERVIEW / "libraries.tsv", sep="\t").set_index("channel")
    cells = pd.read_csv(SOURCE / "SCP2554_author_annotations.tsv.gz", sep="\t").set_index(
        "NAME", verify_integrity=True
    )
    slots = pd.read_csv(OVERVIEW / "hashtag_map.tsv", sep="\t", keep_default_na=False)
    slot_key = ["channel", "hashtag", "cyt_display", "rep"]
    batches = slots.loc[slots.author_cells > 0].set_index(slot_key, verify_integrity=True)
    excluded = sorted(set(cells.celltype) - CELL_TO_GROUP.keys())
    cells = cells.loc[cells.celltype.isin(CELL_TO_GROUP)].join(
        batches[["processing_batch"]], on=slot_key, validate="many_to_one"
    )
    if set(cells.celltype) != CELL_TO_GROUP.keys() or cells[GROUP + ["processing_batch"]].isna().any().any():
        raise ValueError("Cell type or experimental assignment disagrees with the study design.")
    if cells.groupby(GROUP[:2]).processing_batch.nunique().max() != 1:
        raise ValueError("A biological mouse was assigned multiple experimental batches.")

    a = read_annotated_rna(SOURCE / "GSE202186_RAW.tar", cells, libraries)
    design = a.obs
    design["pseudobulk_id"] = design[GROUP].astype(str).agg("|".join, axis=1)
    group = design.groupby("pseudobulk_id", sort=True, observed=True)
    obs = group[GROUP + ["processing_batch"]].first()
    obs["n_cells"] = group.size()
    obs["channels"] = group.channel.agg(lambda x: ",".join(sorted(set(x))))
    obs["rna_gsms"] = obs.channels.map(lambda x: ",".join(libraries.loc[x.split(","), "rna_gsm"]))
    codes = obs.index.get_indexer(design.pseudobulk_id)
    # Mouse sizes are small enough for integer32 gene sums; check the total UMI bound first.
    umi = group.nCount_RNA.sum().to_numpy()
    if umi.max() > np.iinfo(np.int32).max:
        raise OverflowError("A sample exceeds the integer32 UMI bound.")
    incidence = sparse.csr_matrix(
        (np.ones(a.n_obs, dtype=np.int32), (codes, np.arange(a.n_obs))),
        shape=(len(obs), a.n_obs),
    )
    counts = (incidence @ a.X.astype(np.int32, copy=False)).tocsr()
    counts.sort_indices()
    if not np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), umi):
        raise ValueError("Mouse UMI sums disagree with author per-cell library sizes.")
    pb = ad.AnnData(X=counts, obs=obs, var=a.var.copy())
    pb.uns.update(
        geo_accession="GSE202186",
        annotations="SCP2554_author_annotations.tsv.gz",
        raw_source="GSE202186_RAW.tar",
        batch_source="results/GSE202186_overview/hashtag_map.tsv",
        mouse_key="cyt_display + rep",
        celltype_source="SCP2554 celltype",
        expression_unit="sum of raw integer RNA UMI counts",
        excluded_author_labels=excluded,
    )
    write_h5ad(pb, "pseudobulk_counts.h5ad")
    print(f"Published {pb.n_obs} pseudobulk rows × {pb.n_vars} genes from {a.n_obs} cells.", flush=True)

    pbs = a[design.cyt_display == "PBS"].copy()
    write_h5ad(pbs, "pbs_cells_counts.h5ad")
    print(f"Published {pbs.n_obs} PBS cells × {pbs.n_vars} genes.", flush=True)


def prepare_genecompass(device, batch_size):
    import torch
    from downstream.drugpred.models import FrozenILC2Encoder

    if batch_size < 1:
        raise ValueError("The inference batch must contain at least one cell.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    pbs = ad.read_h5ad(OUT / "pbs_cells_counts.h5ad")
    group_sizes = pbs.obs.pseudobulk_id.value_counts()
    pbs = pbs[pbs.obs.pseudobulk_id.isin(group_sizes[group_sizes >= MIN_CELLS].index)]
    cfg = json.loads((ROOT / "configs/2prepare_traindata.json").read_text())
    with (ROOT / cfg["token_dictionary"]).open("rb") as handle:
        tokens = pickle.load(handle)
    with (ROOT / cfg["gene_median_reference"]).open("rb") as handle:
        medians = pickle.load(handle)
    token_col, median, has_median = map_genes(pbs.var_names, tokens, medians)
    ids, values, lengths = encode(pbs.X, token_col, median, cfg["max_gene_tokens"], cfg["encoding"])
    # Sort only to reduce padding; every row is restored to its own mouse and cell type.
    order = np.argsort(lengths, kind="stable")
    groups = pbs.obs.pseudobulk_id.astype(str)
    group_ids = np.asarray(sorted(groups.unique()), dtype=str)
    group_code = pd.Index(group_ids).get_indexer(groups)
    n_cells = np.bincount(group_code)
    # Each shard is published only upon completion. Changing weights or token inputs changes the cache.
    signature = hashlib.sha256(b"mouse-base-final-masked-gene-mean;fp32;length-sort;shard128;v1")
    checkpoint_sha = file_hash(CHECKPOINT / "pytorch_model.bin")
    signature.update(checkpoint_sha.encode())
    for arr in (ids, values, lengths):
        signature.update(arr)
    device_name = torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else str(device)
    cache = OUT / ".genecompass_cache" / device_name / signature.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)

    model = None
    sums = np.zeros((len(group_ids), 768), dtype=np.float64)
    start = time.monotonic()
    print(f"Encoding {len(ids)} PBS cells with {device_name} ({len(group_ids)} mouse/type groups).", flush=True)
    with torch.inference_mode():
        for offset in range(0, len(order), 128):
            rows = order[offset:offset + 128]
            shard_path = cache / f"{offset:06d}.npy"
            if shard_path.exists():
                vectors = np.load(shard_path, allow_pickle=False)
                if vectors.shape != (len(rows), sums.shape[1]) or not np.isfinite(vectors).all():
                    raise ValueError(f"Invalid completed GeneCompass shard at offset {offset}.")
            else:
                if model is None:
                    model = FrozenILC2Encoder(CHECKPOINT).to(device).eval()
                vectors = np.empty((len(rows), sums.shape[1]), dtype=np.float32)
                for first in range(0, len(rows), batch_size):
                    ix = rows[first:first + batch_size]
                    length = int(lengths[ix].max())
                    input_ids = torch.as_tensor(ids[ix, :length], dtype=torch.long, device=device)
                    expr = torch.as_tensor(values[ix, :length], device=device)
                    mask = torch.as_tensor(np.arange(length) < lengths[ix, None], device=device)
                    vectors[first:first + len(ix)] = model(input_ids, expr, mask).cpu().numpy()
                if not np.isfinite(vectors).all():
                    raise ValueError(f"Nonfinite GeneCompass output at offset {offset}.")
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".partial", delete=False) as temp:
                    np.save(temp, vectors)
                Path(temp.name).replace(shard_path)
            np.add.at(sums, group_code[rows], vectors)
            done = offset + len(rows)
            if offset == 0 or done % 1024 == 0 or done == len(order):
                print(f"GeneCompass {done:,}/{len(order):,} cells, {time.monotonic() - start:.1f}s", flush=True)
    embeddings = (sums / n_cells[:, None]).astype(np.float32)
    with tempfile.TemporaryDirectory(dir=OUT) as directory:
        temp = Path(directory) / "pbs_genecompass.npz"
        np.savez_compressed(
            temp, obs_names=group_ids, embeddings=embeddings, n_cells=n_cells,
            checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=checkpoint_sha,
            pooling="mean non-padding final-layer gene tokens excluding mouse CLS, then mean across cells",
            min_pbs_cells=MIN_CELLS, n_genes_in_vocab=(token_col >= 0).sum(),
            n_vocab_genes_with_median=((token_col >= 0) & has_median).sum(),
            token_dictionary=cfg["token_dictionary"], gene_median_reference=cfg["gene_median_reference"],
            gene_median_provenance="fixed human-to-mouse homolog/symbol reference; missing median uses 1",
            encoding=cfg["encoding"], max_gene_tokens=cfg["max_gene_tokens"],
            numerical_precision="float32, highest matmul precision", device=device_name,
        )
        temp.replace(OUT / temp.name)
    print(f"Published GeneCompass {embeddings.shape} for PBS.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["counts", "genecompass"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "counts":
        prepare_counts()
    else:
        prepare_genecompass(args.device, args.batch_size)


if __name__ == "__main__":
    main()
