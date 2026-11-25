"""Read GSE202186 RNA counts for the cells annotated by the Immune Dictionary authors."""

import gzip
import io
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from threadpoolctl import threadpool_limits

from qc_utils import _raw_counts


def author_rna_columns(archive, member, channel, cell_names, n_barcodes):
    """Resolve SCP2554 cell IDs to columns in one GEO RNA library, preserving cell order."""
    with archive.extractfile(member) as packed:
        barcodes = np.frombuffer(gzip.decompress(packed.read()), dtype="S19")
    # GEO stores sorted 16-base barcodes followed by "-1\n"; SCP replaces "-1" with the channel.
    suffix = "-" + channel[2:]
    if len(barcodes) != n_barcodes or not cell_names.str.endswith(suffix).all():
        raise ValueError("Barcode format or channel mismatch: " + channel)
    needles = (cell_names.str.slice(0, -len(suffix)) + "-1\n").to_numpy(dtype="S19")
    cols = np.searchsorted(barcodes, needles)
    if (cols == len(barcodes)).any() or not np.array_equal(barcodes[cols], needles):
        raise ValueError("Unmatched RNA barcode in " + channel)
    return cols


def read_annotated_rna(tar_path, cells, libraries):
    """Build a cell-by-gene CSR AnnData with original counts, annotations and author t-SNE."""
    obs = cells.sort_values("channel", kind="stable")
    blocks = []
    feature_bytes = None
    with tarfile.open(tar_path, "r:") as archive, threadpool_limits(limits=4):
        for channel, group in obs.groupby("channel", sort=False, observed=True):
            library = libraries.loc[channel]
            prefix = f"{library.rna_gsm}_cytokine-samples{channel[2:]}-"
            with archive.extractfile(prefix + "features.tsv.gz") as packed:
                features = gzip.decompress(packed.read())
            if feature_bytes is None:
                feature_bytes = features
                var = pd.read_csv(io.BytesIO(features), sep="\t", header=None,
                                  names=["gene_id", "gene_symbol", "feature_type"], keep_default_na=False)
                var = var.set_index("gene_id", verify_integrity=True)
            elif features != feature_bytes:
                raise ValueError("RNA gene list or order differs in " + channel)

            cols = author_rna_columns(archive, prefix + "barcodes.tsv.gz", channel,
                                      group.index, library.rna_raw_barcodes)
            with archive.extractfile(prefix + "matrix.mtx.gz") as packed, gzip.GzipFile(fileobj=packed) as src:
                raw = mmread(src)
            if raw.shape != (len(var), library.rna_raw_barcodes):
                raise ValueError("RNA matrix dimensions disagree with feature/barcode lists in " + channel)
            block = _raw_counts(raw.T.tocsr()[cols])
            del raw

            umi = np.asarray(block.sum(axis=1)).ravel()
            genes_detected = np.diff(block.indptr)
            if not np.array_equal(umi, group.nCount_RNA) or not np.array_equal(genes_detected, group.nFeature_RNA):
                raise ValueError("Raw RNA counts disagree with author per-cell QC in " + channel)
            blocks.append(block)
            if len(blocks) % 5 == 0 or len(blocks) == len(libraries):
                print(f"Read {len(blocks)}/{len(libraries)} RNA libraries", flush=True)

    coords = ["tsne_1", "tsne_2"]
    adata = ad.AnnData(X=sparse.vstack(blocks, format="csr"), obs=obs.drop(columns=coords), var=var)
    adata.obsm["X_tsne"] = obs[coords].to_numpy(dtype=np.float64)
    adata.uns["geo_accession"] = "GSE202186"
    adata.uns["tsne_source"] = "SCP2554/tsne_all"
    return adata
