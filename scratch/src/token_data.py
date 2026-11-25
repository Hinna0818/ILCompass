"""Flat, memory-mappable unmasked token storage and raw-count validation."""
from pathlib import Path
import numpy as np


def pack_sequences(ids, values, lengths):
    """Discard temporary encoder padding; preserve rank order within each cell."""
    lengths = np.asarray(lengths, dtype=np.int64)
    if ids.shape != values.shape or len(lengths) != len(ids) or (lengths <= 0).any() or (lengths > ids.shape[1]).any():
        raise ValueError("Invalid encoded shapes/lengths")
    valid = np.arange(ids.shape[1])[None, :] < lengths[:, None]
    offsets = np.r_[0, lengths.cumsum()]
    return ids[valid].astype(np.int32), values[valid].astype(np.float32), offsets


def read_shard(path):
    """Use offsets[j]:offsets[j+1] to slice one cell without padding or copying."""
    path = Path(path)
    return {key: np.load(path / (key + ".npy"), mmap_mode="r", allow_pickle=False)
            for key in ("gene_ids", "values", "offsets", "source_rows")}


def verify_cell(ids, values, original_row, token_column, medians, first_gene_id, cp10k_target):
    """Independent checks, not a second encoding algorithm; raw row is sorted CSR."""
    valid = token_column[original_row.indices] >= first_gene_id
    expected_ids = token_column[original_row.indices[valid]]
    if len(ids) != len(expected_ids) or not np.array_equal(np.sort(ids), expected_ids):
        raise ValueError("Cell lost/duplicated genes or contains padding/special IDs")
    gene_cols = np.flatnonzero(token_column >= first_gene_id)[ids - first_gene_id]
    positions = np.searchsorted(original_row.indices, gene_cols)
    raw = original_row.data[positions].astype(np.float64)
    library_size = original_row.sum(dtype=np.int64)
    # Mirror arithmetic precision to test true ties before float32 log rounding.
    ratio = raw * (cp10k_target / library_size) * (1 / medians[gene_cols])
    if (np.diff(ratio) > 0).any():
        raise ValueError("Rank not descending in normalized expression")
    tied = ratio[:-1] == ratio[1:]
    if (ids[:-1][tied] >= ids[1:][tied]).any():
        raise ValueError("Tied expression violates stable gene-ID ordering")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid positive normalized expression")
    expected_values = np.log2(1 + ratio)
    np.testing.assert_allclose(values, expected_values, rtol=3e-7, atol=1e-7)
    return float(np.max(np.abs(values.astype(np.float64) - expected_values)))
