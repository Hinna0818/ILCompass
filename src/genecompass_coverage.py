"""GeneCompass 稳定基因ID映射与原始UMI覆盖；准备流程和新增研究QC共用。"""
import numpy as np

KEY_GENES = ["Gata3", "Rora", "Il7r", "Id2", "Thy1", "Il1rl1", "Il17rb", "Icos", "Klrg1", "Il5", "Il13", "Areg", "Arg1"]


def map_genes(gene_ids, tokens, median_reference):
    col = np.array([tokens.get(g, -1) for g in gene_ids], dtype=np.int32)
    median = np.array([median_reference.get(g, 1.0) for g in gene_ids], dtype=float)
    has_median = np.array([g in median_reference for g in gene_ids])
    return col, median, has_median


def summarize_coverage(a, token_col, has_median):
    umi = np.asarray(a.X.sum(axis=0)).ravel()
    in_vocab = token_col >= 0
    covered = in_vocab & has_median
    return {
        "genes": a.n_vars, "genes_in_vocab": int(in_vocab.sum()),
        "genes_with_median": int(covered.sum()),
        "umi_fraction_in_vocab": float(umi[in_vocab].sum() / umi.sum()),
        "umi_fraction_with_median": float(umi[covered].sum() / umi.sum()),
        "key_genes_not_in_vocab": sorted(set(KEY_GENES) & set(a.var.gene_symbol[~in_vocab])),
        "key_genes_without_median": sorted(set(KEY_GENES) & set(a.var.gene_symbol[in_vocab & ~has_median]))}
