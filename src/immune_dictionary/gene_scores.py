"""Gene-level scoring on measured changes, edgeR calls and Cui et al. Supplementary Table 3."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


DE_METRICS = ["DEG_overlap", "DEG_overlap_direction", "MAE_DEG", "Pearson_DEG",
              "DEG_precision", "DEG_F1", "DEG_Jaccard", "DEG_sign_accuracy", "DEG_AP", "DEG_prevalence"]
PANEL_METRICS = ["PCC_delta", "MAE", "MSE", "MSE_over_zero", "Delta_R2", "direction_accuracy", "top_overlap",
                 "DEG_precision_at_n", "DEG_recall_at_n", "DEG_F1_at_n"]
KEY = ["celltype", "cyt_display"]


def pcc_delta(pred, obs):
    if len(pred) < 3 or np.ptp(pred) == 0 or np.ptp(obs) == 0:
        return np.nan
    x = np.asarray(pred, dtype=np.float64) - np.mean(pred, dtype=np.float64)
    y = np.asarray(obs, dtype=np.float64) - np.mean(obs, dtype=np.float64)
    return np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))


def ranked_credit(magnitude, k):
    """Up to k nonzero predictions; genes tied at the cutoff share the last places."""
    positive = magnitude > 0
    if k <= 0:
        return np.zeros_like(magnitude, dtype=np.float64)
    if positive.sum() <= k:
        return positive.astype(np.float64)
    cutoff = np.partition(magnitude[positive], -k)[-k]
    above = magnitude > cutoff
    tied = magnitude == cutoff
    result = above.astype(np.float64)
    result[tied] = (k - above.sum()) / tied.sum()
    return result


def deg_metrics(pred, observed, codes):
    result = dict.fromkeys(["n_tested", "n_deg", "n_predicted", "DEG_hits", "direction_hits", *DE_METRICS], np.nan)
    if codes is None:
        return result
    tested = codes > 0
    code, p, y = codes[tested], pred[tested], observed[tested]
    deg = code >= 2
    k = int(deg.sum())
    result.update(n_tested=len(code), n_deg=k)
    if k == 0:
        return result
    score = np.abs(p)
    credit = ranked_credit(score, k)
    correct = ((code == 2) & (p < 0)) | ((code == 3) & (p > 0))
    hits, signed = credit[deg].sum(), credit[correct].sum()
    n_pred = int(min(k, np.count_nonzero(score)))
    result.update(n_predicted=n_pred, DEG_hits=hits, direction_hits=signed,
                  DEG_overlap=hits / k, DEG_overlap_direction=signed / k,
                  DEG_precision=hits / n_pred if n_pred else np.nan, DEG_F1=2 * hits / (k + n_pred),
                  DEG_Jaccard=hits / (k + n_pred - hits), DEG_sign_accuracy=correct.sum() / k,
                  DEG_AP=average_precision_score(deg, score), DEG_prevalence=k / len(code),
                  MAE_DEG=np.abs(p[deg] - y[deg]).mean(), Pearson_DEG=pcc_delta(p[deg], y[deg]))
    return result


def top_effect_panels(pred, observed, codes):
    """Select the experimental panel once per condition from |observed delta| on edgeR-testable genes."""
    idx = np.flatnonzero(codes > 0)
    y, p = observed[idx], pred[idx]
    # Stable gene-axis order breaks exact observed ties; predicted ties receive fractional credit.
    magnitude = np.abs(y)
    order = np.argsort(-magnitude, kind="stable")
    significant = codes[idx] >= 2
    k = int(significant.sum())
    for n in (20, 50, 100):
        if len(order) < n or magnitude[order[n - 1]] == 0:
            continue
        panel = order[:n]
        true, model = y[panel], p[panel]
        mse = np.mean((model - true) ** 2)
        var = np.mean((true - true.mean()) ** 2)
        credit = ranked_credit(np.abs(p), n)
        hits = credit[significant].sum()
        yield dict(top_n=n, n_tested=len(idx), n_significant=int(significant[panel].sum()), n_deg=k,
                   observed_cutoff_tie=bool(len(order) > n and magnitude[order[n]] == magnitude[order[n - 1]]),
                   PCC_delta=pcc_delta(model, true), MAE=np.abs(model - true).mean(), MSE=mse,
                   MSE_over_zero=mse / np.mean(true ** 2), Delta_R2=1 - mse / var if np.ptp(true) else np.nan,
                   direction_accuracy=np.mean(np.sign(model) == np.sign(true)),
                   top_overlap=credit[panel].sum() / n, DEG_precision_at_n=hits / n if k else np.nan,
                   DEG_recall_at_n=hits / k if k else np.nan, DEG_F1_at_n=2 * hits / (n + k) if k else np.nan)


def alternative_references(dataset, base, cui_path, effect_path):
    """Build secondary DEG standards after freezing predictions; record lost, ambiguous and filtered genes."""
    refs, audit = {}, []
    effects = pd.read_csv(effect_path, sep="\t")
    axis = pd.read_csv(dataset.folder / "genes.tsv", sep="\t", keep_default_na=False)
    if not np.array_equal(axis.gene_id.to_numpy()[effects.gene_index.to_numpy(dtype=int)], effects.gene_id.to_numpy()):
        raise ValueError("edgeR effect sizes and model gene IDs disagree.")
    effects_by_key = {k: rows for k, rows in effects.groupby(KEY, sort=False)}
    paper = pd.read_csv(cui_path, sep="\t", keep_default_na=False)
    if paper.duplicated(KEY + ["gene_symbol"]).any() or not (paper.published_log2FC.abs() > 0).all():
        raise ValueError("Duplicate published signatures or undefined effect sign.")
    if set(k[0] for k in base) - set(paper.celltype) or set(k[1] for k in base) - set(paper.cyt_display):
        raise ValueError("A scored author type or cytokine does not match Supplementary Table 3.")
    unique = axis.gene_symbol.ne("") & ~axis.gene_symbol.duplicated(keep=False)
    symbol_idx = pd.Series(axis.index[unique], index=axis.loc[unique, "gene_symbol"])
    paper["index"] = paper.gene_symbol.map(symbol_idx)
    paper["ambiguous"] = paper.gene_symbol.isin(axis.loc[axis.gene_symbol.duplicated(False), "gene_symbol"])
    # Table 3 only publishes positives. This explicitly defined candidate panel excludes the named gene families.
    comparable = (unique & ~axis.gene_symbol.str.match(r"^(Rps|Rpl|mt-)", case=False)).to_numpy()
    published = {k: rows for k, rows in paper.groupby(KEY, sort=False)}
    for key, codes in base.items():
        genes = published.get(key, paper.iloc[:0])
        mapped = genes.dropna(subset=["index"])
        gi = mapped["index"].to_numpy(dtype=int)
        full = comparable.astype(np.uint8)
        if not comparable[gi].all():
            raise ValueError(f"A mapped Table 3 signature fell outside its declared universe: {key}")
        full[gi] = np.where(mapped.published_log2FC.to_numpy() > 0, 3, 2)
        common = comparable & (codes > 0)
        edger_common = np.where(common, codes, 0)
        paper_common = np.where(common, full, 0)
        effect = effects_by_key.get(key, effects.iloc[:0])
        ei = effect.gene_index.to_numpy(dtype=int)
        if (len(np.unique(ei)) != len(ei) or not np.array_equal(np.sort(ei), np.flatnonzero(codes >= 2))
                or not np.array_equal(codes[ei], np.where(effect.log2FC < 0, 2, np.where(effect.log2FC > 0, 3, 4)))):
            raise ValueError(f"edgeR significant rows disagree with cached gene codes: {key}")
        strict = codes.copy()
        strict[ei[effect.log2FC.abs().to_numpy() <= 0.25]] = 1
        refs[key] = {"edgeR_FC025": strict, "Cui_published": full,
                     "edgeR_common": edger_common, "Cui_common": paper_common}
        a, b = edger_common >= 2, paper_common >= 2
        both, either = a & b, a | b
        audit.append(dict(celltype=key[0], cyt_display=key[1], published_rows=len(genes), n_mapped=len(gi),
                          n_ambiguous=int(genes.ambiguous.sum()), n_unmapped=int(genes["index"].isna().sum() - genes.ambiguous.sum()),
                          n_common_universe=int(common.sum()), n_edger_common=int(a.sum()), n_cui_common=int(b.sum()),
                          n_cui_outside_tested=int(len(gi) - b.sum()), n_intersection=int(both.sum()),
                          n_union=int(either.sum()), n_same_direction=int((codes[both] == full[both]).sum()),
                          reference_jaccard=both.sum() / either.sum() if either.any() else np.nan))
    return refs, pd.DataFrame(audit)
