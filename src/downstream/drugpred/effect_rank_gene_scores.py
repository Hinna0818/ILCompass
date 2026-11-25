"""Shared post-hoc scores from saved gene moments, ranked by effect magnitude."""
import numpy as np
from il33_lung_metrics import effect_order, effect_rank_metrics, correlation, r2


def score_gene_moments(record, table, functional_ids, top_n=(20, 50, 100, 200)):
    truth, prediction = table.observed_delta.to_numpy(), table.predicted_delta.to_numpy()
    response = effect_rank_metrics(truth, prediction, top_n)
    order = effect_order(truth)
    fi = table.gene_id.isin(functional_ids).to_numpy()
    rows = []
    for panel, idx in [('All_genes', np.arange(len(table)))] + [(f'Top{n}', order[:n]) for n in top_n]:
        a, b = truth[idx], prediction[idx]
        row = {**record, 'panel': panel, 'n_genes': len(idx),
            'delta_MSE': float(np.mean((a - b) ** 2)), 'delta_R2': r2(a, b),
            'delta_Pearson': correlation(a, b),
            'mean_R2': r2(table.observed_mean.iloc[idx], table.predicted_mean.iloc[idx]),
            'variance_R2': r2(table.observed_var.iloc[idx], table.predicted_var.iloc[idx]),
            'direction_accuracy': float(np.mean(np.sign(a) == np.sign(b)))}
        row['delta_RMSE'] = np.sqrt(row['delta_MSE'])
        row['overlap'] = np.nan if panel == 'All_genes' else response[f'effect_rank_overlap_{int(panel[3:])}']
        row['functional_MSE'] = float(np.mean((truth[fi] - prediction[fi]) ** 2)) if fi.any() else np.nan
        rows.append(row)
    return rows, order
