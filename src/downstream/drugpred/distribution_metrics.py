"""Train-fitted, model-independent distribution metrics for unpaired expression."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from il33_lung_metrics import evaluate, effect_order, correlation, r2

TOP_N = (20, 50, 100, 200)


def fit_reference(expression, mean, scale, functional, cfg):
    """Only Train expression may enter this function; no evaluation-driven bins/PCA."""
    rng = np.random.default_rng(cfg['split_seed'])
    rows = np.sort(rng.choice(len(expression), min(cfg['pca_training_cells'], len(expression)), replace=False))
    standardized = (np.array(expression[rows]) - mean) / scale
    dimensions = min(cfg['pca_dimensions'], len(rows) - 1, expression.shape[1])
    pca = PCA(dimensions, svd_solver='randomized', iterated_power=3, random_state=cfg['split_seed'])
    pca.fit(standardized)
    cuts = np.full((expression.shape[1], 7), np.inf, dtype=np.float32)
    bins = np.full(expression.shape[1], 2, dtype=np.int16)
    for left in range(0, expression.shape[1], 256):
        block = np.array(expression[:, left:left + 256])
        eligible = np.flatnonzero((block > 1e-8).sum(0) >= cfg['kl_minimum_positives'])
        if not len(eligible):
            continue
        values = block[:, eligible]; values[values <= 1e-8] = np.nan
        quantiles = np.nanquantile(values, np.arange(1, 8) / 8, axis=0)
        for j, col in enumerate(eligible):
            edges = np.unique(quantiles[:, j].astype(np.float32)); edges = edges[edges > 1e-8]
            cuts[left + col, :len(edges)] = edges
            bins[left + col] += len(edges)
    directions = np.random.default_rng(42).normal(size=(dimensions, cfg['sliced_directions']))
    directions /= np.linalg.norm(directions, axis=0)
    functional_directions = np.random.default_rng(42).normal(size=(len(functional), 32))
    functional_directions /= np.linalg.norm(functional_directions, axis=0)
    return dict(mean=mean, scale=scale, pca_components=pca.components_, pca_mean=pca.mean_,
        pca_training_rows=rows, directions=directions, functional_directions=functional_directions,
        functional_indices=np.asarray(functional), kl_cuts=cuts, kl_bins=bins)


def energy_distance(x, y):
    """Multivariate V-statistic, including zero diagonals; return distance and square."""
    def average(a, b):
        return sum(cdist(a[i:i + 128], b).sum() for i in range(0, len(a), 128)) / (len(a) * len(b))
    cross, within_x, within_y = average(x, y), average(x, x), average(y, y)
    squared = 2 * cross - within_x - within_y
    if squared < -1e-10 * max(1., cross, within_x, within_y):
        raise ValueError('Energy V-statistic is significantly negative')
    return float(np.sqrt(max(squared, 0))), float(max(squared, 0))


def marginal_kl(samples, cuts, bins, alpha):
    """Smoothed probabilities, with zero bin separate and unused padded bins masked."""
    index = np.ones(samples.shape, dtype=np.int16)
    for edge in cuts.T:
        index += samples >= edge
    index[samples <= 1e-8] = 0
    counts = np.bincount((index + np.arange(samples.shape[1]) * 9).ravel(),
                         minlength=samples.shape[1] * 9).reshape(-1, 9)
    return (counts + alpha) / (len(samples) + alpha * bins[:, None])


def distribution_scores(prediction, truth, reference, panels, alpha):
    """One matched-size sampling draw. The two populations have no paired cells."""
    if prediction.shape != truth.shape or not len(prediction):
        raise ValueError('Distribution draws need equal positive sample sizes and gene order')
    gene_w1 = np.abs(np.sort(prediction, axis=0) - np.sort(truth, axis=0)).mean(0, dtype=np.float64)
    p = marginal_kl(truth, reference['kl_cuts'], reference['kl_bins'], alpha)
    q = marginal_kl(prediction, reference['kl_cuts'], reference['kl_bins'], alpha)
    valid = np.arange(9)[None, :] < reference['kl_bins'][:, None]
    kl = np.maximum((p * np.log(p / q) * valid).sum(1), 0)
    wp = reference['pca_components'] / reference['scale'][None, :]
    offset = (reference['mean'] / reference['scale'] + reference['pca_mean']) @ reference['pca_components'].T
    px, py = prediction @ wp.T - offset, truth @ wp.T - offset
    e, e2 = energy_distance(px, py)
    projected_x, projected_y = px @ reference['directions'], py @ reference['directions']
    sliced = np.abs(np.sort(projected_x, axis=0) - np.sort(projected_y, axis=0)).mean()
    rows = []
    for panel, idx in panels.items():
        if not len(idx):
            continue
        row = dict(panel=panel, W1_gene_mean=float(gene_w1[idx].mean()),
                   KL_hist_true_pred=float(kl[idx].mean()))
        if panel == 'All_genes':
            row.update(E_distance_PCA50=e, E_squared_PCA50=e2, SW1_PCA50_128=float(sliced))
        if panel == 'Functional':
            a, b = prediction[:, idx] / reference['scale'][idx], truth[:, idx] / reference['scale'][idx]
            ef, ef2 = energy_distance(a, b)
            aa, bb = a @ reference['functional_directions'], b @ reference['functional_directions']
            row.update(E_distance_functional=ef, E_squared_functional=ef2,
                       functional_sliced_W1_32=float(np.abs(np.sort(aa, axis=0) - np.sort(bb, axis=0)).mean()))
        rows.append(row)
    return rows


def population_scores(prediction, truth, control, functional):
    """Full-cell population means and effect ranks; never p/q-value filtering."""
    observed = truth.mean(0, dtype=np.float64) - control.mean(0, dtype=np.float64)
    predicted = prediction.mean(0, dtype=np.float64) - control.mean(0, dtype=np.float64)
    order, predicted_order = effect_order(observed), effect_order(predicted)
    _, moments = evaluate(prediction, truth, control, order, predicted_order, top_n=TOP_N)
    panels = {'All_genes': np.arange(len(observed)), **{f'Top{n}': order[:n] for n in TOP_N},
              'Functional': np.asarray(functional, dtype=int)}
    rows = []
    for panel, idx in panels.items():
        a, b = observed[idx], predicted[idx]
        mse = float(np.mean((a - b) ** 2)) if len(idx) else None
        row = dict(panel=panel, n_genes=len(idx), MSE_mean=mse, MSE_delta=mse,
            RMSE_delta=float(np.sqrt(mse)) if mse is not None else None,
            PCC_delta=correlation(a, b) if len(idx) else None,
            R2_delta=r2(a, b) if len(idx) else None,
            R2_mean=r2(moments['observed_mean'][idx], moments['predicted_mean'][idx]) if len(idx) else None,
            R2_variance=r2(moments['observed_var'][idx], moments['predicted_var'][idx]) if len(idx) else None)
        if panel.startswith('Top'):
            n = int(panel[3:]); common = len(set(idx) & set(predicted_order[:n]))
            row.update(common_DEGs_count=common, common_DEGs_fraction=common / len(idx) if len(idx) else None,
                predicted_top_n=min(n, len(predicted_order)),
                direction_agreement=float(np.mean(np.sign(np.where(np.abs(b) < 1e-8, 0, b)) == np.sign(a))) if len(idx) else None)
        rows.append(row)
    return rows, moments, panels, order, predicted_order
