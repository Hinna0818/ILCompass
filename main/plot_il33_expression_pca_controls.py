"""Six-method IL33 figure with exactly three genuine training-seed dots."""
import argparse
import json
import os
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    'control_copy': ('Copy-control', '#B6BDC7'), 'mean_delta': ('Mean Δ', '#55A797'),
    'ExpressionMLP': ('Expression MLP', '#9274AE'), 'PCA': ('PCA + MLP', '#C89A35'),
    'GeneCompass': ('GeneCompass', '#2878B5'), 'LoRA': ('LoRA', '#D95335'),
}
METRICS = [('Top100', 'delta_RMSE', 'Top-100 Δ-RMSE ↓', 1, '.4f'),
           ('Top100', 'overlap', 'Top-100 overlap (%) ↑', 100, '.2f'),
           ('All_genes', 'delta_RMSE', 'Overall Δ-RMSE ↓', 1, '.5f'),
           ('All_genes', 'delta_Pearson', 'PCC-delta ↑', 1, '.4f')]
TASKS = [('IL33_lung_mixed_internal', 'Within-lung random holdout', 'background_mean_delta'),
         ('IL33_C_to_D', 'C → D cross-batch', 'C_mean_delta'),
         ('IL33_liver_adaptation', 'Lung → liver adaptation', None),
         ('IL33_cross_study_target_PBS', 'Cross-study transfer', 'global_mean_delta')]


def select_values(data, experiment, adaptation, model, panel, metric):
    selected = data[(data.experiment == experiment) & (data.adaptation == adaptation)
                    & (data.model == model) & (data.panel == panel)].sort_values('seed')
    if model in ('GeneCompass', 'LoRA', 'ExpressionMLP', 'PCA', 'Scratch_v3'):
        if selected.seed.tolist() != [0, 1, 2]:
            raise ValueError(f'Need exactly seeds0/1/2: {experiment}/{adaptation}/{model}/{panel}')
    elif selected.seed.tolist() != [-1]:
        raise ValueError('Deterministic baselines must not be duplicated as three seed runs')
    return selected[metric].to_numpy(dtype=float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/il33_expression_pca_controls_v1.json')
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    styles = dict(STYLES)
    if cfg.get('include_scratch', False):
        styles['Scratch_v3'] = ('Scratch v3', '#7B61A8')
    methods = tuple(m for m in styles if m not in ('control_copy', 'mean_delta'))
    group_width = len(methods)+.8
    base = ROOT / cfg['result_directory']; out = base / 'figures'; out.mkdir(parents=True, exist_ok=True)
    complete = json.loads((base / 'evaluation_complete.json').read_text())
    if complete['seeds'] != [0, 1, 2] or not complete['original_results_unchanged']:
        raise ValueError('Finish matched three-seed evaluation before drawing')
    data = pd.read_csv(base / 'metrics_by_seed.csv')
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .8})
    fig, axes = plt.subplots(4, 4, figsize=(24, 15) if cfg.get('include_scratch') else (18, 13.5))
    fig.subplots_adjust(left=.065, right=.99, bottom=.045, top=.925, wspace=.58, hspace=.62)
    records = []
    for row, (experiment, title, mean_method) in enumerate(TASKS):
        for ax, (panel, metric, ylabel, scale, fmt) in zip(axes[row], METRICS):
            marks = []
            if mean_method is not None:
                for x, style in enumerate(styles):
                    model = mean_method if style == 'mean_delta' else style
                    values = select_values(data, experiment, 'none', model, panel, metric) * scale
                    marks.append((x, style, values))
                ax.set(xlim=(-.65, len(styles)-.35), xticks=[])
            else:
                for gi, adaptation in enumerate(('Zero-shot', '5%', '10%')):
                    for mi, model in enumerate(methods):
                        values = select_values(data, experiment, adaptation, model, panel, metric) * scale
                        marks.append((gi*group_width+mi-.5*(len(methods)-1), model, values))
                ax.set_xlim(-.5*(len(methods)-1)-.8, 2*group_width+.5*(len(methods)-1)+.8)
                ax.set_xticks([0, group_width, 2*group_width], ['Zero-shot', '5%', '10%'], fontsize=9)
            finite = [v for _, _, vv in marks for v in vv if np.isfinite(v)]
            upper = max(finite) * 1.35 if finite and max(finite) > 0 else 1
            lower = min(0, min(finite)*1.15) if finite else 0
            if metric == 'overlap':
                upper = 115
            elif metric == 'delta_Pearson':
                upper = 1.12
            ax.set_ylim(lower, upper)
            for i, (x, model, values) in enumerate(marks):
                value = np.nan if np.isnan(values).all() else np.nanmean(values)
                if not np.isfinite(value):
                    ax.text(x, upper*.025, 'NA', ha='center', fontsize=8)
                    continue
                bars = ax.bar(x, value, color=styles[model][1], width=.62 if mean_method else .67, zorder=2)
                if mean_method:
                    ax.bar_label(bars, labels=[format(value, fmt)], rotation=0,
                                 padding=5 if i % 2 == 0 else 16, fontsize=8)
                else:
                    # Align each four-model group's labels on two rows to avoid collisions.
                    count = len(methods)
                    group = marks[(i//count)*count:(i//count+1)*count]
                    peak = max(np.max(vv) for _, _, vv in group)
                    rows = 3 if count == 5 else 2
                    label_y = peak + (upper-lower)*(.03 + .045*((i % count) % rows))
                    ax.text(x, label_y, format(value, fmt), ha='center', va='bottom', fontsize=7)
                if len(values) == 3:
                    ax.scatter(x+np.array([-.14, 0, .14]), values, c='#252525', s=13,
                               edgecolors='white', linewidths=.35, zorder=3)
                for seed, v in zip([0, 1, 2] if len(values) == 3 else [-1], values):
                    records.append(dict(experiment=experiment, panel=panel, metric=metric,
                                        model=model, x=x, seed=seed, displayed_value=v, scale=scale))
            ax.set_title(title, fontsize=10, pad=12); ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False); ax.tick_params(axis='y', labelsize=9)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.legend(handles=[Patch(color=c, label=label) for label, c in styles.values()],
               loc='upper center', bbox_to_anchor=(.54, .99), ncol=len(styles), frameon=False, fontsize=11)
    stem = 'IL33_seven_models_3seeds_4x4' if cfg.get('include_scratch') else 'IL33_six_models_3seeds_4x4'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    pd.DataFrame(records).to_csv(out / 'plot_data.csv', index=False)
    plt.close(fig)
    print('PLOT', out / f'{stem}.png', flush=True)


if __name__ == '__main__':
    main()
