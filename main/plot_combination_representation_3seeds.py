"""Four rows: two joint stimuli, zero-shot and 5% joint supervision; source means."""
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

from plot_il33_expression_pca_controls import STYLES, METRICS

ROOT = Path(__file__).resolve().parents[1]
TASKS = [('GSE136154', 0., 'IL-33 + CGRP | Zero-shot'),
         ('GSE102299', 0., 'IL-25 + NMU | Zero-shot'),
         ('GSE136154', .05, 'IL-33 + CGRP | 5% joint supervision'),
         ('GSE102299', .05, 'IL-25 + NMU | 5% joint supervision')]
FIELDS = {'delta_RMSE': 'RMSE_delta', 'overlap': 'common_DEGs_fraction', 'delta_Pearson': 'PCC_delta'}


def plotted_values(data, dataset, fraction, model, panel, metric):
    selected = data[(data.dataset == dataset) & data.fraction.eq(fraction)
                    & data.model.eq(model) & data.panel.eq(panel)].sort_values('seed')
    expected = [0, 1, 2] if model in ('ExpressionMLP', 'PCA', 'GeneCompass', 'LoRA', 'Scratch_v3') else [-1]
    if selected.seed.tolist() != expected or not selected.source_count.eq(2).all():
        raise ValueError(f'Need both sources and real seeds {expected}: {dataset}/{fraction}/{model}/{panel}')
    return selected[metric].to_numpy(dtype=float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/combination_representation_3seeds_v1.json')
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    styles = dict(STYLES)
    if 'Scratch_v3' in cfg['models']:
        styles['Scratch_v3'] = ('Scratch v3', '#7B61A8')
    root = ROOT/cfg['result_directory']; complete = json.loads((root/'evaluation_complete.json').read_text())
    if complete['seeds'] != [0,1,2] or not complete['same_Test_for_zero_and_five']:
        raise ValueError('Finish all matched 0/5% evaluations before drawing')
    data = pd.read_csv(root/'source_macro_by_seed.csv')
    out = root/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .8})
    fig, axes = plt.subplots(4,4,figsize=(22,13.5) if len(styles)==7 else (17.4,12.5))
    fig.subplots_adjust(left=.065,right=.99,bottom=.045,top=.925,wspace=.55,hspace=.6)
    records = []
    for row,(dataset,fraction,title) in enumerate(TASKS):
        for ax,(panel,metric,ylabel,scale,fmt) in zip(axes[row],METRICS):
            marks = []
            for x,style in enumerate(styles):
                model = ('mean_delta_additive' if fraction == 0 else 'fewshot_mean_delta') if style == 'mean_delta' else style
                values = plotted_values(data,dataset,fraction,model,panel,FIELDS[metric])*scale
                marks.append((x,style,values))
            finite = [v for _,_,values in marks for v in values if np.isfinite(v)]
            upper = max(finite)*1.3 if finite and max(finite)>0 else 1.
            lower = min(0,min(finite)*1.15) if finite else 0.
            if metric == 'overlap': upper=115
            elif metric == 'delta_Pearson': upper=1.12
            ax.set(xlim=(-.65,len(styles)-.35),ylim=(lower,upper),xticks=[],ylabel=ylabel)
            for x,style,values in marks:
                value = np.nan if np.isnan(values).all() else np.nanmean(values)
                if not np.isfinite(value):
                    ax.text(x,upper*.025,'NA',ha='center',fontsize=8)
                else:
                    bars = ax.bar(x,value,color=styles[style][1],width=.62,zorder=2)
                    ax.bar_label(bars,labels=[format(value,fmt)],padding=5,fontsize=8,rotation=0)
                    if len(values)==3:
                        ax.scatter(x+np.array([-.14,0,.14]),values,c='#252525',s=13,
                                   edgecolors='white',linewidths=.35,zorder=3)
                for seed,v in zip([0,1,2] if len(values)==3 else [-1],values):
                    records.append(dict(dataset=dataset,fraction=fraction,model=style,seed=seed,
                                        panel=panel,metric=metric,displayed_value=v,scale=scale))
            ax.set_title(title,fontsize=10,pad=12)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(axis='y',labelsize=9); ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.legend(handles=[Patch(color=c,label=label) for label,c in styles.values()],
               loc='upper center',bbox_to_anchor=(.54,.99),ncol=len(styles),frameon=False,fontsize=12)
    stem = 'combination_seven_models_3seeds_4x4' if len(styles)==7 else 'combination_six_models_3seeds_4x4'
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
    pd.DataFrame(records).to_csv(out/'plot_data.csv',index=False)
    plt.close(fig)
    print('PLOT',out/(stem+'.png'),flush=True)


if __name__ == '__main__':
    main()
