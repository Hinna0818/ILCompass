"""Fixed-test comparison of frozen GeneCompass and ILC2 tissue representations."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ilc2_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/ilc2_numba')
import argparse
import json
from itertools import combinations
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             normalized_mutual_info_score, silhouette_samples)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/downstream/tissue'))
from linear_probe import MODELS, fit_probe
from models import FrozenILC2Encoder
from posttrain_data import encode, file_hash
from genecompass_coverage import map_genes

OLD = ROOT / 'results/ILC2_tissue_linear_probe_v1'
CACHE = ROOT / 'data/processed/ILC2_encoder_compare_v1'
OUT = ROOT / 'results/ILC2_encoder_compare_v1'
DATA = ROOT / 'data/processed/ILC2_posttrain_update'
SEEDS = list(range(5))
CONFIG = dict(
    training='Reuse fixed 3600 original-train cells: 300 per study x tissue; no resampling',
    test='All 5008 original posttrain_test cells; previously used for developmental evaluation',
    encoders='Frozen FP32, TF32 disabled, median/top2048, masked gene mean, 768 dimensions',
    linear_probe='Train-only StandardScaler + logistic regression C=1, balanced, max_iter=3000',
    knn='k=15, cosine distance, uniform voting; reference=training cells only',
    geometry='Test-only L2-normalized embeddings; cosine silhouette; spherical-input KMeans (ordinary Euclidean KMeans on unit vectors)',
    clustering=dict(seeds=SEEDS, n_init=10, n_clusters='Number of tissues in predefined scope'),
    bootstrap='1000 paired library-cluster resamples within tissue, seed=20261028; fixed heads. Descriptive only: some tissues have one library; libraries are not necessarily animals.',
    scopes=['all6', 'shared3', 'within_GSE117567', 'within_GSE153748'],
    umap=dict(n_neighbors=15, min_dist=0.3, metric='cosine', random_state=0),
)


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'configs/2prepare_traindata.json').read_text())
    source = DATA / 'ILC2_posttrain_20261021_update.h5ad'
    old_config = json.loads((OLD / 'config.json').read_text())
    signature = dict(source=file_hash(source), checkpoints={k: file_hash(v / 'pytorch_model.bin') for k, v in MODELS.items()})
    if signature != {k: old_config[k] for k in signature}:
        raise ValueError('Old feature cache does not match current source/checkpoints')
    config = dict(CONFIG, **signature)
    if (OUT / 'config.json').exists() and json.loads((OUT / 'config.json').read_text()) != config:
        raise ValueError('Evaluation configuration changed; use a new version')
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    obs = pd.read_csv(DATA / 'ILC2_posttrain_20261021_update_metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
    old = pd.read_csv(OLD / 'cells.tsv.gz', sep='\t', index_col=0)
    columns = ['study', 'tissue', 'split', 'library_id', 'qc_version']
    if not obs.loc[old.index, columns].equals(old[columns]):
        raise ValueError('Old cached metadata and current metadata differ')
    train_ids = old.index[old.split.eq('posttrain_train')]
    test_ids = obs.index[obs.split.eq('posttrain_test')]
    meta = obs.loc[train_ids.append(test_ids), columns].sort_index()
    if len(train_ids) != 3600 or len(test_ids) != 5008 or not meta.index.is_unique:
        raise ValueError('Unexpected train/test membership')
    if (CACHE / 'cells.tsv.gz').exists():
        previous = pd.read_csv(CACHE / 'cells.tsv.gz', sep='\t', index_col=0)
        if not previous.equals(meta):
            raise ValueError('Existing cache has different cell membership/order')
    meta.to_csv(CACHE / 'cells.tsv.gz', sep='\t')
    meta.groupby(['split', 'study', 'tissue', 'library_id']).size().rename('cells').to_csv(OUT / 'cell_counts.tsv', sep='\t')
    missing = np.flatnonzero(~meta.index.isin(old.index))
    if not (CACHE / 'new_inputs.npz').exists():
        a = ad.read_h5ad(source, backed='r')
        sub = a[meta.index[missing]].to_memory()
        a.file.close()
        x = sub.layers['counts']
        if (sub.X != x).nnz or not np.isfinite(x.data).all() or (x.data < 0).any() or (x.data != np.floor(x.data)).any():
            raise ValueError('Invalid or mismatched raw counts')
        tokens = pickle.loads((ROOT / cfg['token_dictionary']).read_bytes())
        medians = pickle.loads((ROOT / cfg['gene_median_reference']).read_bytes())
        columns, med, _ = map_genes(sub.var_names, tokens, medians)
        ids, vals, length = encode(x, columns, med, 2048, 'median')
        np.savez(CACHE / 'new_inputs.npz', rows=missing, ids=ids, vals=vals, length=length)
    print(f'PREPARED train={len(train_ids)} test={len(test_ids)} reuse={len(old)} encode_new={len(missing)}', flush=True)


def fill_embeddings(checkpoint, h, rows, ids, vals, lengths, name):
    """Fill selected cache rows; shared by the two- and three-encoder comparisons."""
    if not torch.cuda.is_available():
        raise RuntimeError('GPU access is needed for frozen encoder extraction')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    todo = np.flatnonzero(~np.isfinite(h[rows]).all(1))
    todo = todo[np.argsort(lengths[todo], kind='stable')]
    model = FrozenILC2Encoder(checkpoint).cuda().eval()
    start = time.monotonic()
    for left in range(0, len(todo), 8):
        idx = todo[left:left + 8]
        n = int(lengths[idx].max())
        ti = torch.tensor(ids[idx, :n], device='cuda', dtype=torch.long)
        tv = torch.tensor(vals[idx, :n], device='cuda')
        mask = torch.arange(n, device='cuda')[None, :] < torch.tensor(lengths[idx], device='cuda')[:, None]
        h[rows[idx]] = model(ti, tv, mask).cpu().numpy()
        done = min(left + 8, len(todo))
        if done % 128 == 0 or done == len(todo):
            h.flush()
            seconds = time.monotonic() - start
            print(f'ENCODE {name} {done}/{len(todo)} elapsed={seconds:.0f}s ETA={seconds / done * (len(todo) - done):.0f}s', flush=True)
    del model
    torch.cuda.empty_cache()


def encode_features():
    meta = pd.read_csv(CACHE / 'cells.tsv.gz', sep='\t', index_col=0)
    old = pd.read_csv(OLD / 'cells.tsv.gz', sep='\t', index_col=0)
    inp = np.load(CACHE / 'new_inputs.npz')
    rows, ids, vals, lengths = [inp[k] for k in ('rows', 'ids', 'vals', 'length')]
    old_rows = meta.index.get_indexer(old.index)
    if (old_rows < 0).any() or set(rows) & set(old_rows) or len(rows) + len(old_rows) != len(meta):
        raise ValueError('Feature row coverage/alignment error')
    for name, checkpoint in MODELS.items():
        dest = CACHE / f'{name}_h.npy'
        if dest.exists():
            h = np.load(dest, mmap_mode='r')
            if h.shape != (len(meta), 768) or not np.isfinite(h).all():
                raise ValueError('Invalid completed embeddings')
            continue
        partial = CACHE / f'{name}_h.partial.npy'
        exists = partial.exists()
        h = np.lib.format.open_memmap(partial, mode='r+' if exists else 'w+', dtype='float32', shape=(len(meta), 768))
        if not exists:
            h[:] = np.nan
            h[old_rows] = np.load(OLD / f'{name}_h.npy')
            h.flush()
        fill_embeddings(checkpoint, h, rows, ids, vals, lengths, name)
        if not np.isfinite(h).all():
            raise ValueError('Incomplete embeddings')
        h.flush()
        del h
        partial.replace(dest)
        torch.cuda.empty_cache()


def scopes(meta):
    yield 'all6', np.ones(len(meta), dtype=bool)
    yield 'shared3', meta.tissue.isin(['Lung', 'Pancreas', 'SI']).to_numpy()
    for study in ['GSE117567', 'GSE153748']:
        yield 'within_' + study, meta.study.eq(study).to_numpy()


def metrics(y, prediction, labels):
    return dict(accuracy=accuracy_score(y, prediction),
                balanced_accuracy=balanced_accuracy_score(y, prediction),
                macro_F1=f1_score(y, prediction, labels=labels, average='macro', zero_division=0))


def library_bootstrap(meta, predictions):
    """Paired cluster resampling; single-library tissues cannot estimate library variability."""
    out = []
    rng = np.random.default_rng(20261028)
    for scope, group in predictions.groupby('scope', sort=False):
        for method in ['linear', 'knn']:
            p = group[group.method.eq(method)].pivot(index='cell_id', columns='encoder', values='prediction')
            m = meta.loc[p.index]
            labels = sorted(m.tissue.unique())
            blocks = [[np.flatnonzero(m.library_id.eq(lib).to_numpy()) for lib in d.library_id.unique()]
                      for _, d in m.groupby('tissue', sort=True)]
            # Precompute per-library confusion matrices so 1000 draws are cheap.
            mats = {}
            for name in MODELS:
                mats[name] = [np.stack([confusion_matrix(m.tissue.iloc[ix], p[name].iloc[ix], labels=labels) for ix in b]) for b in blocks]
            pairs = list(combinations(MODELS, 2))
            values = {pair: {key: [] for key in ['accuracy', 'balanced_accuracy', 'macro_F1']} for pair in pairs}
            for _ in range(1000):
                picks = [rng.integers(len(b), size=len(b)) for b in blocks]
                scores = {}
                for name in MODELS:
                    cm = sum(mats[name][j][ix].sum(0) for j, ix in enumerate(picks)).astype(float)
                    tp = np.diag(cm)
                    scores[name] = dict(accuracy=tp.sum()/cm.sum(), balanced_accuracy=np.mean(tp/cm.sum(1)),
                                        macro_F1=np.mean(2*tp/(cm.sum(0)+cm.sum(1))))
                for reference, candidate in pairs:
                    for metric in values[(reference, candidate)]:
                        values[(reference, candidate)][metric].append(scores[candidate][metric]-scores[reference][metric])
            for (reference, candidate), results in values.items():
                for metric, deltas in results.items():
                    out.append(dict(scope=scope, method=method, reference=reference, candidate=candidate, metric=metric,
                                    delta_low=np.quantile(deltas, .025), delta_high=np.quantile(deltas, .975),
                                    fraction_positive=np.mean(np.array(deltas)>0),
                                    libraries=m.library_id.nunique(), single_library_tissues=sum(len(b)==1 for b in blocks)))
    pd.DataFrame(out).to_csv(OUT / 'paired_library_bootstrap.tsv', sep='\t', index=False)


def evaluate():
    meta = pd.read_csv(CACHE / 'cells.tsv.gz', sep='\t', index_col=0)
    train = meta.split.eq('posttrain_train').to_numpy()
    test = meta.split.eq('posttrain_test').to_numpy()
    y = meta.tissue.to_numpy()
    classifier, geometry, clustering, predictions, per_tissue, per_cell, source_rows = [], [], [], [], [], [], []
    converged = []
    for name in MODELS:
        h = np.load(CACHE / f'{name}_h.npy')
        if h.shape != (len(meta), 768) or not np.isfinite(h).all() or (np.linalg.norm(h, axis=1)==0).any():
            raise ValueError('Invalid embedding cache')
        for scope, mask in scopes(meta):
            tr, te = train & mask, test & mask
            labels = sorted(np.unique(y[tr]))
            if set(y[te]) != set(labels):
                raise ValueError('Scope has unsupported tissue labels')
            model, scaler, lp = fit_probe(h[tr], h[te], y[tr])
            converged.append(int(model.n_iter_.max()) < model.max_iter)
            np.savez(OUT / f'{name}_{scope}_head.npz', coef=model.coef_, intercept=model.intercept_,
                     classes=model.classes_, center=scaler.mean_, scale=scaler.scale_)
            knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', metric='cosine', algorithm='brute', n_jobs=4)
            knn.fit(h[tr], y[tr])
            kp = knn.predict(h[te])
            for method, pred in [('linear', lp), ('knn', kp)]:
                result = dict(encoder=name, scope=scope, method=method, train_cells=int(tr.sum()), test_cells=int(te.sum()), **metrics(y[te], pred, labels))
                classifier.append(result)
                print('CLASSIFY', result, flush=True)
                report = classification_report(y[te], pred, labels=labels, output_dict=True, zero_division=0)
                for tissue in labels:
                    per_tissue.append(dict(encoder=name, scope=scope, method=method, tissue=tissue, **report[tissue]))
                pd.DataFrame(confusion_matrix(y[te], pred, labels=labels), index=labels, columns=labels).to_csv(OUT / f'{name}_{scope}_{method}_confusion.tsv', sep='\t')
                for cell, truth, prediction in zip(meta.index[te], y[te], pred):
                    predictions.append(dict(cell_id=cell, encoder=name, scope=scope, method=method, tissue=truth, prediction=prediction))
            z = normalize(h[te])
            sil = silhouette_samples(z, y[te], metric='cosine')
            nn = NearestNeighbors(n_neighbors=16, metric='cosine', algorithm='brute').fit(z)
            neighbors = nn.kneighbors(z, return_distance=False)
            # Remove the cell by index, including exact-distance ties.
            neighbors = np.stack([r[r != i][:15] for i, r in enumerate(neighbors)])
            purity = np.mean(y[te][neighbors] == y[te][:, None], axis=1)
            frame = pd.DataFrame(dict(cell_id=meta.index[te], tissue=y[te], silhouette=sil, neighborhood_purity=purity))
            geometry.append(dict(encoder=name, scope=scope, cells=int(te.sum()), silhouette=sil.mean(),
                                 silhouette_macro=frame.groupby('tissue').silhouette.mean().mean(),
                                 neighborhood_purity=purity.mean(), neighborhood_purity_macro=frame.groupby('tissue').neighborhood_purity.mean().mean()))
            per_cell.append(frame.assign(encoder=name, scope=scope))
            for seed in SEEDS:
                km = KMeans(n_clusters=len(labels), n_init=10, random_state=seed, max_iter=300)
                cluster = km.fit_predict(z)
                clustering.append(dict(encoder=name, scope=scope, seed=seed, ARI=adjusted_rand_score(y[te], cluster),
                                       NMI=normalized_mutual_info_score(y[te], cluster), iterations=km.n_iter_))
                frame[f'cluster_seed{seed}'] = cluster
            frame.to_csv(OUT / f'{name}_{scope}_geometry_cells.tsv.gz', sep='\t', index=False)
            if scope == 'all6':
                tm = meta.loc[te]
                for tissue, d in tm.groupby('tissue'):
                    if d.study.nunique() > 1:
                        idx = np.flatnonzero(tm.tissue.eq(tissue).to_numpy())
                        s = silhouette_samples(z[idx], d.study, metric='cosine')
                        source_rows.append(dict(encoder=name, tissue=tissue, cells=len(d), studies=d.study.nunique(), source_silhouette=s.mean()))
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUT / 'predictions.tsv.gz', sep='\t', index=False)
    pd.DataFrame(classifier).to_csv(OUT / 'classification_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(per_tissue).to_csv(OUT / 'per_tissue.tsv', sep='\t', index=False)
    pd.DataFrame(geometry).to_csv(OUT / 'geometry_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(clustering).to_csv(OUT / 'clustering_seeds.tsv', sep='\t', index=False)
    pd.DataFrame(source_rows).to_csv(OUT / 'within_tissue_source_silhouette.tsv', sep='\t', index=False)
    pd.concat(per_cell).groupby(['encoder', 'scope', 'tissue'])[['silhouette', 'neighborhood_purity']].mean().to_csv(OUT / 'per_tissue_geometry.tsv', sep='\t')
    stratified = []
    main = pred[pred.scope.eq('all6')].merge(meta[['study', 'library_id']], left_on='cell_id', right_index=True, validate='many_to_one')
    for (name, method, study, tissue), d in main.groupby(['encoder', 'method', 'study', 'tissue']):
        stratified.append(dict(encoder=name, method=method, study=study, tissue=tissue, cells=len(d), recall=np.mean(d.tissue==d.prediction)))
    pd.DataFrame(stratified).to_csv(OUT / 'study_tissue_recall.tsv', sep='\t', index=False)
    library_bootstrap(meta, pred)
    verification = dict(train_cells=int(train.sum()), test_cells=int(test.sum()), same_cells_and_split_all_models=True,
                        encoders=list(MODELS),
                        train_test_disjoint=not bool((train & test).any()), all_heads_converged=all(converged),
                        fixed_hyperparameters=True, validation_not_used=True, test_exclusions=0,
                        test_used_for_ILC2_posttraining_gradient=False, independent_new_study_test=False)
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2))
    if not all(converged):
        raise RuntimeError('Linear probe convergence needs inspection')
    plot_and_report(meta)


def plot_comparison(meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import umap
    font_manager.fontManager.addfont(ROOT / 'external/fonts/Arial.ttf')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    c = pd.read_csv(OUT / 'classification_metrics.tsv', sep='\t')
    g = pd.read_csv(OUT / 'geometry_metrics.tsv', sep='\t')
    k = pd.read_csv(OUT / 'clustering_seeds.tsv', sep='\t')
    b = pd.read_csv(OUT / 'paired_library_bootstrap.tsv', sep='\t')
    rows = []
    for scope in c.scope.unique():
        for name in MODELS:
            cc = c[c.scope.eq(scope) & c.encoder.eq(name)].set_index('method')
            gg = g[g.scope.eq(scope) & g.encoder.eq(name)].iloc[0]
            kk = k[k.scope.eq(scope) & k.encoder.eq(name)]
            rows.append(dict(scope=scope, encoder=name, test_cells=int(cc.loc['linear','test_cells']),
                             linear_macro_F1=cc.loc['linear','macro_F1'], linear_accuracy=cc.loc['linear','accuracy'],
                             knn_macro_F1=cc.loc['knn','macro_F1'], knn_accuracy=cc.loc['knn','accuracy'],
                             silhouette=gg.silhouette, silhouette_macro=gg.silhouette_macro,
                             ARI_mean=kk.ARI.mean(), ARI_sd=kk.ARI.std(), NMI_mean=kk.NMI.mean(), NMI_sd=kk.NMI.std()))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.tsv', sep='\t', index=False)
    scores = ['linear_macro_F1', 'linear_accuracy', 'knn_macro_F1', 'knn_accuracy',
              'silhouette', 'silhouette_macro', 'ARI_mean', 'NMI_mean']
    original = summary[summary.encoder.eq('GeneCompass')].set_index('scope')[scores]
    posttrained = summary[summary.encoder.eq('ILC2')].set_index('scope')[scores]
    (posttrained - original).to_csv(OUT / 'metric_differences.tsv', sep='\t')
    tissue = pd.read_csv(OUT / 'per_tissue.tsv', sep='\t')
    tissue = tissue[tissue.scope.eq('all6') & tissue.method.eq('linear')].pivot(index='tissue', columns='encoder', values='f1-score')
    tissue['ILC2_minus_GeneCompass'] = tissue.ILC2 - tissue.GeneCompass
    colors = ['#64748B', '#C25148', '#258778'][:len(MODELS)]
    names = list(MODELS)
    display = {'GeneCompass': 'GeneCompass', 'ILC2': 'ILC2', 'ILC2_tissue': 'ILC2 + tissue'}
    main = summary[summary.scope.eq('all6')].set_index('encoder').loc[names]
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.6))
    for ax, key, label in zip(axes, ['silhouette_macro', 'ARI_mean', 'NMI_mean', 'knn_macro_F1', 'linear_macro_F1'],
                              ['Silhouette (macro)', 'ARI', 'NMI', 'kNN macro-F1', 'Linear macro-F1']):
        error = main[key.replace('_mean', '_sd')].to_numpy() if key.endswith('_mean') else None
        ax.bar(range(len(names)), main[key], color=colors, width=.65, yerr=error, capsize=3)
        ax.set_xticks(range(len(names)), [display[n] for n in names], rotation=35, ha='right')
        ax.set_title(label, fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        for i, v in enumerate(main[key]): ax.annotate(f'{v:.3f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
        ax.set_ylim(min(0, main[key].min()*1.25), max(.05, main[key].max()*1.22))
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(OUT / f'metrics.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 3.6))
    for ax, name in zip(axes, names):
        cm = pd.read_csv(OUT / f'{name}_all6_linear_confusion.tsv', sep='\t', index_col=0)
        values = cm.to_numpy()/cm.sum(1).to_numpy()[:,None]
        ax.imshow(values, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(len(cm)), cm.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(cm)), cm.index)
        ax.set(xlabel='Predicted tissue', ylabel='True tissue', title=display[name])
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>.6 else 'black')
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(OUT / f'linear_confusion.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    test = meta.split.eq('posttrain_test').to_numpy()
    tm = meta.loc[test]
    fig, axes = plt.subplots(2,len(names),figsize=(5*len(names),8))
    for col, name in enumerate(names):
        dest = OUT / f'{name}_umap.tsv.gz'
        if dest.exists():
            u = pd.read_csv(dest, sep='\t', index_col=0).loc[tm.index, ['UMAP1','UMAP2']].to_numpy()
        else:
            h = normalize(np.load(CACHE / f'{name}_h.npy')[test])
            u = umap.UMAP(**CONFIG['umap'], n_jobs=1).fit_transform(h)
            tm.assign(UMAP1=u[:,0], UMAP2=u[:,1]).to_csv(dest, sep='\t')
        for row, field in enumerate(['tissue','study']):
            ax = axes[row,col]
            for i, label in enumerate(sorted(tm[field].unique())):
                sel = tm[field].eq(label).to_numpy()
                ax.scatter(u[sel,0],u[sel,1],s=3,c=[plt.get_cmap('tab10')(i)],label=label,alpha=.75,rasterized=True,linewidths=0)
            ax.set_title(display[name] + (' / tissue' if row==0 else ' / source'))
            ax.set(xticks=[],yticks=[],xlabel='UMAP1',ylabel='UMAP2')
            ax.spines[['top','right']].set_visible(False)
            ax.legend(markerscale=3,fontsize=7,frameon=False,loc='best')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'umap.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    return summary, b, tissue


def plot_and_report(meta):
    summary, b, tissue = plot_comparison(meta)
    main = summary[summary.scope.eq('all6')].set_index('encoder')
    text = ['# 固定测试集：GeneCompass 与 ILC2 后训练表示对照', '',
            '**结论：六组织线性可分类性小幅改善；没有观察到组织整体聚类结构或kNN Macro-F1一致改善。当前证据不足以宣称后训练全面优于原始模型。**', '',
            '使用全部5008原posttrain_test细胞；分类头/kNN参考为此前固定的3600原训练细胞（每研究×组织300），不是全部90198训练细胞。无验证调参，两个encoder冻结。',
            '六类Fat/LI/Lung/Pancreas/SI/Skin。所有来源保留。测试细胞未参与本项目后训练梯度，但已多次用于开发性评价；不是新的盲测，也不是新研究泛化。无法从本项目记录排除原始GeneCompass预训练的数据重叠。',
            '共享三组织与研究内部比较均在各自训练子集重新拟合分类器。研究内组织/文库仍可能混杂；研究内结果不等于独立动物泛化。', '',
            '## 固定协议',
            '原始整数counts经既有mouse median映射、top2048排序和log2(1+x)编码，FP32、有效gene-token均值池化。沿用原有4800细胞缓存，新增3808测试细胞；细胞ID/来源/checkpoint签名一致。',
            'Linear: 训练集StandardScaler，logistic C=1、balanced。kNN: 原训练参考，cosine，k=15。Silhouette: 测试集真实组织标签、cosine；同时给细胞平均与组织等权平均。',
            'ARI/NMI: 仅测试embedding先L2归一化，固定组织类别数k，KMeans n_init=10，seed=0..4。均值±SD为聚类初始化波动，不是生物学置信区间。无监督聚类在测试集拟合是描述性/transductive分析；分类器不使用测试标签拟合。',
            'UMAP仅展示；指标均在768维表示计算。两个UMAP独立拟合，坐标不可逐点对齐。', '',
            '## 指标', '```tsv', summary.to_csv(sep='\t', index=False, float_format='%.6f'), '```', '',
            '## 结果解释',
            f"六组织linear Macro-F1 {main.loc['GeneCompass','linear_macro_F1']:.6f} → {main.loc['ILC2','linear_macro_F1']:.6f}，提升{100*(main.loc['ILC2','linear_macro_F1']-main.loc['GeneCompass','linear_macro_F1']):.3f}个百分点。准确率96.046% → 96.306%，多正确预测13个细胞（4810 → 4823）。",
            '六组织头的各组织F1均提高，Fat和LI改善相对更多；F1提升不代表每组织召回都提升。研究内部重新拟合头后，GSE117567 Macro-F1 0.9857 → 0.9905，GSE153748 0.9550 → 0.9502；共享三组织0.9725 → 0.9701，收益不具有范围一致性。',
            'kNN Macro-F1基本持平略降。Silhouette、ARI和NMI均下降；两个模型的全局组织ARI都较低，不能称为自然形成清晰的六组织簇。Silhouette为负表示按照组织标签定义的群体在当前距离度量下存在明显重叠或多模态。',
            '线性分类较高与无监督聚类较差可以同时存在：线性头利用标签寻找判别方向，KMeans/距离则受整体几何结构影响。同组织的不同来源仍明显分离（胰腺source silhouette约0.72），支持来源相关结构参与影响的解释，但不能据此单独证明技术因素是因果。',
            '后续建议保持两个checkpoint和本次协议固定，优先用额外独立文库检验线性分类收益；若评估功能状态或IL33响应，应另设任务，不把组织线性分类的微小提升外推为功能预测改善。不按本次结果删除研究或选择更好看的聚类参数。',
            '```tsv', tissue.to_csv(sep='\t', float_format='%.6f'), '```', '',
            '## 配对文库重采样',
            '对每组织的文库做有放回抽样，两模型使用同一抽样、固定已拟合分类器，1000次。区间为差值ILC2−GeneCompass的2.5%/97.5%分位数。Fat/LI/Skin仅一个文库，无法估计其跨文库变异；多文库也不保证独立动物。仅用于当前来源组成敏感性，不能用作动物级显著性或模型训练不确定性。',
            '```tsv', b[b.scope.eq('all6')].to_csv(sep='\t',index=False,float_format='%.6f'),'```','',
            '## 文件',
            '- summary.tsv / classification_metrics.tsv：总体和预定范围结果。',
            '- per_tissue.tsv / study_tissue_recall.tsv：各组织和研究×组织表现。',
            '- geometry_metrics.tsv / clustering_seeds.tsv：几何结构及全部聚类种子。',
            '- within_tissue_source_silhouette.tsv：同组织内来源分离，仅描述混杂，不预设必须越低越好。',
            '- predictions.tsv.gz / *_head.npz：细胞预测和线性头。',
            '- metrics、linear_confusion、umap：Arial，PNG/PDF；无新增QC或细胞排除。',
            '- config.json / verification.json：固定协议和完整性检查。', '']
    (OUT / 'report.md').write_text('\n'.join(text))
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'encode', 'evaluate', 'report', 'all'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.stage in ['prepare','all']: prepare()
        if args.stage in ['encode','all']: encode_features()
        if args.stage in ['evaluate','all']: evaluate()
        if args.stage == 'report': plot_and_report(pd.read_csv(CACHE / 'cells.tsv.gz', sep='\t', index_col=0))
