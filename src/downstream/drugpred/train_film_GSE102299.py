"""GSE102299: independent replicate IL25/NMU learning and unseen-combination tests."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_film_multistimulus as base

ROOT = base.ROOT


def split_metadata(meta, cfg):
    rep = str(cfg['within_replicate'])
    selected = meta.author_replicate.astype(str).eq(rep) & meta.condition.isin(['PBS', 'IL25', 'NMU', 'IL25_NMU'])
    m = meta[selected].copy()
    if not m.index.is_unique or set(m.condition) != {'PBS', 'IL25', 'NMU', 'IL25_NMU'}:
        raise ValueError('Expected unique cells and all four conditions in the selected replicate')
    m['split'] = 'test'
    rng = np.random.default_rng(cfg['split_seed'])
    for condition in ('PBS', 'IL25', 'NMU'):
        rows = m.index[m.condition.eq(condition)]
        nv = round(len(rows) * (cfg['control_validation_fraction'] if condition == 'PBS' else cfg['validation_fraction']))
        nt = round(len(rows) * cfg['control_test_fraction']) if condition == 'PBS' else 0
        if nv < 1 or len(rows) - nv - nt < 1 or (condition == 'PBS' and nt < 1):
            raise ValueError('Insufficient source cells for fixed roles')
        order = rng.permutation(rows)
        m.loc[order[:nv], 'split'] = 'validation'
        m.loc[order[nv + nt:], 'split'] = 'train'
    return m


def cached_features(a, cfg):
    """Reuse no-mask v3 PBS vectors only after raw-count and identity alignment."""
    folder = ROOT / cfg['feature_directory']
    manifest = json.loads((folder / 'prepare_manifest.json').read_text())
    digest = base.file_hash(ROOT / cfg['checkpoint'])
    if manifest['checkpoint_sha256']['Scratch_v3'] != digest:
        raise ValueError('v3 feature cache/checkpoint mismatch')
    state = torch.load(ROOT / cfg['checkpoint'], map_location='cpu', weights_only=False)
    if state['provenance'] != manifest['scratch_provenance']:
        raise ValueError('v3 cached pretraining provenance differs')
    frames, matrices = [], []
    for role in ('train', 'validation'):
        path = folder / role / 'Scratch_v3_h.npy'
        if base.file_hash(path) != manifest['aligned_feature_sha256'][role]:
            raise ValueError('Cached v3 feature file changed')
        meta = pd.read_csv(folder / role / 'metadata.tsv.gz', sep='\t', index_col=0,
                           usecols=['cell_id', 'study', 'condition'])
        meta = meta[meta.condition.eq('PBS')]
        h = np.load(path)
        if len(h) != len(meta):
            raise ValueError('Cached PBS feature count mismatch')
        keep = meta.study.eq('GSE102299')
        frames.append(meta.loc[keep]); matrices.append(h[keep])
    meta = pd.concat(frames); h = np.concatenate(matrices)
    desired = a.obs[a.obs.condition.eq('PBS')]
    loc = meta.index.get_indexer(desired.index)
    if not meta.index.is_unique or (loc < 0).any() or set(meta.index) != set(desired.index) or not np.isfinite(h).all():
        raise ValueError('Incomplete/duplicated or invalid GSE102299 PBS features')
    raw = ad.read_h5ad(folder / 'source_full_counts.h5ad')
    rows = raw.obs_names.get_indexer(desired.index)
    cols = raw.var_names.get_indexer(a.var_names)
    if (rows < 0).any() or (cols < 0).any() or (raw.X[rows][:, cols] != a[desired.index].X).nnz:
        raise ValueError('Cached PBS raw matrix differs from canonical GSE102299 counts')
    print('Verified cached v3 PBS:', len(desired), 'cells,', h.shape[1], 'dimensions', flush=True)
    return pd.DataFrame(h[loc], index=desired.index), dict(checkpoint_sha256=digest,
        checkpoint_epoch=state['state']['epoch'], original_manifest=str((folder / 'prepare_manifest.json').relative_to(ROOT)),
        original_manifest_sha256=base.file_hash(folder / 'prepare_manifest.json'),
        raw_counts_identity='Every PBS count and cell matched against original cached source matrix',
        encoding=manifest['encoding'], encoder_updates=0)


def prepare(cfg):
    data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if (data / 'complete.json').exists():
        base.ready(cfg); print('PREPARE reused Replicate', cfg['within_replicate'], flush=True); return
    a = ad.read_h5ad(ROOT / cfg['source'])
    counts = base._raw_counts(a.layers['counts'])
    if (not a.obs.study.eq('GSE102299').all() or not a.obs.qc_version.eq('ilc2_balanced_v1').all()
            or not a.obs.selected_ILC2.all() or not a.obs.tissue.eq('Lung').all()
            or not a.var_names.is_unique or not a.var_names.str.match(r'^ENSMUSG\d+$').all() or (counts != a.X).nnz):
        raise ValueError('Invalid canonical ILC2 raw counts/metadata')
    corpus = pd.read_csv(ROOT / cfg['pretraining_manifest'], sep='\t', usecols=['cell_id', 'study'])
    if corpus.study.eq('GSE102299').any() or set(corpus.cell_id) & set(a.obs_names):
        raise ValueError('GSE102299 overlaps v3 pretraining')
    h, signature = cached_features(a, cfg)
    fields = ['study', 'gsm', 'library_id', 'barcode', 'condition', 'author_replicate', 'author_batch',
              'animal_id', 'tissue', 'qc_version']
    meta = split_metadata(a.obs[fields], cfg)
    meta['source_row'] = a.obs_names.get_indexer(meta.index)
    meta.to_csv(data / 'split_manifest.tsv.gz', sep='\t', index_label='cell_id')
    genes = a.var[['gene_symbol']].copy(); genes.index.name = 'gene_id'
    genes['functional'] = genes.index.isin(pd.read_csv(ROOT / cfg['functional_genes']).gene_id)
    genes.to_csv(data / 'gene_panel.tsv', sep='\t')
    counts = counts[meta.source_row.to_numpy()]
    depth = np.asarray(counts.sum(1)).ravel()
    if (depth <= 0).any():
        raise ValueError('Zero-depth cells')
    expression = counts.astype(np.float32).multiply((1e4 / depth)[:, None]).tocsr()
    expression.data = np.log1p(expression.data)
    for role in ('train', 'validation', 'test'):
        folder = data / role; folder.mkdir(exist_ok=True)
        rows = np.flatnonzero(meta.split.eq(role)); m = meta.iloc[rows]
        m.to_csv(folder / 'metadata.tsv.gz', sep='\t', index_label='cell_id')
        y = np.lib.format.open_memmap(folder / 'y.npy', mode='w+', dtype=np.float32, shape=(len(rows), len(genes)))
        for left in range(0, len(rows), 256):
            y[left:left + 256] = expression[rows[left:left + 256]].toarray()
        y.flush(); del y
        np.save(folder / 'h.npy', h.loc[m.index[m.condition.eq('PBS')]].to_numpy())
    del a, counts, expression, h
    train = base.load_role(cfg, 'train')
    y = np.load(data / 'train/y.npy', mmap_mode='r')
    mean, std = base.array_moments(y)
    cm, cs = base.array_moments(train['pbs'])
    target = [base.array_moments(train['targets'][c]) for c in base.training_conditions(cfg)]
    stats = dict(mean=mean.astype(np.float32), scale=np.maximum(std, cfg['minimum_expression_scale']).astype(np.float32),
        h_mean=train['h'].mean(0), h_std=train['h'].std(0), control_mean=cm, control_std=cs,
        target_mean=np.stack([v[0] for v in target]), target_std=np.stack([v[1] for v in target]))
    np.savez_compressed(data / 'statistics.npz', **stats)
    print('Fitting Train-only PCA/KL, Replicate', cfg['within_replicate'], flush=True)
    reference = base.fit_reference(y, stats['mean'], stats['scale'], np.flatnonzero(genes.functional), cfg)
    np.savez_compressed(data / 'distribution_reference.npz', **reference)
    for role in ('validation', 'test'):
        m = pd.read_csv(data / role / 'metadata.tsv.gz', sep='\t', index_col=0)
        controls = m.index[m.condition.eq('PBS')]
        conditions = base.training_conditions(cfg) if role == 'validation' else base.evaluation_conditions(cfg)
        for c in conditions:
            targets = m.index[m.condition.eq(c)]
            n = min(cfg['distribution_cells'], len(controls), len(targets))
            pairs = []
            for seed in cfg['distribution_seeds']:
                rng = np.random.default_rng(seed)
                pairs.append((rng.choice(len(controls), n, replace=False), rng.choice(len(targets), n, replace=False)))
            np.savez_compressed(data / role / f'{c}_draws.npz', predicted_indices=np.stack([p for p, t in pairs]),
                target_indices=np.stack([t for p, t in pairs]), control_ids=controls.to_numpy(dtype=str), target_ids=targets.to_numpy(dtype=str))
    summary = meta.groupby(['author_replicate', 'condition', 'split'], observed=True).size().rename('cells').reset_index()
    summary.to_csv(out / 'split_counts.csv', index=False)
    base.atomic_json(data / 'complete.json', dict(config=cfg, source_sha256=base.file_hash(ROOT / cfg['source']),
        genes=len(genes), functional_genes=int(genes.functional.sum()), cells=len(meta), frozen_features=signature,
        statistics_fit='Selected replicate Train PBS/IL25/NMU only; no joint condition',
        normalization='log1p(CP10K) on all 27998 measured genes', source='Lung; replicate label only, batch/animal unknown',
        split='PBS 70/15/15; IL25 and NMU 80/20; IL25_NMU all Test; IL33 excluded'))
    base.atomic_json(out / 'config.json', cfg)
    print('PREPARED\n' + summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'train', 'evaluate', 'all'])
    parser.add_argument('--replicate', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    for rep in (('1', '2') if args.replicate == 'both' else (args.replicate,)):
        cfg = json.loads((ROOT / f'configs/film_combination_GSE102299_rep{rep}_v1.json').read_text())
        torch.set_num_threads(cfg['cpu_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        print('EXPERIMENT GSE102299 Replicate', rep, 'PBS/IL25/NMU Train; combination Test', flush=True)
        with threadpool_limits(cfg['cpu_threads']):
            if args.command in ('prepare', 'all'):
                prepare(cfg)
            if args.command == 'prepare':
                continue
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            if args.command == 'check':
                base.check(cfg, device)
            if args.command in ('train', 'all'):
                for seed in cfg['seeds']:
                    base.train_seed(cfg, seed, device)
            if args.command in ('evaluate', 'all'):
                base.evaluate_models(cfg, device)


if __name__ == '__main__':
    main()
