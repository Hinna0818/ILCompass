"""Extend the four locked IL33 benchmarks with three genuine Scratch v3 head seeds.

Only frozen feature caches are aligned; the native task trainers remain unchanged.
Existing seed0 weights/results are read-only, and no reconstruction task is run.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_il33_mixed_delta as task
import train_il33_bidirectional_tissue as tissue
import train_il33_lung_full as cd
from train_il33_expression_pca_controls import link, read_json, score_prediction, EXPERIMENTS

ROOT = task.ROOT
CONFIG = ROOT / 'configs/il33_scratch_v3_3seeds_v1.json'
MODEL = 'Scratch_v3'
NAMES = ('mixed', 'crossstudy', 'lung_liver')


def location(cfg, kind, name=''):
    return ROOT / cfg[kind+'_directory'] / name


def configuration(cfg, name):
    source = ROOT / cfg['mixed_source' if name == 'mixed' else name+'_source']
    original = read_json(source / 'prepare_manifest.json')
    return dict(original['config'], version=cfg['version']+'_'+name, seeds=cfg['seeds'],
        models={MODEL: cfg['checkpoint']}, **{k+'_directory': str(location(cfg, k, name).relative_to(ROOT))
        for k in ['data', 'result', 'model']})


def head(cfg, name, seed):
    if name == 'C_D':
        return location(cfg, 'result', f'C_D/{MODEL}/seed{seed}/head.pt')
    return location(cfg, 'model', f'{name}/{MODEL}/seed{seed}') / ('final.pt' if name == 'lung_liver' else 'best.pt')


def adaptation(cfg, seed, fraction, step=50):
    return location(cfg, 'model', f'adaptation/seed{seed}/{round(100*fraction):02d}pct_step{step}.pt')


def audit(cfg):
    if cfg['seeds'] != [0, 1, 2] or cfg['adaptation_fractions'] != [.05, .1]:
        raise ValueError('This extension uses fixed seeds0/1/2 and 5/10% adaptation')
    out = location(cfg, 'result'); out.mkdir(parents=True, exist_ok=True)
    comparison = ROOT / cfg['comparison_results']
    if not read_json(comparison / 'evaluation_complete.json')['original_results_unchanged']:
        raise ValueError('Complete the previous fixed benchmark first')
    cross = read_json(ROOT / cfg['crossstudy_source'] / 'prepare_manifest.json')
    if cross['parent_manifest_sha256'] != task.file_hash(ROOT / cfg['mixed_source'] / 'prepare_manifest.json'):
        raise ValueError('Mixed and cross-study caches do not share the original source corpus')
    digest = task.file_hash(ROOT / cfg['checkpoint'])
    protected = {str((comparison / 'metrics_by_seed.csv').relative_to(ROOT)): task.file_hash(comparison / 'metrics_by_seed.csv')}
    for name in NAMES:
        source = ROOT / cfg['mixed_source' if name == 'mixed' else name+'_source']
        manifest = source / 'prepare_manifest.json'
        old = read_json(manifest)
        task.prepared(old['config'])
        protected[str(manifest.relative_to(ROOT))] = task.file_hash(manifest)
        if name != 'mixed' and old['checkpoint_sha256'][MODEL] != digest:
            raise ValueError('Different frozen v3 checkpoint')
    native = ROOT / cfg['cd_source'] / MODEL
    if read_json(native / 'signature.json')['checkpoint']['sha256'] != digest:
        raise ValueError('C/D cache uses a different frozen encoder')
    paths = [ROOT / cfg['checkpoint'], ROOT / cfg['adaptation_split'], native / 'signature.json',
        ROOT / cfg['cd_results'] / f'{MODEL}/seed0/head.pt']
    for name in ['crossstudy', 'lung_liver']:
        old = read_json(ROOT / cfg[name+'_source'] / 'config.json')
        paths.append(ROOT / old['model_directory'] / MODEL / 'seed0' / ('best.pt' if name == 'crossstudy' else 'final.pt'))
    for f in cfg['adaptation_fractions']:
        paths.append(ROOT / cfg['adaptation_results'] / f'checkpoints/{round(100*f):02d}pct_step50.pt')
    few = read_json(ROOT / cfg['adaptation_data'] / 'manifest.json')
    for f in cfg['adaptation_fractions']:
        label = f'{round(100*f):02d}pct'
        path = ROOT / cfg['adaptation_data'] / f'{label}.npz'
        if task.file_hash(path) != few['adaptation_sha256'][label]:
            raise ValueError('Historical adaptation training subset changed')
        paths.append(path)
    protected.update({str(p.relative_to(ROOT)): task.file_hash(p) for p in paths})
    signature = dict(config=cfg, protected_sha256=protected)
    if (out / 'audit.json').exists() and read_json(out / 'audit.json') != signature:
        raise ValueError('Historical inputs changed')
    task.atomic_json(out / 'audit.json', signature)
    task.atomic_json(out / 'config.json', cfg)
    plan = [dict(task=name, seed=seed, action='reuse' if name != 'mixed' and seed == 0 else 'train')
            for name in ['mixed', 'C_D', 'crossstudy', 'lung_liver'] for seed in cfg['seeds']]
    plan += [dict(task=f'adaptation_{round(100*f)}pct', seed=seed, action='reuse' if seed == 0 else 'train')
             for seed in cfg['seeds'] for f in cfg['adaptation_fractions']]
    pd.DataFrame(plan).to_csv(out / 'run_plan.csv', index=False)
    print('AUDIT', pd.DataFrame(plan).groupby(['task', 'action']).size().to_dict(), flush=True)


def aligned_mixed_features(cfg):
    """A frozen feature can cross downstream split roles; no statistic is fitted here."""
    source = ROOT / cfg['crossstudy_source']
    manifest = read_json(source / 'prepare_manifest.json')
    metadata, features = [], []
    for role in ['train', 'validation']:
        m, _, h = task.load_split(source, manifest, role, MODEL)
        if task.file_hash(source / role / f'{MODEL}_h.npy') != manifest['aligned_feature_sha256'][role]:
            raise ValueError('Frozen source feature cache changed')
        metadata.append(m[m.condition.eq('PBS')]); features.append(np.asarray(h))
    m, h = pd.concat(metadata), np.concatenate(features)
    if not m.index.is_unique:
        raise ValueError('Duplicate cached control cells')
    target = pd.read_csv(ROOT / cfg['mixed_source'] / 'cell_manifest.tsv.gz', sep='\t', index_col=0, low_memory=False)
    target = target[target.condition.eq('PBS')]
    if set(target.index) != set(m.index):
        raise ValueError('Mixed benchmark cells differ from cached source controls')
    fields = ['study', 'response_group', 'library_id', 'condition']
    if not target[fields].astype(str).equals(m.loc[target.index, fields].astype(str)):
        raise ValueError('Cached cell provenance differs')
    return pd.Series(np.arange(len(m)), index=m.index), h


def reuse_head(cfg, name):
    if name == 'mixed':
        return
    source = ROOT / cfg[name+'_source']
    old = read_json(source / 'config.json')
    filename = 'final.pt' if name == 'lung_liver' else 'best.pt'
    origin = ROOT / old['model_directory'] / MODEL / 'seed0' / filename
    saved = read_json(origin.parent / 'complete.json')
    key = 'final_sha256' if name == 'lung_liver' else 'best_sha256'
    if saved['fingerprint'] != task.file_hash(source / 'prepare_manifest.json') or saved[key] != task.file_hash(origin):
        raise ValueError('Historical seed0 checkpoint changed')
    data = location(cfg, 'data', name)
    rc = configuration(cfg, name)
    model = task.restore_head(torch.load(origin, map_location='cpu', weights_only=False), rc,
                              len(pd.read_csv(data / 'gene_panel.tsv', sep='\t')), torch.device('cpu'))
    m, _, h = task.load_split(data, read_json(data / 'prepare_manifest.json'), 'train', MODEL)
    torch.testing.assert_close(model.h_mean, torch.tensor(np.array(h)).mean(0), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(model.h_std, torch.tensor(np.array(h)).std(0, unbiased=False).clamp_min(1e-4), rtol=1e-5, atol=2e-6)
    destination = head(cfg, name, 0)
    link(origin, destination)
    task.atomic_json(destination.parent / 'complete.json', dict(saved,
        fingerprint=task.file_hash(data / 'prepare_manifest.json'), reused_from=str(origin.relative_to(ROOT)),
        original_fingerprint=saved['fingerprint']))


def prepare(cfg):
    audit(cfg)
    mapping, mixed_h = aligned_mixed_features(cfg)
    for name in NAMES:
        source = ROOT / cfg['mixed_source' if name == 'mixed' else name+'_source']
        original = read_json(source / 'prepare_manifest.json'); rc = configuration(cfg, name)
        data = location(cfg, 'data', name); data.mkdir(parents=True, exist_ok=True)
        if (data / 'prepare_manifest.json').exists():
            task.prepared(rc); reuse_head(cfg, name); continue
        for f in ['gene_panel.tsv', 'functional_gene_sets.tsv', 'cell_manifest.tsv.gz']:
            link(source / f, data / f)
        hashes = {}
        for role in original['cell_map_sha256']:
            for f in ['metadata.tsv.gz', 'y.npy']+(['statistics.npz'] if role == 'train' else []):
                link(source / role / f, data / role / f)
            feature = data / role / f'{MODEL}_h.npy'
            if name == 'mixed':
                m = pd.read_csv(source / role / 'metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
                selected = m.index[m.condition.eq('PBS')]
                np.save(feature, mixed_h[mapping.loc[selected].to_numpy()])
            else:
                feature_source = source / role / f'{MODEL}_h.npy'
                if task.file_hash(feature_source) != original['aligned_feature_sha256'][role]:
                    raise ValueError('Frozen feature cache changed')
                link(feature_source, feature)
            hashes[role] = task.file_hash(feature)
        manifest = dict(original, config=rc, encoder_dimensions={MODEL: 384},
            checkpoint_sha256={MODEL: task.file_hash(ROOT / cfg['checkpoint'])}, aligned_feature_sha256=hashes,
            frozen_features_reused=True, encoder_inference=False)
        task.atomic_json(data / 'config.json', rc)
        task.atomic_json(data / 'prepare_manifest.json', manifest)
        reuse_head(cfg, name)
        print('PREPARED', name, flush=True)
    source = ROOT / cfg['cd_source']; data = location(cfg, 'data', 'C_D')
    for f in ['genes.tsv', 'train_y.npy', 'validation_y.npy', 'train_metadata.tsv.gz', 'validation_metadata.tsv.gz', 'PBS_metadata.tsv.gz']:
        link(source / f, data / f)
    link(source / MODEL / 'PBS_h.npy', data / 'PBS_h.npy')
    task.atomic_json(data / 'signature.json', read_json(source / MODEL / 'signature.json'))
    link(ROOT / cfg['cd_results'] / f'{MODEL}/seed0', head(cfg, 'C_D', 0).parent)
    fixed = pd.read_csv(ROOT / cfg['adaptation_split'], sep='\t', index_col=0)
    old = pd.read_csv(ROOT / cfg['adaptation_data'] / 'split_manifest.tsv.gz', sep='\t', index_col=0)
    np.testing.assert_array_equal(fixed.index, old.index)
    for key in ['role', '05pct', '10pct']:
        np.testing.assert_array_equal(fixed[key], old[key])
    if fixed.loc[fixed.role.eq('test'), ['05pct', '10pct']].any().any():
        raise ValueError('Adaptation/Holdout overlap')
    print('PREPARE COMPLETE; no encoder extraction or normalization refit', flush=True)


def lock_head(cfg, name, seed):
    path = head(cfg, name, seed)
    state = torch.load(path, map_location='cpu', weights_only=False)
    saved = read_json(path.parent / 'complete.json')
    if name == 'C_D':
        base = read_json(ROOT / cfg['cd_config'])
        if state['epoch'] != base['epochs'] or saved['D_used_for_gradients']:
            raise ValueError('C/D must use the fixed final epoch')
        for k in ['epochs', 'batch_size', 'hidden_size', 'learning_rate', 'weight_decay',
                  'variance_loss_weight', 'residual_penalty', 'minimum_expression_scale', 'gradient_clip']:
            if state['config'][k] != base[k]:
                raise ValueError('C/D training recipe differs')
        if state['signature'] != read_json(location(cfg, 'data', 'C_D/signature.json')):
            raise ValueError('C/D encoder cache differs')
    else:
        key = 'final_sha256' if name == 'lung_liver' else 'best_sha256'
        if saved['fingerprint'] != task.file_hash(location(cfg, 'data', name) / 'prepare_manifest.json') or saved[key] != task.file_hash(path):
            raise ValueError('Unfinished or changed head')
    return dict(path=str(path.relative_to(ROOT)), sha256=task.file_hash(path))


def check(cfg):
    for name in NAMES:
        rc = configuration(cfg, name); data, _, _, manifest, _ = task.prepared(rc)
        seen = set()
        for role in manifest['cell_map_sha256']:
            m, _, h = task.load_split(data, manifest, role, MODEL)
            if (seen.intersection(m.index) or h.shape[1] != 384 or not np.isfinite(h).all()
                    or task.file_hash(data / role / f'{MODEL}_h.npy') != manifest['aligned_feature_sha256'][role]):
                raise ValueError('Overlapping roles or invalid frozen features')
            seen.update(m.index)
        if name != 'mixed':
            lock_head(cfg, name, 0)
    lock_head(cfg, 'C_D', 0)
    task.atomic_json(location(cfg, 'result', 'input_acceptance.json'), dict(passed=True,
        feature_dimensions=384, all_original_splits_and_panels=True, encoder_inference=False,
        seeds=[0, 1, 2], fixed_adaptation_Holdout=True))
    print('INPUT CHECK PASSED', flush=True)


def train(cfg, device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the unchanged native C/D trainer')
    data = location(cfg, 'data', 'C_D'); result = location(cfg, 'result', f'C_D/{MODEL}')
    result.mkdir(parents=True, exist_ok=True)
    cd.CFG = dict(read_json(ROOT / cfg['cd_config']), seeds=cfg['seeds'], checkpoint=cfg['checkpoint'],
        cache_directory=str(data.relative_to(ROOT)), result_directory=str(result.relative_to(ROOT)))
    cd.CACHE, cd.OUT = data, result
    cd.fit()
    for name in NAMES:
        rc = configuration(cfg, name)
        for seed in cfg['seeds']:
            trainer = tissue.train_fixed if name == 'lung_liver' else task.train_one
            trainer(rc, MODEL, seed, device, resume=True)


def adapt(cfg, device):
    for seed in cfg['seeds']:
        lock_head(cfg, 'lung_liver', seed)
    ac = read_json(ROOT / cfg['adaptation_config']); rc = configuration(cfg, 'lung_liver')
    genes = len(pd.read_csv(location(cfg, 'data', 'lung_liver/gene_panel.tsv'), sep='\t'))
    history_record = read_json(ROOT / cfg['adaptation_results'] / 'training_complete.json')
    for seed in cfg['seeds']:
        initial = torch.load(head(cfg, 'lung_liver', seed), map_location='cpu', weights_only=False)
        for f in cfg['adaptation_fractions']:
            path = adaptation(cfg, seed, f); marker = path.with_suffix('.json'); path.parent.mkdir(parents=True, exist_ok=True)
            key = f'{round(100*f):02d}pct'; values_path = ROOT / cfg['adaptation_data'] / f'{key}.npz'
            signature = dict(initial=lock_head(cfg, 'lung_liver', seed), seed=seed, fraction=f,
                config=ac, split=task.file_hash(ROOT / cfg['adaptation_split']), data=task.file_hash(values_path))
            if marker.exists():
                saved = read_json(marker)
                if saved['signature'] != signature or saved['sha256'] != task.file_hash(path):
                    raise ValueError('Adaptation protocol changed')
                continue
            if seed == 0:
                for step in ac['save_steps']:
                    origin = ROOT / cfg['adaptation_results'] / f'checkpoints/{key}_step{step}.pt'
                    if history_record['checkpoints']['checkpoints/'+origin.name] != task.file_hash(origin):
                        raise ValueError('Old adaptation checkpoint changed')
                    link(origin, adaptation(cfg, seed, f, step))
            else:
                torch.manual_seed(seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)
                rng = np.random.default_rng(seed)
                model = task.restore_head(initial, rc, genes, device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=ac['learning_rate'], weight_decay=ac['weight_decay'])
                with np.load(values_path) as values:
                    x, h = values['control'], values['h']
                    target = torch.tensor(values['delta'], dtype=torch.float32, device=device)
                history = []
                for step in range(1, ac['steps']+1):
                    rows = rng.choice(len(x), min(len(x), ac['batch_size']), replace=False)
                    xx, hh = torch.tensor(x[rows], device=device), torch.tensor(h[rows], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = task.delta_loss(xx, model(hh), target)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite adaptation loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ac['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); history.append(dict(step=step, train_loss=float(loss.detach())))
                    if step in ac['save_steps']:
                        task.atomic_torch(adaptation(cfg, seed, f, step), dict(model=model.state_dict(), seed=seed,
                            step=step, fraction=f, selection='fixed_step_no_test_selection'))
                for name, value in model.named_buffers():
                    torch.testing.assert_close(value.cpu(), initial['model'][name], rtol=0, atol=0)
                pd.DataFrame(history).to_csv(path.with_suffix('.csv'), index=False)
            task.atomic_json(marker, dict(signature=signature, sha256=task.file_hash(path), Test_used_for_training_or_selection=False))
            print('ADAPT', seed, key, 'reuse' if seed == 0 else 'trained', flush=True)


def evaluate(cfg, device):
    locks = {f'{name}/{seed}': lock_head(cfg, name, seed) for name in ['C_D', *NAMES] for seed in cfg['seeds']}
    for seed in cfg['seeds']:
        for f in cfg['adaptation_fractions']:
            path = adaptation(cfg, seed, f)
            if read_json(path.with_suffix('.json'))['sha256'] != task.file_hash(path):
                raise ValueError('Finish all adaptation checkpoints before Holdout scoring')
            locks[f'adaptation/{seed}/{f}'] = dict(path=str(path.relative_to(ROOT)), sha256=task.file_hash(path))
    out = location(cfg, 'result'); task.atomic_json(out / 'test_protocol.json', locks)
    rows = []
    for name in ['C_D', *NAMES]:
        data = location(cfg, 'data', name)
        panel = pd.read_csv(data / ('genes.tsv' if name == 'C_D' else 'gene_panel.tsv'), sep='\t')
        if name == 'C_D':
            m = pd.read_csv(data / 'validation_metadata.tsv.gz', sep='\t', index_col=0)
            y = np.load(data / 'validation_y.npy', mmap_mode='r')
            x, truth = np.asarray(y[m.condition.eq('PBS')]), np.asarray(y[m.condition.eq('IL33')])
            for seed in cfg['seeds']:
                pred = np.load(head(cfg, name, seed).parent / 'D_predicted_log1p.npy', mmap_mode='r')
                rows.extend(score_prediction(cfg, pred, truth, x, panel,
                    dict(experiment=EXPERIMENTS[name], model=MODEL, seed=seed, background='all', adaptation='none'),
                    out / name / f'seed{seed}/test'))
            continue
        rc = configuration(cfg, name); manifest = read_json(data / 'prepare_manifest.json')
        m, y, h = task.load_split(data, manifest, 'test', MODEL)
        pm = m[m.condition.eq('PBS')]
        if name == 'lung_liver':
            split = pd.read_csv(ROOT / cfg['adaptation_split'], sep='\t', index_col=0)
            np.testing.assert_array_equal(m.index, split.index)
            holdout = split.role.eq('test').to_numpy(); pbs = m.condition.eq('PBS').to_numpy()
            x, truth, hh = np.asarray(y[holdout & pbs]), np.asarray(y[holdout & ~pbs]), np.asarray(h[holdout[pbs]])
            for seed in cfg['seeds']:
                for f in [0., *cfg['adaptation_fractions']]:
                    path = head(cfg, name, seed) if f == 0 else adaptation(cfg, seed, f)
                    model = task.restore_head(torch.load(path, map_location='cpu', weights_only=False), rc, len(panel), device)
                    pred = x + task.predict_delta(model, hh, x, rc['prediction_batch_size'], device)
                    label = 'Zero-shot' if f == 0 else f'{round(100*f)}%'
                    rows.extend(score_prediction(cfg, pred, truth, x, panel,
                        dict(experiment=EXPERIMENTS['adaptation'], model=MODEL, seed=seed, background='all', adaptation=label),
                        out / 'adaptation' / f'seed{seed}' / label))
            continue
        for seed in cfg['seeds']:
            model = task.restore_head(torch.load(head(cfg, name, seed), map_location='cpu', weights_only=False), rc, len(panel), device)
            for group in rc['backgrounds'] if name == 'mixed' else rc['external_backgrounds']:
                selected = pm.response_group.eq(group).to_numpy()
                x = np.asarray(y[m.condition.eq('PBS') & m.response_group.eq(group)])
                truth = np.asarray(y[m.condition.eq('IL33') & m.response_group.eq(group)])
                pred = x + task.predict_delta(model, h[selected], x, rc['prediction_batch_size'], device)
                rows.extend(score_prediction(cfg, pred, truth, x, panel,
                    dict(experiment=EXPERIMENTS[name], model=MODEL, seed=seed, background=group, adaptation='none'),
                    out / name / f'seed{seed}/test' / group))
    raw = pd.DataFrame(rows); raw.to_csv(out / 'scratch_metrics_by_background_seed.csv', index=False)
    keys = ['experiment', 'model', 'seed', 'adaptation', 'panel']
    values = ['delta_MSE', 'delta_R2', 'delta_Pearson', 'mean_R2', 'variance_R2', 'direction_accuracy', 'overlap', 'functional_MSE']
    macro = raw.groupby(keys)[values].agg(lambda s: np.mean(s.to_numpy())).reset_index()
    macro['delta_RMSE'] = np.sqrt(macro.delta_MSE)
    old = pd.read_csv(ROOT / cfg['comparison_results'] / 'metrics_by_seed.csv')
    combined = pd.concat([old, macro], ignore_index=True)
    if combined.duplicated(keys).any():
        raise ValueError('Duplicate model/seed/panel results')
    for _, group in combined[combined.seed.ge(0)].groupby(['experiment', 'adaptation', 'model', 'panel']):
        if sorted(group.seed.tolist()) != [0, 1, 2]:
            raise ValueError('Exactly three genuine optimization seeds required')
    combined.to_csv(out / 'metrics_by_seed.csv', index=False)
    combined.groupby(['experiment', 'adaptation', 'model', 'panel'])[values+['delta_RMSE']].agg(['mean', 'std']).to_csv(out / 'metrics_summary.csv')
    for path, digest in read_json(out / 'audit.json')['protected_sha256'].items():
        if task.file_hash(ROOT / path) != digest:
            raise ValueError('Historical inputs or outputs were changed')
    task.atomic_json(out / 'evaluation_complete.json', dict(original_results_unchanged=True, seeds=cfg['seeds'],
        locked_checkpoints=locks, encoder_updates=0, reconstruction_task_run=False,
        ranking='Absolute observed mean log1p(CP10K) change; no p-value selection',
        adaptation='Same 80% target Holdout for zero-shot/5%/10%, fixed50steps',
        selection='Original task-specific selection; C/D D monitoring retained, never used for checkpoint selection'))
    print('EVALUATION COMPLETE', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'prepare', 'check', 'train', 'adapt', 'evaluate', 'plot', 'all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = read_json(args.config)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    out = location(cfg, 'result'); out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(cfg['cpu_threads']), (out / 'workflow.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'audit': audit(cfg)
        if args.stage in ['prepare', 'all']: prepare(cfg)
        if args.stage in ['check', 'all']: check(cfg)
        if args.stage in ['train', 'all']: train(cfg, device)
        if args.stage in ['adapt', 'all']: adapt(cfg, device)
        if args.stage in ['evaluate', 'all']: evaluate(cfg, device)
        if args.stage in ['plot', 'all']:
            subprocess.run([sys.executable, str(ROOT / 'main/plot_il33_expression_pca_controls.py'),
                            '--config', str(args.config)], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
