"""GSE136154 FiLM: supervised C->D or independent same-batch combination holdout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.extend([str(ROOT), str(ROOT / 'src')])
from scratch.drug_pred_FM.model import FiLMPerturbationModel
from qc_utils import _raw_counts
from posttrain_data import file_hash
from distribution_metrics import fit_reference, population_scores, distribution_scores

CONFIG = ROOT / 'configs/film_multistimulus_GSE136154_v1.json'
CONDITIONS = {'IL33': [1, 0], 'CGRP': [0, 1], 'IL33_CGRP': [1, 1]}
ALL_CONDITIONS = ('PBS', *CONDITIONS)


def training_conditions(cfg):
    if cfg.get('within_replicate'):
        return tuple(cfg['model']['stimuli'])
    return ('IL33', 'CGRP') if cfg.get('within_batch') else tuple(CONDITIONS)


def evaluation_conditions(cfg):
    if cfg.get('within_replicate'):
        return (cfg['combination'],)
    return ('IL33_CGRP',) if cfg.get('within_batch') else tuple(CONDITIONS)


def condition_vectors(cfg):
    if cfg.get('within_replicate'):
        a, b = training_conditions(cfg)
        return {a: [1, 0], b: [0, 1], cfg['combination']: [1, 1]}
    return CONDITIONS


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)


def split_metadata(meta, cfg):
    """One role per cell, with the same PBS split shared by all target stimuli."""
    if cfg.get('within_batch'):
        batch = cfg['within_batch']
        if batch not in ('C', 'D'):
            raise ValueError('within_batch must be C or D')
        meta = meta[meta.author_batch.eq(batch)].copy()
        if set(meta.condition) != set(ALL_CONDITIONS) or not meta.index.is_unique:
            raise ValueError('Expected unique cells for all four conditions in the selected batch')
        meta['split'] = 'test'
        rng = np.random.default_rng(cfg['split_seed'])
        for condition in ('PBS', *training_conditions(cfg)):
            rows = meta.index[meta.condition.eq(condition)]
            fraction = cfg['control_validation_fraction'] if condition == 'PBS' else cfg['validation_fraction']
            nv = round(len(rows) * fraction)
            nt = round(len(rows) * cfg['control_test_fraction']) if condition == 'PBS' else 0
            if nv < 1 or len(rows) - nv - nt < 1 or (condition == 'PBS' and nt < 1):
                raise ValueError(f'Insufficient cells or invalid split fractions for {condition}')
            order = rng.permutation(rows)
            meta.loc[order[:nv], 'split'] = 'validation'
            meta.loc[order[nv + nt:], 'split'] = 'train'
        return meta
    meta = meta.copy(); meta['split'] = 'test'
    rng = np.random.default_rng(cfg['split_seed'])
    for condition in ALL_CONDITIONS:
        rows = meta.index[meta.author_batch.eq('C') & meta.condition.eq(condition)]
        nv = round(len(rows) * cfg['validation_fraction'])
        if not 0 < nv < len(rows):
            raise ValueError(f'Insufficient C cells for {condition}')
        order = rng.permutation(rows)
        meta.loc[order[nv:], 'split'] = 'train'
        meta.loc[order[:nv], 'split'] = 'validation'
    expected = {(batch, c) for batch in ['C', 'D'] for c in ALL_CONDITIONS}
    if set(zip(meta.author_batch, meta.condition)) != expected or not meta.index.is_unique:
        raise ValueError('Expected unique GSE136154 cells for all four conditions in C and D')
    return meta


def array_moments(array):
    total = np.zeros(array.shape[1], dtype=np.float64); squares = total.copy()
    for left in range(0, len(array), 128):
        x = np.array(array[left:left + 128], dtype=np.float64)
        total += x.sum(0); squares += (x * x).sum(0)
    mean = total / len(array)
    return mean, np.sqrt(np.maximum(squares / len(array) - mean * mean, 0))


def ready(cfg):
    data = ROOT / cfg['data_directory']
    manifest = json.loads((data / 'complete.json').read_text())
    if manifest['config'] != cfg:
        raise ValueError('Prepared configuration differs; use a new experiment directory')
    return data, ROOT / cfg['result_directory'], manifest


def prepare(cfg):
    data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if (data / 'complete.json').exists():
        ready(cfg); print('PREPARE existing data reused', flush=True); return
    cache = ROOT / cfg['feature_directory']
    signature = json.loads((cache / 'Scratch_v3/signature.json').read_text())
    for path, digest in [(cache / 'Scratch_v3/PBS_h.npy', signature['features_sha256']),
        (cache / 'PBS_metadata.tsv.gz', signature['common_assets_sha256']['PBS_metadata.tsv.gz']),
        (ROOT / signature['checkpoint']['path'], signature['checkpoint']['sha256'])]:
        if file_hash(path) != digest:
            raise ValueError(f'Frozen v3 feature provenance changed: {path}')
    a = ad.read_h5ad(ROOT / cfg['source'])
    if (not a.obs.study.eq('GSE136154').all() or not a.obs.qc_version.eq('ilc2_balanced_v1').all()
            or not a.obs.selected_ILC2.all() or not a.var_names.is_unique
            or not a.var_names.str.match(r'^ENSMUSG\d+$').all()):
        raise ValueError('Expected stable mouse genes and balanced_v1 selected ILC2')
    counts = _raw_counts(a.layers['counts'])
    if (counts != a.X).nnz:
        raise ValueError('Source X and integer counts differ')
    columns = ['study', 'author_batch', 'condition', 'gsm', 'library_id', 'qc_version']
    meta = split_metadata(a.obs[columns], cfg)
    source_rows = a.obs_names.get_indexer(meta.index)
    pretraining = pd.read_csv(ROOT / cfg['pretraining_manifest'], sep='\t', usecols=['cell_id', 'study'])
    if pretraining.study.eq('GSE136154').any() or set(pretraining.cell_id).intersection(meta.index):
        raise ValueError('Downstream cells/study overlap v3 pretraining')
    old_genes = pd.read_csv(cache / 'genes.tsv', sep='\t')
    np.testing.assert_array_equal(a.var_names, old_genes.gene_id)
    genes = a.var[['gene_symbol']].copy(); genes.index.name = 'gene_id'
    genes['functional'] = genes.index.isin(pd.read_csv(ROOT / cfg['functional_genes']).gene_id)
    if not genes.functional.any():
        raise ValueError('No functional genes mapped to measured panel')
    genes.to_csv(data / 'gene_panel.tsv', sep='\t')
    meta.to_csv(data / 'split_manifest.tsv.gz', sep='\t', index_label='cell_id')
    summary = meta.groupby(['split', 'author_batch', 'condition'], observed=True).size().rename('cells').reset_index()
    summary.to_csv(out / 'split_counts.csv', index=False)
    pm = pd.read_csv(cache / 'PBS_metadata.tsv.gz', sep='\t', index_col=0)
    h = np.load(cache / 'Scratch_v3/PBS_h.npy', mmap_mode='r')
    depth = np.asarray(counts.sum(1)).ravel()
    if (depth <= 0).any():
        raise ValueError('Zero-count cells')
    expression = counts.astype(np.float32).multiply((1e4 / depth)[:, None]).tocsr()
    expression.data = np.log1p(expression.data)
    for role in ['train', 'validation', 'test']:
        folder = data / role; folder.mkdir(exist_ok=True)
        rows = np.flatnonzero(meta.split.eq(role)); m = meta.iloc[rows]
        m.to_csv(folder / 'metadata.tsv.gz', sep='\t', index_label='cell_id')
        y = np.lib.format.open_memmap(folder / 'y.npy', mode='w+', dtype=np.float32,
                                     shape=(len(rows), len(genes)))
        for left in range(0, len(rows), 256):
            y[left:left + 256] = expression[source_rows[rows[left:left + 256]]].toarray()
        y.flush(); del y
        pbs = m[m.condition.eq('PBS')]; loc = pm.index.get_indexer(pbs.index)
        if (loc < 0).any() or not np.array_equal(pm.iloc[loc].author_batch, pbs.author_batch):
            raise ValueError('Control features do not match source cells/batches')
        # Verify the cached FM features refer to the same measured control expression.
        old_role = 'validation' if pbs.author_batch.eq('D').all() else 'train'
        old_meta = pd.read_csv(cache / f'{old_role}_metadata.tsv.gz', sep='\t', index_col=0)
        old_rows = old_meta.index.get_indexer(pbs.index)
        if (old_rows < 0).any():
            raise ValueError('Cached control expression identities differ')
        old_y = np.load(cache / f'{old_role}_y.npy', mmap_mode='r')
        new_y = np.load(folder / 'y.npy', mmap_mode='r')
        pbs_rows = np.flatnonzero(m.condition.eq('PBS'))
        for left in range(0, len(pbs_rows), 256):
            np.testing.assert_allclose(new_y[pbs_rows[left:left + 256]], old_y[old_rows[left:left + 256]], atol=1e-6, rtol=1e-6)
        np.save(folder / 'h.npy', np.array(h[loc]))
    del a, counts, expression
    tr = load_role(cfg, 'train')
    train_y = np.load(data / 'train/y.npy', mmap_mode='r')
    mean, std = array_moments(train_y)
    scale = np.maximum(std, cfg['minimum_expression_scale']).astype(np.float32)
    control_mean, control_std = array_moments(tr['pbs'])
    target_moments = [array_moments(tr['targets'][c]) for c in training_conditions(cfg)]
    stats = dict(mean=mean.astype(np.float32), scale=scale,
        h_mean=tr['h'].mean(0), h_std=tr['h'].std(0),
        control_mean=control_mean, control_std=control_std,
        target_mean=np.stack([v[0] for v in target_moments]),
        target_std=np.stack([v[1] for v in target_moments]))
    np.savez_compressed(data / 'statistics.npz', **stats)
    print('PREPARE fitting Train-only PCA and KL bins', flush=True)
    reference = fit_reference(train_y, stats['mean'], stats['scale'], np.flatnonzero(genes.functional), cfg)
    np.savez_compressed(data / 'distribution_reference.npz', **reference)
    # Evaluation subsamples are fixed from cell identities, never from response strength.
    for role in ['validation', 'test']:
        m = pd.read_csv(data / role / 'metadata.tsv.gz', sep='\t', index_col=0)
        control_ids = m.index[m.condition.eq('PBS')]
        for condition in (training_conditions(cfg) if role == 'validation' else evaluation_conditions(cfg)):
            target_ids = m.index[m.condition.eq(condition)]
            n = min(cfg['distribution_cells'], len(control_ids), len(target_ids))
            draws = []
            for seed in cfg['distribution_seeds']:
                rng = np.random.default_rng(seed)
                draws.append((rng.choice(len(control_ids), n, replace=False), rng.choice(len(target_ids), n, replace=False)))
            np.savez_compressed(data / role / f'{condition}_draws.npz',
                predicted_indices=np.stack([d[0] for d in draws]), target_indices=np.stack([d[1] for d in draws]),
                control_ids=control_ids.to_numpy(dtype=str), target_ids=target_ids.to_numpy(dtype=str))
    atomic_json(data / 'complete.json', dict(config=cfg, source_sha256=file_hash(ROOT / cfg['source']),
        frozen_features_signature=signature, cells=len(meta), genes=len(genes),
        statistics_fit='Selected batch Train PBS and singles only; h uses Train PBS only' if cfg.get('within_batch') else 'C Train only; h statistics use C Train PBS only',
        conditions=CONDITIONS, training_conditions=training_conditions(cfg),
        selection='Validation single-stimulus macro MSE-delta; combination excluded' if cfg.get('within_batch') else 'C Validation macro MSE-delta; D excluded',
        normalization='log1p(CP10K), full measured GEX denominator',
        source_boundary='One batch only; all combination cells are Test' if cfg.get('within_batch') else 'New supervised-combination experiment; historical combo holdout roles are not reused'))
    atomic_json(out / 'config.json', cfg)
    print('PREPARED\n' + summary.to_string(index=False), flush=True)


def load_role(cfg, role):
    folder = ROOT / cfg['data_directory'] / role
    meta = pd.read_csv(folder / 'metadata.tsv.gz', sep='\t', index_col=0)
    expected = cfg.get('within_replicate') or cfg.get('within_batch') or ('D' if role == 'test' else 'C')
    source_label = meta.author_replicate.astype(str) if cfg.get('within_replicate') else meta.author_batch
    if not source_label.eq(expected).all() or not meta.split.eq(role).all():
        raise ValueError(f'Unexpected data boundary for {role}')
    y = np.load(folder / 'y.npy', mmap_mode='r')
    conditions = evaluation_conditions(cfg) if role == 'test' else training_conditions(cfg)
    if set(meta.condition) != {'PBS', *conditions}:
        raise ValueError(f'Unexpected conditions for {role}; combination must remain held out')
    return dict(meta=meta, pbs=np.array(y[meta.condition.eq('PBS').to_numpy()]),
        targets={c: np.array(y[meta.condition.eq(c).to_numpy()]) for c in conditions},
        h=np.load(folder / 'h.npy'))


def protocol(cfg):
    data, _, _ = ready(cfg)
    files = [Path(__file__), ROOT / 'scratch/drug_pred_FM/model.py',
             Path(__file__).with_name('distribution_metrics.py'), Path(__file__).with_name('il33_lung_metrics.py')]
    if cfg.get('workflow_script'):
        files.append(ROOT / cfg['workflow_script'])
    return dict(config=cfg, prepared_sha256=file_hash(data / 'complete.json'),
        statistics_sha256=file_hash(data / 'statistics.npz'),
        reference_sha256=file_hash(data / 'distribution_reference.npz'),
        code_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in files})


def make_model(genes, stats, cfg, device):
    return FiLMPerturbationModel(genes.gene_id.tolist(), stats['h_mean'], stats['h_std'], **cfg['model']).to(device)


def as_tensors(group, device):
    return dict(h=torch.as_tensor(group['h'], device=device), pbs=torch.as_tensor(group['pbs'], device=device),
                targets={c: torch.as_tensor(y, device=device) for c, y in group['targets'].items()})


@torch.no_grad()
def predict(model, group, condition, cfg, device, additive=False):
    model.eval(); prediction = np.empty_like(group['pbs'])
    for left in range(0, len(prediction), cfg['prediction_batch_size']):
        x = torch.as_tensor(group['pbs'][left:left + cfg['prediction_batch_size']], device=device)
        h = torch.as_tensor(group['h'][left:left + len(x)], device=device)
        if additive:
            # Add signed shifts first; clipping each single response would bias the sum.
            delta = sum(model(h, x, x.new_tensor(condition_vectors(cfg)[c]).expand(len(x), -1))['raw_delta']
                        for c in training_conditions(cfg))
            value = (x + delta).clamp_min(0)
        else:
            c = x.new_tensor(condition_vectors(cfg)[condition]).expand(len(x), -1)
            value = model(h, x, c)['prediction']
        prediction[left:left + len(x)] = value.cpu().numpy()
    return prediction


def validation_scores(model, group, cfg, device):
    scores = {}
    for condition in training_conditions(cfg):
        prediction = predict(model, group, condition, cfg, device)
        error = prediction.mean(0, dtype=np.float64) - group['targets'][condition].mean(0, dtype=np.float64)
        scores[condition] = float(np.mean(error ** 2))
    return scores


def epoch_batches(targets, cfg, rng, conditions=None):
    conditions = training_conditions(cfg) if conditions is None else conditions
    batches = []
    for c in conditions:
        order = rng.permutation(len(targets[c]))
        for left in range(0, len(order), cfg['batch_size']):
            ids = order[left:left + cfg['batch_size']]
            batches.append((c, ids, len(ids) / len(order) / len(conditions)))
    return [batches[i] for i in rng.permutation(len(batches))]


def loss_terms(result, target, scale, cfg):
    pred = result['prediction']
    mean = ((pred.mean(0) - target.mean(0)) / scale).square().mean()
    variance = ((pred.var(0, unbiased=False) - target.var(0, unbiased=False)) / scale.square()).square().mean()
    penalty = (result['raw_delta'] / scale).square().mean()
    loss = mean + cfg['variance_loss_weight'] * variance + cfg['shift_loss_weight'] * penalty
    return loss, mean, variance, penalty


def train_seed(cfg, seed, device):
    data, out, _ = ready(cfg); signature = protocol(cfg)
    folder = out / f'seed{seed}'; folder.mkdir(exist_ok=True)
    with (folder / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'complete.json').exists():
            complete = json.loads((folder / 'complete.json').read_text())
            if complete['protocol'] != signature or complete['best_sha256'] != file_hash(folder / 'best.pt'):
                raise ValueError('Completed training has changed inputs/code/checkpoint')
            print('TRAIN complete reused', seed, flush=True); return
        tr, va = load_role(cfg, 'train'), load_role(cfg, 'validation')
        genes = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
        with np.load(data / 'statistics.npz') as f:
            stats = dict(f)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        model = make_model(genes, stats, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        rng = np.random.default_rng(seed)
        tensor = as_tensors(tr, device); scale = torch.tensor(stats['scale'], device=device)
        history, best, stale, start = [], float('inf'), 0, 1
        latest = folder / 'last.pt'
        if latest.exists():
            last = torch.load(latest, map_location='cpu', weights_only=False)
            if last['protocol'] != signature or last['device'] != str(device):
                raise ValueError('Resume protocol/device differs; use a new result directory')
            model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
            rng.bit_generator.state = last['numpy_rng']; torch.set_rng_state(last['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(last['cuda_rng'])
            history, best, stale, start = last['history'], last['best'], last['stale'], last['epoch'] + 1
            print('RESUME', seed, 'epoch', start, flush=True)
        clock = time.monotonic()
        for epoch in range(start, cfg['epochs'] + 1):
            if epoch > cfg['minimum_epochs'] and stale >= cfg['patience']:
                break
            model.train(); batches = epoch_batches(tr['targets'], cfg, rng); totals = np.zeros(4)
            for condition, target_ids, factor in batches:
                src = rng.choice(len(tr['pbs']), len(target_ids), replace=True)
                x, h, target = tensor['pbs'][src], tensor['h'][src], tensor['targets'][condition][target_ids]
                c = x.new_tensor(condition_vectors(cfg)[condition]).expand(len(x), -1)
                optimizer.zero_grad(set_to_none=True)
                loss, mean, variance, penalty = loss_terms(model(h, x, c), target, scale, cfg)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite FiLM objective')
                (loss * factor * len(batches)).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([v.detach().item() for v in [loss, mean, variance, penalty]]) * factor
            validation = validation_scores(model, va, cfg, device)
            score = float(np.mean(list(validation.values())))
            improved = score < best
            if improved:
                best, stale = score, 0
                atomic_torch(folder / 'best.pt', dict(model=model.state_dict(), epoch=epoch, seed=seed, protocol=signature))
            else:
                stale += 1
            row = dict(epoch=epoch, train_loss=totals[0], train_mean=totals[1], train_variance=totals[2],
                train_shift=totals[3], val_macro_MSE_delta=score, best=best, stale=stale,
                grad_norm=float(norm), **{f'val_{c}_MSE_delta': v for c, v in validation.items()})
            history.append(row)
            atomic_torch(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                seed=seed, device=str(device), protocol=signature, history=history, best=best, stale=stale,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
            print(f'FiLM/seed{seed} epoch={epoch} train={totals[0]:.6g} val_MSE={score:.6g} '
                  f'val_RMSE={np.sqrt(score):.6g} best={best:.6g} stale={stale} '
                  + ' '.join(f'{c}={np.sqrt(v):.5f}' for c, v in validation.items())
                  + f' elapsed={time.monotonic()-clock:.0f}s', flush=True)
        # If interrupted just after writing last.pt, the persisted history is authoritative.
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        selected = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        atomic_json(folder / 'complete.json', dict(protocol=signature, best_epoch=selected['epoch'],
            epochs_run=history[-1]['epoch'], best_validation_MSE=best, best_sha256=file_hash(folder / 'best.pt'),
            encoder_updates=0, D_used_for_training_or_selection=cfg.get('within_batch') == 'D',
            combination_used_for_training_or_selection=not bool(cfg.get('within_batch') or cfg.get('within_replicate')),
            parameters=sum(p.numel() for p in model.parameters())))


def evaluate_models(cfg, device):
    data, out, _ = ready(cfg); signature = protocol(cfg); locked = {}
    for seed in cfg['seeds']:
        folder = out / f'seed{seed}'
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['protocol'] != signature or complete['best_sha256'] != file_hash(folder / 'best.pt'):
            raise ValueError('Finish and lock every configured seed before reading Test expression')
        locked[str(seed)] = complete
    lock_path = out / 'test_protocol.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != locked:
        raise ValueError('Test already evaluated with a different selected checkpoint')
    atomic_json(lock_path, locked)
    test = load_role(cfg, 'test')
    genes = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
    with np.load(data / 'statistics.npz') as f:
        stats = dict(f)
    with np.load(data / 'distribution_reference.npz') as f:
        reference = dict(f)
    functional = reference['functional_indices']; all_rows, distribution_rows = [], []
    x = test['pbs']
    for ci, condition in enumerate(evaluation_conditions(cfg)):
        truth = test['targets'][condition]
        if cfg.get('within_batch') or cfg.get('within_replicate'):
            delta = (stats['target_mean'] - stats['control_mean']).sum(0)
            methods = ['control_copy', 'mean_delta_additive']
            methods += [f'{name}_seed{s}' for s in cfg['seeds'] for name in ('FiLM', 'FiLM_additive')]
        else:
            delta = stats['target_mean'][ci] - stats['control_mean']
            ratio = np.maximum(stats['target_std'][ci], cfg['minimum_expression_scale']) / np.maximum(stats['control_std'], cfg['minimum_expression_scale'])
            methods = ['control_copy', 'mean_delta', 'target_mean', 'variance_scale'] + [f'FiLM_seed{s}' for s in cfg['seeds']]
        draws = np.load(data / 'test' / f'{condition}_draws.npz')
        for method in methods:
            seed = -1
            if method == 'control_copy':
                prediction = x
            elif method in ('mean_delta', 'mean_delta_additive'):
                prediction = np.maximum(x + delta, 0).astype(np.float32)
            elif method == 'target_mean':
                prediction = np.broadcast_to(stats['target_mean'][ci].astype(np.float32), x.shape)
            elif method == 'variance_scale':
                prediction = np.maximum((x - stats['control_mean']) * ratio + stats['target_mean'][ci], 0).astype(np.float32)
            else:
                seed = int(method.rsplit('_seed', 1)[1])
                model = make_model(genes, stats, cfg, device)
                model.load_state_dict(torch.load(out / f'seed{seed}/best.pt', map_location=device, weights_only=False)['model'])
                prediction = predict(model, test, condition, cfg, device, additive=method.startswith('FiLM_additive'))
            method_name = method.rsplit('_seed', 1)[0] if seed >= 0 else method
            folder = out / 'test' / condition / method; folder.mkdir(parents=True, exist_ok=True)
            if seed >= 0:
                np.save(folder / 'predicted_log1p.npy', prediction)
            rows, moments, panels, observed_order, predicted_order = population_scores(prediction, truth, x, functional)
            all_rows.extend(dict(condition=condition, method=method_name, seed=seed, **r) for r in rows)
            g = pd.concat([genes[['gene_id', 'gene_symbol']], pd.DataFrame(moments)], axis=1)
            g.to_csv(folder / 'gene_predictions.tsv.gz', sep='\t', index=False)
            for name, order in [('observed', observed_order), ('predicted', predicted_order)]:
                g.iloc[order[:200]].to_csv(folder / f'{name}_effect_top200.tsv.gz', sep='\t', index=False)
            for draw, sample_seed in enumerate(cfg['distribution_seeds']):
                pidx, tidx = draws['predicted_indices'][draw], draws['target_indices'][draw]
                scores = distribution_scores(prediction[pidx], truth[tidx], reference, panels, cfg['kl_alpha'])
                distribution_rows.extend(dict(condition=condition, method=method_name,
                    seed=seed, sample_seed=sample_seed, sampled_cells=len(pidx), **r) for r in scores)
            print('TEST', condition, method, 'MSE', rows[0]['MSE_delta'], 'PCC', rows[0]['PCC_delta'], flush=True)
            pd.DataFrame(all_rows).to_csv(out / 'test_population_metrics.csv', index=False)
            pd.DataFrame(distribution_rows).to_csv(out / 'test_distribution_draws.csv', index=False)
    population, distributions = pd.DataFrame(all_rows), pd.DataFrame(distribution_rows)
    distribution_means = distributions.groupby(['condition', 'method', 'seed', 'panel'], sort=False).mean(numeric_only=True).reset_index()
    distribution_means = distribution_means.drop(columns=['sample_seed'])
    distribution_means.to_csv(out / 'test_distribution_metrics.csv', index=False)
    dist_sd = distributions.groupby(['condition', 'method', 'seed', 'panel'], sort=False).std(numeric_only=True).reset_index()
    dist_sd.to_csv(out / 'test_distribution_sampling_sd.csv', index=False)
    combined = population.merge(distribution_means, on=['condition', 'method', 'seed', 'panel'], how='left')
    combined.to_csv(out / 'test_metrics.csv', index=False)
    numeric = list(combined.select_dtypes(include=np.number).columns.difference(['seed', 'n_genes', 'sampled_cells']))
    macro = combined.groupby(['method', 'seed', 'panel'], sort=False)[numeric].mean().reset_index()
    macro['RMSE_delta'] = np.sqrt(macro.MSE_delta)
    macro.to_csv(out / 'test_macro_metrics.csv', index=False)
    combined.groupby(['condition', 'method', 'panel'], sort=False)[numeric].agg(['mean', 'std']).to_csv(out / 'test_seed_summary.csv')
    atomic_json(out / 'evaluation_complete.json', dict(protocol=signature, locked_checkpoints=locked,
        ranking='abs mean log-expression delta; no p/q filtering', Test_used_for_fitting=False,
        combination_used_for_fitting=not bool(cfg.get('within_batch') or cfg.get('within_replicate')),
        KL='true||pred; Train bins; nats; alpha=' + str(cfg['kl_alpha']),
        distribution_space='Train-fitted PCA and functional genes; 5 shared sampling draws',
        uncertainty='Sampling SD and training-seed SD separate; neither is animal uncertainty'))
    if cfg.get('within_replicate'):
        title = f'GSE102299 FiLM：Replicate {cfg["within_replicate"]}单刺激训练，IL25+NMU零样本测试'
        design = ('PBS按70/15/15划分，IL25和NMU各按80/20划分，全部IL25_NMU只用于Test。'
                  '按两个单刺激Validation MSE等权均值选择best；不混合replicate，不使用IL33训练。'
                  'FiLM使用IL25与NMU的可学习向量之和；FiLM_additive则将两个预测raw shift相加后截断。'
                  '作者replicate标签用于来源分层，实际batch/animal ID缺失，不能认定独立动物或精确配对。')
    elif cfg.get('within_batch'):
        title = f'GSE136154 FiLM：{cfg["within_batch"]}批次内单刺激训练，联合刺激留出'
        design = ('PBS按70/15/15划分，单刺激按80/20划分；全部联合细胞只用于Test。'
                  '仅按两个单刺激Validation MSE等权均值选择best，另一批次完全不用。'
                  'FiLM使用[1,1]联合条件；FiLM_additive是两个单刺激raw shift相加后只截断一次。'
                  'mean_delta_additive仅用Train单刺激均值变化之和，没有联合均值/方差监督。')
    else:
        title = 'GSE136154 FiLM 多刺激：C内训练/验证，D测试'
        design = '三种刺激均参与C Train。按C Validation三条件等权MSE选择best，D只在模型锁定后评分。'
    report = ['# ' + title, '', design, '',
        'Top-N按绝对平均log1p(CP10K)变化排名，无p/q筛选。MSE/PCC为群体响应，不是配对单细胞准确率。', '',
        '## 全基因Test', '', '| 条件 | 方法 | seed | MSE_delta | PCC_delta | R2_delta | Energy PCA | SW1 PCA | W1 gene | KL true/pred |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in combined[combined.panel.eq('All_genes')].itertuples():
        vals = [row.condition, row.method, str(row.seed)] + [f'{v:.6g}' if pd.notna(v) else 'NA' for v in
            [row.MSE_delta, row.PCC_delta, row.R2_delta, row.E_distance_PCA50, row.SW1_PCA50_128, row.W1_gene_mean, row.KL_hist_true_pred]]
        report.append('| ' + ' | '.join(vals) + ' |')
    report += ['', '完整All/Top20/50/100/200/Functional：test_metrics.csv；条件宏平均：test_macro_metrics.csv。',
        '采样SD单列test_distribution_sampling_sd.csv；训练seed汇总test_seed_summary.csv。',
        'Energy为V-statistic开根；SW1为固定128个PCA方向；W1/KL为基因边际，不能代表完整联合分布。',
        '每条件来源是作者批次/混池，不假定独立动物；这些数据已用于历史项目开发，不是新盲测。',
        '单刺激训练不能识别任意真实药物交互；联合预测好坏是本次实测问题，不能由embedding求和直接保证。', '']
    (out / 'report.md').write_text('\n'.join(report))


def check(cfg, device):
    """Temporary Train-only optimization checks, never a formal run or Test score."""
    data, out, _ = ready(cfg)
    tr = load_role(cfg, 'train'); genes = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
    with np.load(data / 'statistics.npz') as f:
        stats = dict(f)
    torch.manual_seed(123)
    model = make_model(genes, stats, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    tensor = as_tensors(tr, device)
    scale = torch.as_tensor(stats['scale'], device=device)
    n = min(cfg['batch_size'], len(tr['pbs']), *(len(y) for y in tr['targets'].values()))
    for step, condition in enumerate([*training_conditions(cfg)] * 2):
        x, h = tensor['pbs'][:n], tensor['h'][:n]
        target = tensor['targets'][condition][:n]
        c = x.new_tensor(condition_vectors(cfg)[condition]).expand(len(x), -1)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = loss_terms(model(h, x, c), target, scale, cfg)
        loss.backward(); optimizer.step()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite Train-only smoke objective')
    atomic_json(out / 'model_acceptance.json', dict(passed=True,
        scope=f'{2 * len(training_conditions(cfg))} temporary Train-only minibatch updates, discarded',
        device=str(device), genes=len(genes), FM_dimensions=tr['h'].shape[1], Test_read=False,
        batch_size=n, objective='Same mean/variance/raw-shift objective as formal training',
        parameters=sum(p.numel() for p in model.parameters())))
    print('CHECK Train-only temporary updates passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'train', 'evaluate', 'all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    with threadpool_limits(cfg['cpu_threads']):
        if args.command in ['prepare', 'all']:
            prepare(cfg)
        if args.command in ['check', 'train', 'evaluate', 'all']:
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; use --device cpu for checks or run in the GPU server environment')
            if args.command == 'check':
                check(cfg, device)
            if args.command in ['train', 'all']:
                for seed in cfg['seeds']:
                    train_seed(cfg, seed, device)
            if args.command in ['evaluate', 'all']:
                evaluate_models(cfg, device)


if __name__ == '__main__':
    main()
