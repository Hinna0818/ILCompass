"""Within-source joint retraining with nested combination supervision and a fixed Test."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_film_multistimulus as base

ROOT = base.ROOT
CONFIG = ROOT / 'configs/film_combination_fewshot_GSE136154_v1.json'
COMBO = 'IL33_CGRP'


def label(fraction):
    return f'{round(100 * fraction):02d}pct'


def sources(cfg):
    return {b: json.loads((ROOT / p).read_text()) for b, p in cfg['source_configs'].items()}


def combination(cfg):
    return cfg.get('combination', COMBO)


def combo_split(n, cfg):
    """Fractions refer to the original combination population, not the smaller pool."""
    order = np.random.default_rng(cfg['split_seed']).permutation(n)
    holdout = np.sort(order[:round(n * cfg['holdout_fraction'])])
    pool = order[len(holdout):]
    counts = {label(f): math.ceil(n * f) for f in cfg['fractions']}
    if len(holdout) < 1 or max(counts.values()) > len(pool) or min(counts.values()) < 0:
        raise ValueError('Invalid holdout/few-shot fractions')
    return holdout, pool, counts


def prepare(cfg):
    data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if (data / 'complete.json').exists():
        base.ready(cfg); print('PREPARE few-shot split reused', flush=True); return
    if (len(cfg['source_configs']) != 2 or cfg['fractions'][0] != 0
            or cfg['fractions'] != sorted(set(cfg['fractions']))
            or len({label(f) for f in cfg['fractions']}) != len(cfg['fractions'])):
        raise ValueError('Expected two sources and unique increasing fractions starting at zero')
    signatures, counts_table = {}, []
    for batch, source in sources(cfg).items():
        folder, old_out, _ = base.ready(source)
        if (str(source.get('within_replicate') or source.get('within_batch')) != batch
                or source['seeds'] != [cfg['seed']]
                or base.evaluation_conditions(source) != (combination(cfg),)):
            raise ValueError('Source identity, combination or seed mismatch')
        zero = json.loads((old_out / f'seed{cfg["seed"]}/complete.json').read_text())
        if zero['combination_used_for_training_or_selection'] or zero['best_sha256'] != base.file_hash(old_out / f'seed{cfg["seed"]}/best.pt'):
            raise ValueError('Zero-shot reference was not sealed or used combination training')
        meta = pd.read_csv(folder / 'test/metadata.tsv.gz', sep='\t', index_col=0)
        combo = meta[meta.condition.eq(combination(cfg))]
        identity = meta.author_replicate.astype(str) if source.get('within_replicate') else meta.author_batch
        if not meta.index.is_unique or not identity.eq(batch).all():
            raise ValueError('Unexpected original Test identities')
        holdout, pool, counts = combo_split(len(combo), cfg)
        dest = data / batch; dest.mkdir(exist_ok=True)
        manifest = combo.copy(); manifest['combo_source_row'] = np.arange(len(combo))
        manifest['role'] = 'adaptation_pool'; manifest.iloc[holdout, manifest.columns.get_loc('role')] = 'test'
        for fraction in cfg['fractions']:
            selected = np.zeros(len(combo), dtype=bool); selected[pool[:counts[label(fraction)]]] = True
            manifest[label(fraction)] = selected
            counts_table.append(dict(batch=batch, fraction=fraction, train_combo=counts[label(fraction)],
                                     test_combo=len(holdout), test_PBS=int(meta.condition.eq('PBS').sum())))
        manifest.to_csv(dest / 'combo_manifest.tsv.gz', sep='\t', index_label='cell_id')
        np.savez(dest / 'indices.npz', holdout=holdout, pool=pool,
                 **{label(f): np.array(counts[label(f)]) for f in cfg['fractions']})
        # Only the largest nested Train subset is materialized; held-out expression is not read here.
        y = np.load(folder / 'test/y.npy', mmap_mode='r')
        combo_rows = np.flatnonzero(meta.condition.eq(combination(cfg)))
        np.save(dest / 'train_combo.npy', np.array(y[combo_rows[pool[:max(counts.values())]]]))
        n = min(source['distribution_cells'], int(meta.condition.eq('PBS').sum()), len(holdout))
        draws = []
        for seed in source['distribution_seeds']:
            rng = np.random.default_rng(seed)
            draws.append((rng.choice(meta.condition.eq('PBS').sum(), n, replace=False), rng.choice(len(holdout), n, replace=False)))
        np.savez(dest / 'draws.npz', predicted_indices=np.stack([p for p, t in draws]),
                 target_indices=np.stack([t for p, t in draws]))
        signatures[batch] = dict(source={name: base.file_hash(folder / name) for name in
            ('complete.json', 'split_manifest.tsv.gz', 'statistics.npz', 'distribution_reference.npz')},
            zero_sha256=zero['best_sha256'], split_sha256=base.file_hash(dest / 'combo_manifest.tsv.gz'),
            train_combo_sha256=base.file_hash(dest / 'train_combo.npy'))
    pd.DataFrame(counts_table).to_csv(out / 'cell_counts.csv', index=False)
    base.atomic_json(data / 'complete.json', dict(config=cfg, signatures=signatures,
        design='Independent within-source retraining; nested combination Train, fixed combination Test',
        standardization='Original Train PBS/singles statistics and PCA/KL retained for every fraction',
        selection='Original single-stimulus Validation only; no combination or Test selection'))
    base.atomic_json(out / 'config.json', cfg)
    print('PREPARED\n' + pd.DataFrame(counts_table).to_string(index=False), flush=True)


def protocol(cfg):
    data, _, manifest = base.ready(cfg)
    for batch, source in sources(cfg).items():
        folder, out, _ = base.ready(source); recorded = manifest['signatures'][batch]
        if any(base.file_hash(folder / p) != h for p, h in recorded['source'].items()):
            raise ValueError('Original source split/statistics changed')
        if (base.file_hash(data / batch / 'combo_manifest.tsv.gz') != recorded['split_sha256']
                or base.file_hash(data / batch / 'train_combo.npy') != recorded['train_combo_sha256']
                or base.file_hash(out / f'seed{cfg["seed"]}/best.pt') != recorded['zero_sha256']):
            raise ValueError('Few-shot subset or zero-shot weights changed')
    files = [Path(__file__), Path(base.__file__), ROOT / 'scratch/drug_pred_FM/model.py',
             Path(base.__file__).with_name('distribution_metrics.py'), Path(base.__file__).with_name('il33_lung_metrics.py')]
    return dict(config=cfg, prepared_sha256=base.file_hash(data / 'complete.json'),
                evaluation_indices_sha256={b: {p: base.file_hash(data / b / p)
                    for p in ('indices.npz', 'draws.npz')} for b in sources(cfg)},
                code_sha256={str(p.relative_to(ROOT)): base.file_hash(p) for p in files})


def train_group(cfg, batch, fraction):
    source = sources(cfg)[batch]
    group = base.load_role(source, 'train')
    with np.load(ROOT / cfg['data_directory'] / batch / 'indices.npz') as idx:
        n = int(idx[label(fraction)])
    group['targets'][combination(cfg)] = np.array(np.load(ROOT / cfg['data_directory'] / batch / 'train_combo.npy', mmap_mode='r')[:n])
    return group


def train_one(cfg, batch, fraction, device):
    data, out, _ = base.ready(cfg); signature = protocol(cfg)
    dest = out / batch / label(fraction); dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest / 'complete.json').exists():
            done = json.loads((dest / 'complete.json').read_text())
            if done['protocol'] != signature or done['best_sha256'] != base.file_hash(dest / 'best.pt'):
                raise ValueError('Completed few-shot run changed')
            print('TRAIN reused', batch, label(fraction), flush=True); return
        source = sources(cfg)[batch]; folder, _, _ = base.ready(source)
        train, validation = train_group(cfg, batch, fraction), base.load_role(source, 'validation')
        genes = pd.read_csv(folder / 'gene_panel.tsv', sep='\t')
        with np.load(folder / 'statistics.npz') as f:
            stats = dict(f)
        torch.manual_seed(cfg['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(cfg['seed'])
        model = base.make_model(genes, stats, source, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=source['learning_rate'], weight_decay=source['weight_decay'])
        tensor = base.as_tensors(train, device); scale = torch.tensor(stats['scale'], device=device)
        rng = np.random.default_rng(cfg['seed'])
        history, best, stale, start = [], float('inf'), 0, 1
        if (dest / 'last.pt').exists():
            last = torch.load(dest / 'last.pt', map_location='cpu', weights_only=False)
            if last['protocol'] != signature or last['device'] != str(device):
                raise ValueError('Resume protocol/device differs')
            model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
            rng.bit_generator.state = last['numpy_rng']; torch.set_rng_state(last['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(last['cuda_rng'])
            history, best, stale, start = last['history'], last['best'], last['stale'], last['epoch'] + 1
        clock = time.monotonic()
        for epoch in range(start, source['epochs'] + 1):
            if epoch > source['minimum_epochs'] and stale >= source['patience']:
                break
            # The unchanged three-condition sampler gives each condition equal total weight.
            batches = base.epoch_batches(train['targets'], source, rng, conditions=tuple(train['targets']))
            model.train(); totals = np.zeros(4)
            for condition, target_ids, weight in batches:
                control = rng.choice(len(train['pbs']), len(target_ids), replace=True)
                x, h = tensor['pbs'][control], tensor['h'][control]
                target = tensor['targets'][condition][target_ids]
                c = x.new_tensor(base.condition_vectors(source)[condition]).expand(len(x), -1)
                optimizer.zero_grad(set_to_none=True)
                terms = base.loss_terms(model(h, x, c), target, scale, source)
                if not torch.isfinite(terms[0]):
                    raise ValueError('Nonfinite few-shot objective')
                (terms[0] * weight * len(batches)).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), source['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([v.detach().item() for v in terms]) * weight
            values = base.validation_scores(model, validation, source, device)
            score = float(np.mean(list(values.values())))
            if score < best:
                best, stale = score, 0
                base.atomic_torch(dest / 'best.pt', dict(model=model.state_dict(), epoch=epoch, protocol=signature))
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_loss=totals[0], train_mean=totals[1], train_variance=totals[2],
                train_shift=totals[3], val_macro_MSE_delta=score, best=best, stale=stale,
                grad_norm=float(norm), **{f'val_{c}_MSE_delta': v for c, v in values.items()}))
            base.atomic_torch(dest / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, protocol=signature, device=str(device), history=history, best=best, stale=stale,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            pd.DataFrame(history).to_csv(dest / 'history.csv', index=False)
            print(f'Fewshot/{batch}/{label(fraction)} epoch={epoch} train={totals[0]:.6g} '
                  f'val={score:.6g} best={best:.6g} stale={stale} elapsed={time.monotonic()-clock:.0f}s', flush=True)
        pd.DataFrame(history).to_csv(dest / 'history.csv', index=False)
        selected = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
        base.atomic_json(dest / 'complete.json', dict(protocol=signature, batch=batch, fraction=fraction,
            train_combo=len(train['targets'][combination(cfg)]), best_epoch=selected['epoch'], epochs_run=history[-1]['epoch'],
            best_validation_MSE=best, best_sha256=base.file_hash(dest / 'best.pt'), encoder_updates=0,
            initialization='Same random seed as zero-shot, not a warm start',
            selection='Single-stimulus Validation only', combination_loss_weight=1 / 3))


def load_test(cfg, batch):
    source = sources(cfg)[batch]; folder, _, _ = base.ready(source)
    meta = pd.read_csv(folder / 'test/metadata.tsv.gz', sep='\t', index_col=0)
    rows = np.flatnonzero(meta.condition.eq(combination(cfg)))
    with np.load(ROOT / cfg['data_directory'] / batch / 'indices.npz') as indices:
        holdout = indices['holdout']
    y = np.load(folder / 'test/y.npy', mmap_mode='r')
    return dict(pbs=np.array(y[meta.condition.eq('PBS').to_numpy()]), h=np.load(folder / 'test/h.npy'),
                targets={combination(cfg): np.array(y[rows[holdout]])})


def evaluate(cfg, device):
    data, out, manifest = base.ready(cfg); signature = protocol(cfg); locked = {}
    for batch in sources(cfg):
        for fraction in cfg['fractions'][1:]:
            dest = out / batch / label(fraction)
            done = json.loads((dest / 'complete.json').read_text())
            if done['protocol'] != signature or base.file_hash(dest / 'best.pt') != done['best_sha256']:
                raise ValueError('Lock every fraction before scoring held-out combination expression')
            locked[f'{batch}/{label(fraction)}'] = done
    path = out / 'test_protocol.json'
    if path.exists() and json.loads(path.read_text()) != locked:
        raise ValueError('Test checkpoints changed')
    base.atomic_json(path, locked)
    population, distributions = [], []
    for batch, source in sources(cfg).items():
        folder, zero_out, _ = base.ready(source)
        group = load_test(cfg, batch); x = group['pbs']; truth = group['targets'][combination(cfg)]
        genes = pd.read_csv(folder / 'gene_panel.tsv', sep='\t')
        with np.load(folder / 'statistics.npz') as f:
            stats = dict(f)
        reference = dict(np.load(folder / 'distribution_reference.npz'))
        draws = np.load(data / batch / 'draws.npz')
        tasks = [(0., 'control_copy'), (0., 'single_mean_delta_additive')]
        tasks += [(f, method) for f in cfg['fractions'] for method in
                  (('FiLM',) if f == 0 else ('FiLM', 'combo_train_mean', 'combo_mean_delta'))]
        for fraction, method in tasks:
            if method == 'control_copy':
                prediction = x
            elif method == 'single_mean_delta_additive':
                prediction = np.maximum(x + (stats['target_mean'] - stats['control_mean']).sum(0), 0).astype(np.float32)
            elif method == 'FiLM':
                checkpoint = (zero_out / f'seed{cfg["seed"]}/best.pt' if fraction == 0
                              else out / batch / label(fraction) / 'best.pt')
                model = base.make_model(genes, stats, source, device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'])
                prediction = base.predict(model, group, combination(cfg), source, device)
            else:
                with np.load(data / batch / 'indices.npz') as indices:
                    n = int(indices[label(fraction)])
                train_combo = np.load(data / batch / 'train_combo.npy', mmap_mode='r')[:n]
                mean = train_combo.mean(0, dtype=np.float64)
                prediction = (np.broadcast_to(mean.astype(np.float32), x.shape) if method == 'combo_train_mean'
                              else np.maximum(x + mean - stats['control_mean'], 0).astype(np.float32))
            dest = out / 'test' / batch / label(fraction) / method; dest.mkdir(parents=True, exist_ok=True)
            if method == 'FiLM':
                np.save(dest / 'predicted_log1p.npy', prediction)
            rows, moments, panels, observed, predicted = base.population_scores(prediction, truth, x, reference['functional_indices'])
            population.extend(dict(batch=batch, fraction=fraction, method=method, seed=cfg['seed'], **r) for r in rows)
            g = pd.concat([genes[['gene_id', 'gene_symbol']], pd.DataFrame(moments)], axis=1)
            g.to_csv(dest / 'gene_predictions.tsv.gz', sep='\t', index=False)
            for name, order in [('observed', observed), ('predicted', predicted)]:
                g.iloc[order[:200]].to_csv(dest / f'{name}_effect_top200.tsv.gz', sep='\t', index=False)
            for draw, seed in enumerate(source['distribution_seeds']):
                pi, ti = draws['predicted_indices'][draw], draws['target_indices'][draw]
                values = base.distribution_scores(prediction[pi], truth[ti], reference, panels, source['kl_alpha'])
                distributions.extend(dict(batch=batch, fraction=fraction, method=method, seed=cfg['seed'],
                    sample_seed=seed, **r) for r in values)
            print('TEST fewshot', batch, label(fraction), method, 'MSE', rows[0]['MSE_delta'], flush=True)
            pd.DataFrame(population).to_csv(out / 'test_population_metrics.csv', index=False)
            pd.DataFrame(distributions).to_csv(out / 'test_distribution_draws.csv', index=False)
    p, d = pd.DataFrame(population), pd.DataFrame(distributions)
    keys = ['batch', 'fraction', 'method', 'seed', 'panel']
    mean = d.drop(columns='sample_seed').groupby(keys, sort=False).mean(numeric_only=True).reset_index()
    d.drop(columns='sample_seed').groupby(keys, sort=False).std(numeric_only=True).to_csv(out / 'test_distribution_sampling_sd.csv')
    scores = p.merge(mean, on=keys, how='left')
    scores.to_csv(out / 'test_metrics.csv', index=False)
    counts = pd.read_csv(out / 'cell_counts.csv', dtype={'batch': str})
    scores[scores.method.eq('FiLM')].merge(counts, on=['batch', 'fraction'],
        validate='many_to_one').to_csv(out / 'FiLM_summary.csv', index=False)
    base.atomic_json(out / 'evaluation_complete.json', dict(protocol=signature, locked=locked,
        zero_references=manifest['signatures'], selection='Single-stimulus validation; no Test',
        ranking='abs mean log1p(CP10K) delta; no p-values',
        uncertainty='One optimization seed; distribution subsampling SD is not animal uncertainty'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(cfg['cpu_threads']):
        if args.command in ('prepare', 'all'):
            prepare(cfg)
        if args.command == 'prepare':
            return
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.command in ('train', 'all'):
            for batch in sources(cfg):
                for fraction in cfg['fractions'][1:]:
                    train_one(cfg, batch, fraction, device)
        if args.command in ('evaluate', 'all'):
            evaluate(cfg, device)


if __name__ == '__main__':
    main()
