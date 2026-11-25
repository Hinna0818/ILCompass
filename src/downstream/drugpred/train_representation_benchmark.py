"""Fixed few-shot splits, three cell representations, paired optimization seeds.

Expression MLP, Train-PBS PCA and frozen Scratch v3 share FiLM/decoder, losses,
sampling and single-stimulus checkpoint selection. PCA and v3 heads have the
same input dimension and parameter count. Training seeds do not change splits.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from threadpoolctl import threadpool_limits

import train_expression_film_comparison as expression

few, base = expression.few, expression.base
ROOT = base.ROOT
CONFIG = ROOT / 'configs/representation_benchmark_GSE102299_05pct_v1.json'
METHODS = ('ExpressionMLP_FiLM', 'PCA_FiLM', 'Scratch_v3_FiLM')


def parent(cfg):
    return json.loads((ROOT / cfg['fewshot_config']).read_text())


def source_inputs(cfg, rep):
    fc = parent(cfg); source = few.sources(fc)[rep]
    folder, _, _ = base.ready(source)
    genes = pd.read_csv(folder / 'gene_panel.tsv', sep='\t')
    with np.load(folder / 'statistics.npz') as f:
        stats = dict(f)
    return fc, source, genes, stats


def fit_pca(group, stats, source, cfg):
    """No treated, Validation or Test cells enter the representation fit."""
    x = group['pbs']
    if not 0 < cfg['pca_dimensions'] < min(x.shape):
        raise ValueError('PCA dimensions must be below Train PBS cell/gene counts')
    scale = np.maximum(stats['control_std'], source['minimum_expression_scale'])
    z = (x - stats['control_mean']) / scale
    pca = PCA(cfg['pca_dimensions'], svd_solver='randomized', iterated_power=3, random_state=cfg['pca_seed'])
    pca.fit(z)
    # Use transform for both Train and inference; randomized fit_transform can differ.
    h = pca.transform(z).astype(np.float32)
    return dict(input_mean=stats['control_mean'], input_scale=scale,
        pca_mean=pca.mean_, components=pca.components_, explained_variance_ratio=pca.explained_variance_ratio_,
        h_mean=h.mean(0), h_std=h.std(0)), h


def transform(x, reference):
    return (((x - reference['input_mean']) / reference['input_scale'] - reference['pca_mean'])
            @ reference['components'].T).astype(np.float32)


def prepare(cfg):
    fc = parent(cfg); data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    parent_signature = few.protocol(fc)
    if (data / 'complete.json').exists():
        saved = json.loads((data / 'complete.json').read_text())
        if saved['config'] != cfg or saved['parent'] != parent_signature:
            raise ValueError('Prepared benchmark changed; use a new experiment directory')
        print('PREPARE benchmark reused', flush=True); return
    if cfg['replicates'] != ['1', '2'] or cfg['seeds'] != [0, 1, 2] or cfg['fraction'] not in fc['fractions'][1:]:
        raise ValueError('Expected Rep1/Rep2, seeds 0/1/2 and a prepared positive fraction')
    checks = {}
    for rep in cfg['replicates']:
        _, source, _, stats = source_inputs(cfg, rep)
        group = base.load_role(source, 'train')
        if group['h'].shape[1] != cfg['pca_dimensions']:
            raise ValueError('PCA must match the frozen FM embedding dimension')
        reference, h = fit_pca(group, stats, source, cfg)
        va = base.load_role(source, 'validation')
        dest = data / rep; dest.mkdir(exist_ok=True)
        np.savez(dest / 'pca.npz', **reference)
        np.save(dest / 'train_h.npy', h)
        np.save(dest / 'validation_h.npy', transform(va['pbs'], reference))
        ids = group['meta'].index[group['meta'].condition.eq('PBS')]
        pd.Series(ids, name='cell_id').to_csv(dest / 'pca_fit_cells.tsv.gz', sep='\t', index=False)
        combo = pd.read_csv(ROOT / fc['data_directory'] / rep / 'combo_manifest.tsv.gz', sep='\t', index_col=0)
        train_ids = set(combo.index[combo[few.label(cfg['fraction'])]])
        test_ids = set(combo.index[combo.role.eq('test')])
        if train_ids & test_ids:
            raise ValueError('Joint Train/Test overlap')
        checks[rep] = dict(train_PBS=len(ids), train_combo=len(train_ids), test_combo=len(test_ids),
            pca_dimensions=h.shape[1], explained_variance_ratio_sum=float(reference['explained_variance_ratio'].sum()),
            pca_fit_scope='Original Train PBS only', validation_or_test_used_for_PCA=False,
            hashes={p: base.file_hash(dest / p) for p in ('pca.npz', 'train_h.npy', 'validation_h.npy', 'pca_fit_cells.tsv.gz')})
        print('PREPARE', rep, {k: v for k, v in checks[rep].items() if k != 'hashes'}, flush=True)
    base.atomic_json(data / 'complete.json', dict(config=cfg, parent=parent_signature, checks=checks))
    base.atomic_json(out / 'config.json', cfg)


def protocol(cfg):
    data = ROOT / cfg['data_directory']
    saved = json.loads((data / 'complete.json').read_text())
    if saved['config'] != cfg or saved['parent'] != few.protocol(parent(cfg)):
        raise ValueError('Source benchmark protocol changed')
    for rep, check in saved['checks'].items():
        if any(base.file_hash(data / rep / name) != digest for name, digest in check['hashes'].items()):
            raise ValueError('PCA features/reference changed')
    return dict(config=cfg, prepared_sha256=base.file_hash(data / 'complete.json'),
        parent=saved['parent'], code_sha256=base.file_hash(Path(__file__)),
        expression_code_sha256=base.file_hash(Path(expression.__file__)))


def represent(group, method, reference=None, cached=None):
    if method == 'ExpressionMLP_FiLM':
        return expression.expression_group(group)
    if method == 'PCA_FiLM':
        return dict(group, h=cached if cached is not None else transform(group['pbs'], reference))
    if method == 'Scratch_v3_FiLM':
        return group
    raise ValueError(f'Unknown representation: {method}')


def make_model(genes, stats, source, method, reference, device):
    if method == 'ExpressionMLP_FiLM':
        return expression.make_model(genes, stats, source, device)
    if method == 'PCA_FiLM':
        return base.FiLMPerturbationModel(genes.gene_id.tolist(), reference['h_mean'], reference['h_std'],
                                        **source['model']).to(device)
    return base.make_model(genes, stats, source, device)


def run_path(cfg, rep, method, seed):
    return ROOT / cfg['result_directory'] / f'Rep{rep}' / method / f'seed{seed}'


def reference_run(cfg, rep, method, seed):
    """Reuse three completed seed0 runs, never mix their splits or select by Test."""
    if seed != 0 or method == 'PCA_FiLM':
        return None
    fc = parent(cfg)
    if method == 'Scratch_v3_FiLM':
        path = ROOT / fc['result_directory'] / rep / few.label(cfg['fraction'])
        done = json.loads((path / 'complete.json').read_text())
        expected = few.protocol(fc)
    elif rep == '1':
        ec = json.loads((ROOT / cfg['expression_reference_config']).read_text())
        if ec['fraction'] != cfg['fraction'] or ec['replicate'] != rep or ec['fewshot_config'] != cfg['fewshot_config']:
            raise ValueError('Expression reference task differs')
        path = ROOT / ec['result_directory']
        done = json.loads((path / 'complete.json').read_text())
        expected = expression.protocol(ec)
    else:
        return None
    if done['protocol'] != expected or done['best_sha256'] != base.file_hash(path / 'best.pt'):
        raise ValueError('Original seed0 reference changed')
    return path, done


def train_one(cfg, rep, method, seed, tr, va, reference, signature, device):
    dest = run_path(cfg, rep, method, seed); dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest / 'complete.json').exists():
            done = json.loads((dest / 'complete.json').read_text())
            if done['protocol'] != signature or done['best_sha256'] != base.file_hash(dest / 'best.pt'):
                raise ValueError('Completed benchmark run changed')
            print('TRAIN reused', rep, method, seed, flush=True); return
        fc, source, genes, stats = source_inputs(cfg, rep)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        model = make_model(genes, stats, source, method, reference, device)
        parameters = sum(p.numel() for p in model.parameters())
        old = reference_run(cfg, rep, method, seed)
        if old is not None:
            origin, done = old
            shutil.copyfile(origin / 'best.pt', dest / 'best.pt')
            shutil.copyfile(origin / 'history.csv', dest / 'history.csv')
            base.atomic_json(dest / 'complete.json', dict(protocol=signature, replicate=rep, method=method, seed=seed,
                **{k: done[k] for k in ('best_epoch', 'epochs_run', 'best_validation_MSE', 'best_sha256', 'train_combo')},
                trainable_parameters=parameters, reused_from=str(origin.relative_to(ROOT)),
                origin_complete_sha256=base.file_hash(origin / 'complete.json')))
            print('TRAIN original seed0 copied', rep, method, flush=True); return
        optimizer = torch.optim.AdamW(model.parameters(), lr=source['learning_rate'], weight_decay=source['weight_decay'])
        tensor = base.as_tensors(tr, device); scale = torch.tensor(stats['scale'], device=device)
        rng = np.random.default_rng(seed)
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
            batches = base.epoch_batches(tr['targets'], source, rng, conditions=tuple(tr['targets']))
            model.train(); totals = np.zeros(4)
            for condition, ids, weight in batches:
                control = rng.choice(len(tr['pbs']), len(ids), replace=True)
                x, h = tensor['pbs'][control], tensor['h'][control]
                c = x.new_tensor(base.condition_vectors(source)[condition]).expand(len(x), -1)
                optimizer.zero_grad(set_to_none=True)
                terms = base.loss_terms(model(h, x, c), tensor['targets'][condition][ids], scale, source)
                if not torch.isfinite(terms[0]):
                    raise ValueError('Nonfinite objective')
                (terms[0] * weight * len(batches)).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), source['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([v.detach().item() for v in terms]) * weight
            values = base.validation_scores(model, va, source, device)
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
            print(f'Rep{rep}/{method}/seed{seed} epoch={epoch} train={totals[0]:.6g} val={score:.6g} '
                  f'best={best:.6g} stale={stale} elapsed={time.monotonic()-clock:.0f}s', flush=True)
        selected = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
        base.atomic_json(dest / 'complete.json', dict(protocol=signature, replicate=rep, method=method, seed=seed,
            best_epoch=selected['epoch'], epochs_run=history[-1]['epoch'], best_validation_MSE=best,
            best_sha256=base.file_hash(dest / 'best.pt'), train_combo=len(tr['targets'][few.combination(fc)]),
            trainable_parameters=parameters, selection='Single-stimulus Validation only',
            combination_loss_weight=1/3, reused_from=None))


def train(cfg, device):
    signature = protocol(cfg); fc = parent(cfg)
    for rep in cfg['replicates']:
        _, source, _, _ = source_inputs(cfg, rep)
        tr, va = few.train_group(fc, rep, cfg['fraction']), base.load_role(source, 'validation')
        data = ROOT / cfg['data_directory'] / rep
        with np.load(data / 'pca.npz') as f:
            reference = dict(f)
        for method in METHODS:
            a = represent(tr, method, reference, np.load(data / 'train_h.npy') if method == 'PCA_FiLM' else None)
            b = represent(va, method, reference, np.load(data / 'validation_h.npy') if method == 'PCA_FiLM' else None)
            for seed in cfg['seeds']:
                train_one(cfg, rep, method, seed, a, b, reference, signature, device)


def evaluate(cfg, device):
    signature = protocol(cfg); out = ROOT / cfg['result_directory']; fc = parent(cfg)
    locked = {}
    for rep in cfg['replicates']:
        for method in METHODS:
            for seed in cfg['seeds']:
                dest = run_path(cfg, rep, method, seed)
                done = json.loads((dest / 'complete.json').read_text())
                if done['protocol'] != signature or done['best_sha256'] != base.file_hash(dest / 'best.pt'):
                    raise ValueError('All 18 checkpoints must be locked before Test')
                locked[f'{rep}/{method}/{seed}'] = done
    if (out / 'test_protocol.json').exists() and json.loads((out / 'test_protocol.json').read_text()) != locked:
        raise ValueError('Test checkpoints changed')
    base.atomic_json(out / 'test_protocol.json', locked)
    populations, distributions = [], []
    for rep in cfg['replicates']:
        _, source, genes, stats = source_inputs(cfg, rep)
        folder, _, _ = base.ready(source)
        group = few.load_test(fc, rep); x, truth = group['pbs'], group['targets'][few.combination(fc)]
        with np.load(folder / 'distribution_reference.npz') as f:
            metric_reference = dict(f)
        with np.load(ROOT / cfg['data_directory'] / rep / 'pca.npz') as f:
            reference = dict(f)
        old = pd.read_csv(ROOT / fc['result_directory'] / f'test/{rep}/{few.label(cfg["fraction"])}/FiLM/gene_predictions.tsv.gz', sep='\t')
        for method in METHODS:
            inputs = represent(group, method, reference)
            for seed in cfg['seeds']:
                dest = run_path(cfg, rep, method, seed)
                model = make_model(genes, stats, source, method, reference, device)
                model.load_state_dict(torch.load(dest / 'best.pt', map_location=device, weights_only=False)['model'])
                prediction = base.predict(model, inputs, few.combination(fc), source, device)
                rows, moments, panels, observed, predicted = base.population_scores(prediction, truth, x, metric_reference['functional_indices'])
                if not genes.gene_id.equals(old.gene_id):
                    raise ValueError('Panel mismatch with original Test')
                np.testing.assert_allclose(moments['observed_delta'], old.observed_delta, rtol=1e-12, atol=1e-12)
                populations.extend(dict(replicate=rep, method=method, seed=seed, **r) for r in rows)
                test = dest / 'test'; test.mkdir(exist_ok=True)
                np.save(test / 'predicted_log1p.npy', prediction)
                g = pd.concat([genes[['gene_id', 'gene_symbol']], pd.DataFrame(moments)], axis=1)
                g.to_csv(test / 'gene_predictions.tsv.gz', sep='\t', index=False)
                for name, order in [('observed', observed), ('predicted', predicted)]:
                    g.iloc[order[:200]].to_csv(test / f'{name}_effect_top200.tsv.gz', sep='\t', index=False)
                with np.load(ROOT / fc['data_directory'] / rep / 'draws.npz') as draws:
                    for i, draw_seed in enumerate(source['distribution_seeds']):
                        pi, ti = draws['predicted_indices'][i], draws['target_indices'][i]
                        values = base.distribution_scores(prediction[pi], truth[ti], metric_reference, panels, source['kl_alpha'])
                        distributions.extend(dict(replicate=rep, method=method, seed=seed, sample_seed=draw_seed, **r) for r in values)
                print('TEST', rep, method, seed, 'MSE', rows[0]['MSE_delta'], 'Top100 RMSE', rows[3]['RMSE_delta'], flush=True)
                pd.DataFrame(populations).to_csv(out / 'population_metrics.csv', index=False)
                pd.DataFrame(distributions).to_csv(out / 'distribution_draws.csv', index=False)
    p, d = pd.DataFrame(populations), pd.DataFrame(distributions)
    keys = ['replicate', 'method', 'seed', 'panel']
    grouped = d.drop(columns='sample_seed').groupby(keys, sort=False)
    grouped.std(numeric_only=True).to_csv(out / 'distribution_sampling_sd.csv')
    p.merge(grouped.mean(numeric_only=True).reset_index(), on=keys, how='left').to_csv(out / 'test_metrics.csv', index=False)
    base.atomic_json(out / 'evaluation_complete.json', dict(protocol=signature, locked=locked,
        same_test_as_original=True, ranking='absolute mean log1p(CP10K) effect, no p-values',
        uncertainty='Three optimization seeds; fixed cell split and fixed few-shot draw; not animal replicates'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    _, source, _, _ = source_inputs(cfg, cfg['replicates'][0])
    torch.set_num_threads(source['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(source['cpu_threads']):
        if args.command in ('prepare', 'all'):
            prepare(cfg)
        if args.command == 'prepare':
            return
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.command in ('train', 'all'):
            train(cfg, device)
        if args.command in ('evaluate', 'all'):
            evaluate(cfg, device)


if __name__ == '__main__':
    main()
