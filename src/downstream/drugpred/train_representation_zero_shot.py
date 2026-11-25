"""Zero-shot combination benchmark matched to the locked 5% experiment.

Reuse the original single-stimulus trainer, PCA384 and exact expression assets.
No combination cells or combination loss term enter training/selection. All
18 independently initialized heads must finish before fixed-holdout evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_representation_benchmark as benchmark

base, few, ROOT = benchmark.base, benchmark.few, benchmark.ROOT
METHODS = benchmark.METHODS
CONFIG = ROOT / 'configs/representation_benchmark_GSE102299_00pct_v1.json'


def reference_config(cfg):
    return json.loads((ROOT / cfg['reference_benchmark_config']).read_text())


def run_config(cfg, rep, method):
    _, source, _, _ = benchmark.source_inputs(reference_config(cfg), rep)
    return dict(source, seeds=cfg['seeds'],
        data_directory=str(Path(cfg['data_directory']) / f'Rep{rep}' / method),
        result_directory=str(Path(cfg['result_directory']) / f'Rep{rep}' / method),
        workflow_script=str(Path(__file__).relative_to(ROOT)))


def link_asset(source, dest):
    source = source.resolve()
    if dest.is_symlink() or dest.exists():
        if not dest.is_symlink() or dest.resolve() != source:
            raise ValueError(f'Asset destination differs: {dest}')
    else:
        dest.symlink_to(source)


def prepare_model(cfg, rep, method, source, genes, stats, groups, provenance):
    """Materialize only cell features; shared expression/metadata remain read-only inputs."""
    rc = run_config(cfg, rep, method)
    dest = ROOT / rc['data_directory']; dest.mkdir(parents=True, exist_ok=True)
    (ROOT / rc['result_directory']).mkdir(parents=True, exist_ok=True)
    folder, _, _ = base.ready(source)
    ref_cfg = reference_config(cfg)
    pca_folder = ROOT / ref_cfg['data_directory'] / rep
    model_stats = dict(stats)
    if method == 'ExpressionMLP_FiLM':
        model_stats.update(h_mean=stats['control_mean'],
            h_std=np.maximum(stats['control_std'], source['minimum_expression_scale']))
    elif method == 'PCA_FiLM':
        with np.load(pca_folder / 'pca.npz') as f:
            model_stats.update(h_mean=f['h_mean'], h_std=f['h_std'])
    for role, group in groups.items():
        if set(group['targets']) != set(base.training_conditions(source)):
            raise ValueError('Zero-shot roles may contain only PBS and single stimuli')
        d = dest / role; d.mkdir(exist_ok=True)
        for name in ('y.npy', 'metadata.tsv.gz'):
            link_asset(folder / role / name, d / name)
        if method == 'ExpressionMLP_FiLM':
            np.save(d / 'h.npy', group['pbs'])
        elif method == 'PCA_FiLM':
            link_asset(pca_folder / f'{role}_h.npy', d / 'h.npy')
        else:
            link_asset(folder / role / 'h.npy', d / 'h.npy')
    for name in ('gene_panel.tsv', 'distribution_reference.npz'):
        link_asset(folder / name, dest / name)
    np.savez(dest / 'statistics.npz', **model_stats)
    feature_shapes = {role: list(np.load(dest / role / 'h.npy', mmap_mode='r').shape) for role in groups}
    if any(shape[0] != len(groups[role]['pbs']) for role, shape in feature_shapes.items()):
        raise ValueError('Control cells and cell features are misaligned')
    base.atomic_json(dest / 'complete.json', dict(config=rc, reference=provenance,
        method=method, replicate=rep, train_combination_cells=0,
        training_conditions=list(base.training_conditions(source)), condition_weights=[.5, .5],
        feature_shapes=feature_shapes, pca_refit=False, test_expression_prepared=False))


def prepare(cfg):
    ref_cfg = reference_config(cfg)
    if (cfg['fraction'] != 0 or cfg['replicates'] != ref_cfg['replicates']
            or cfg['seeds'] != ref_cfg['seeds'] or cfg['seeds'] != [0, 1, 2]):
        raise ValueError('Expected 0% joint supervision and the same Rep1/Rep2, seeds0/1/2')
    provenance = benchmark.protocol(ref_cfg)
    data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if (data / 'complete.json').exists():
        ready(cfg); print('PREPARE zero-shot reused', flush=True); return
    fc = benchmark.parent(ref_cfg); counts = []
    for rep in cfg['replicates']:
        _, source, genes, stats = benchmark.source_inputs(ref_cfg, rep)
        groups = {role: base.load_role(source, role) for role in ('train', 'validation')}
        folder, _, _ = base.ready(source)
        meta = pd.read_csv(folder / 'test/metadata.tsv.gz', sep='\t', index_col=0)
        combo = pd.read_csv(ROOT / fc['data_directory'] / rep / 'combo_manifest.tsv.gz', sep='\t', index_col=0)
        selected = set(combo.index[combo.role.eq('test')])
        test = meta[meta.condition.eq('PBS') | meta.index.isin(selected)]
        if set(test.index[test.condition.ne('PBS')]) != selected:
            raise ValueError('Fixed joint Test IDs differ from the 5% experiment')
        dest = data / f'Rep{rep}'; dest.mkdir(exist_ok=True)
        test.to_csv(dest / 'test_manifest.tsv.gz', sep='\t', index_label='cell_id')
        for role, group in groups.items():
            if set(group['meta'].index) & set(test.index):
                raise ValueError('Train/Validation and Test overlap')
            counts.extend(dict(replicate=rep, split=role, condition=c, cells=int(n))
                          for c, n in group['meta'].condition.value_counts().items())
        counts.extend(dict(replicate=rep, split='test', condition=c, cells=int(n))
                      for c, n in test.condition.value_counts().items())
        counts.append(dict(replicate=rep, split='train', condition=few.combination(fc), cells=0))
        for method in METHODS:
            prepare_model(cfg, rep, method, source, genes, stats, groups, provenance)
    pd.DataFrame(counts).to_csv(out / 'cell_counts.csv', index=False)
    base.atomic_json(data / 'complete.json', dict(config=cfg, reference=provenance,
        train_combination_cells=0, pca_refit=False, same_test_as_05pct=True,
        model_manifests={f'{r}/{m}': base.file_hash(ROOT / run_config(cfg, r, m)['data_directory'] / 'complete.json')
                         for r in cfg['replicates'] for m in METHODS}))
    base.atomic_json(out / 'config.json', cfg)
    print('PREPARED zero-shot\n'+pd.DataFrame(counts).to_string(index=False), flush=True)


def ready(cfg):
    data = ROOT / cfg['data_directory']
    saved = json.loads((data / 'complete.json').read_text())
    if saved['config'] != cfg or saved['reference'] != benchmark.protocol(reference_config(cfg)):
        raise ValueError('Zero-shot source/PCA protocol changed')
    for rep in cfg['replicates']:
        for method in METHODS:
            rc = run_config(cfg, rep, method)
            _, _, manifest = base.ready(rc)
            if (manifest['train_combination_cells'] != 0
                    or base.file_hash(ROOT / rc['data_directory'] / 'complete.json') != saved['model_manifests'][f'{rep}/{method}']):
                raise ValueError('Zero-shot model input manifest changed')
    return saved


def check(cfg):
    ready(cfg)
    checks = {}
    ref_cfg = reference_config(cfg)
    for rep in cfg['replicates']:
        _, source, genes, stats = benchmark.source_inputs(ref_cfg, rep)
        folder, _, _ = base.ready(source)
        pca = dict(np.load(ROOT / ref_cfg['data_directory'] / rep / 'pca.npz'))
        for method in METHODS:
            rc = run_config(cfg, rep, method); data = ROOT / rc['data_directory']
            for role in ('train', 'validation'):
                meta = pd.read_csv(data / role / 'metadata.tsv.gz', sep='\t', index_col=0)
                if set(meta.condition) != {'PBS', *base.training_conditions(rc)}:
                    raise ValueError('Joint or unexpected cells in zero-shot Train/Validation')
                if (data / role / 'y.npy').resolve() != (folder / role / 'y.npy').resolve():
                    raise ValueError('Expression assets differ from the 5% task')
                h = np.load(data / role / 'h.npy', mmap_mode='r')
                if method == 'ExpressionMLP_FiLM':
                    y = np.load(folder / role / 'y.npy', mmap_mode='r')
                    np.testing.assert_array_equal(h, y[meta.condition.eq('PBS').to_numpy()])
                else:
                    expected = (ROOT / ref_cfg['data_directory'] / rep / f'{role}_h.npy'
                                if method == 'PCA_FiLM' else folder / role / 'h.npy')
                    if (data / role / 'h.npy').resolve() != expected.resolve():
                        raise ValueError('Cell features differ from 5%')
            model_stats = dict(np.load(data / 'statistics.npz'))
            for field in ('scale', 'control_mean', 'target_mean'):
                np.testing.assert_array_equal(model_stats[field], stats[field])
            lengths = meta.condition.value_counts()
            batches = base.epoch_batches({c: np.empty((lengths[c], 1)) for c in base.training_conditions(rc)},
                                         rc, np.random.default_rng(0))
            for c in base.training_conditions(rc):
                np.testing.assert_allclose(sum(w for condition, _, w in batches if condition == c), .5)
            for seed in cfg['seeds']:
                torch.manual_seed(seed)
                current = base.make_model(genes, model_stats, rc, torch.device('cpu'))
                torch.manual_seed(seed)
                original = benchmark.make_model(genes, stats, source, method, pca, torch.device('cpu'))
                for name, value in current.state_dict().items():
                    other = original.state_dict()[name]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, other, rtol=0, atol=0)
                    elif value != other:
                        raise ValueError('Model semantics differ from 5%')
            checks[f'{rep}/{method}'] = dict(passed=True, joint_train_cells=0,
                single_condition_weights=[.5, .5], initial_model_identical_to_05pct_all_seeds=True,
                trainable_parameters=sum(p.numel() for p in current.parameters()), input_features=h.shape[1])
    base.atomic_json(ROOT / cfg['result_directory'] / 'input_acceptance.json',
        dict(passed=True, checks=checks, parameter_updates_in_check=0))
    print('CHECK passed: zero combination supervision; original features/scales and seed initializations', flush=True)


def lock_models(cfg):
    locked = {}
    for rep in cfg['replicates']:
        for method in METHODS:
            rc = run_config(cfg, rep, method)
            for seed in cfg['seeds']:
                dest = ROOT / rc['result_directory'] / f'seed{seed}'
                done = json.loads((dest / 'complete.json').read_text())
                if (done['protocol'] != base.protocol(rc) or done['best_sha256'] != base.file_hash(dest / 'best.pt')
                        or done['combination_used_for_training_or_selection']):
                    raise ValueError('All 18 zero-shot checkpoints must be complete and sealed')
                locked[f'{rep}/{method}/{seed}'] = done
    return locked


def evaluate(cfg, device):
    prepared = ready(cfg); locked = lock_models(cfg)
    out = ROOT / cfg['result_directory']; ref_cfg = reference_config(cfg); fc = benchmark.parent(ref_cfg)
    if (out / 'test_protocol.json').exists() and json.loads((out / 'test_protocol.json').read_text()) != locked:
        raise ValueError('Test checkpoints changed')
    base.atomic_json(out / 'test_protocol.json', locked)
    population, distribution = [], []
    for rep in cfg['replicates']:
        _, source, genes, stats = benchmark.source_inputs(ref_cfg, rep)
        group = few.load_test(fc, rep)
        x, truth = group['pbs'], group['targets'][few.combination(fc)]
        folder, _, _ = base.ready(source)
        metric_ref = dict(np.load(folder / 'distribution_reference.npz'))
        pca_ref = dict(np.load(ROOT / ref_cfg['data_directory'] / rep / 'pca.npz'))
        old = pd.read_csv(ROOT / fc['result_directory'] / f'test/{rep}/{few.label(ref_cfg["fraction"])}/FiLM/gene_predictions.tsv.gz', sep='\t')
        for method in METHODS:
            rc = run_config(cfg, rep, method)
            inputs = benchmark.represent(group, method, pca_ref)
            model_stats = dict(np.load(ROOT / rc['data_directory'] / 'statistics.npz'))
            for seed in cfg['seeds']:
                dest = ROOT / rc['result_directory'] / f'seed{seed}'
                model = base.make_model(genes, model_stats, rc, device)
                model.load_state_dict(torch.load(dest / 'best.pt', map_location=device, weights_only=False)['model'])
                prediction = base.predict(model, inputs, few.combination(fc), rc, device)
                rows, moments, panels, observed, predicted = base.population_scores(prediction, truth, x, metric_ref['functional_indices'])
                if not genes.gene_id.equals(old.gene_id):
                    raise ValueError('Output panels differ from 5%')
                np.testing.assert_allclose(moments['observed_delta'], old.observed_delta, rtol=1e-12, atol=1e-12)
                population.extend(dict(replicate=rep, method=method, seed=seed, **r) for r in rows)
                test = dest / 'test'; test.mkdir(exist_ok=True)
                np.save(test / 'predicted_log1p.npy', prediction)
                g = pd.concat([genes[['gene_id', 'gene_symbol']], pd.DataFrame(moments)], axis=1)
                g.to_csv(test / 'gene_predictions.tsv.gz', sep='\t', index=False)
                for name, order in [('observed', observed), ('predicted', predicted)]:
                    g.iloc[order[:200]].to_csv(test / f'{name}_effect_top200.tsv.gz', sep='\t', index=False)
                with np.load(ROOT / fc['data_directory'] / rep / 'draws.npz') as draws:
                    for i, draw_seed in enumerate(source['distribution_seeds']):
                        pi, ti = draws['predicted_indices'][i], draws['target_indices'][i]
                        values = base.distribution_scores(prediction[pi], truth[ti], metric_ref, panels, source['kl_alpha'])
                        distribution.extend(dict(replicate=rep, method=method, seed=seed, sample_seed=draw_seed, **r) for r in values)
                print('TEST zero-shot', rep, method, seed, 'MSE', rows[0]['MSE_delta'], flush=True)
                pd.DataFrame(population).to_csv(out / 'population_metrics.csv', index=False)
                pd.DataFrame(distribution).to_csv(out / 'distribution_draws.csv', index=False)
    p, d = pd.DataFrame(population), pd.DataFrame(distribution)
    keys = ['replicate', 'method', 'seed', 'panel']
    grouped = d.drop(columns='sample_seed').groupby(keys, sort=False)
    grouped.std(numeric_only=True).to_csv(out / 'distribution_sampling_sd.csv')
    scores = p.merge(grouped.mean(numeric_only=True).reset_index(), on=keys, how='left')
    scores.to_csv(out / 'test_metrics.csv', index=False)
    old_scores = pd.read_csv(ROOT / ref_cfg['result_directory'] / 'test_metrics.csv', dtype={'replicate': str})
    combined = pd.concat([scores.assign(fraction=0.), old_scores.assign(fraction=ref_cfg['fraction'])], ignore_index=True)
    combined.to_csv(out / 'zero_vs_05pct_metrics.csv', index=False)
    for data, name, group_keys in [(scores, 'seed_summary.csv', ['replicate', 'method', 'panel']),
            (combined, 'zero_vs_05pct_summary.csv', ['fraction', 'replicate', 'method', 'panel'])]:
        numeric = data.select_dtypes(include='number').columns.difference(['seed', 'fraction'])
        summary = data.groupby(group_keys, sort=False)[numeric].agg(['mean', 'std'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(out / name, index=False)
    base.atomic_json(out / 'evaluation_complete.json', dict(config=cfg, prepared=prepared, locked=locked,
        train_combination_cells=0, same_test_as_05pct=True, ranking='absolute mean log1p(CP10K) effect; no p-values',
        uncertainty='Three downstream optimization seeds, not independent animals or pretraining seeds'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'train', 'evaluate', 'all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    _, source, _, _ = benchmark.source_inputs(reference_config(cfg), cfg['replicates'][0])
    torch.set_num_threads(source['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(source['cpu_threads']):
        if args.command in ('prepare', 'all'):
            prepare(cfg)
        if args.command == 'prepare':
            return
        ready(cfg)
        if args.command in ('check', 'all'):
            check(cfg)
        if args.command == 'check':
            return
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.command in ('train', 'all'):
            for rep in cfg['replicates']:
                for method in METHODS:
                    rc = run_config(cfg, rep, method)
                    print('TRAIN zero-shot', rep, method, flush=True)
                    for seed in cfg['seeds']:
                        base.train_seed(rc, seed, device)
        if args.command in ('evaluate', 'all'):
            evaluate(cfg, device)


if __name__ == '__main__':
    main()
