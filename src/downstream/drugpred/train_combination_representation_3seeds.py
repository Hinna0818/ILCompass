"""Frozen cell representations, three seeds, matched 0/5% joint supervision.

Reuse the original single-stimulus and few-shot trainers without changing them.
Source-specific runs stay independent; their metrics are macro-averaged per seed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_film_genecompass_zero_shot as gc
import train_representation_benchmark as representation
import train_representation_zero_shot as old_zero

base, few, ROOT = representation.base, representation.few, representation.ROOT
CONFIG = ROOT / 'configs/combination_representation_3seeds_v1.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def folder(cfg, key, suffix=''):
    return ROOT / cfg[key] / suffix


def source_inputs(cfg, experiment):
    gcc = read_json(ROOT / cfg['gc_config'])
    source = gc.source_config(gcc, experiment)
    info = cfg['experiments'][experiment]
    fc = read_json(ROOT / cfg['fewshot_configs'][info['dataset']])
    return gcc, source, fc, info


def zero_config(cfg, experiment, model):
    _, source, _, _ = source_inputs(cfg, experiment)
    return dict(source, seeds=cfg['seeds'],
        data_directory=str(Path(cfg['data_directory']) / experiment / 'zero_shot' / model),
        result_directory=str(Path(cfg['result_directory']) / experiment / 'zero_shot' / model),
        workflow_script=str(Path(__file__).relative_to(ROOT)), representation=model)


def five_config(cfg, experiment, model, seed):
    _, _, parent, info = source_inputs(cfg, experiment)
    rc = zero_config(cfg, experiment, model)
    return dict(source_configs={info['source']: str(Path(rc['data_directory']) / 'config.json')},
        combination=few.combination(parent), fractions=cfg['fractions'],
        holdout_fraction=parent['holdout_fraction'], split_seed=parent['split_seed'], seed=seed,
        cpu_threads=cfg['cpu_threads'],
        data_directory=str(Path(cfg['data_directory']) / experiment / 'five_percent' / model / f'seed{seed}'),
        result_directory=str(Path(cfg['result_directory']) / experiment / 'five_percent' / model / f'seed{seed}'))


def run_dir(cfg, experiment, model, seed, fraction):
    if fraction == 0:
        return ROOT / zero_config(cfg, experiment, model)['result_directory'] / f'seed{seed}'
    fc = five_config(cfg, experiment, model, seed)
    return ROOT / fc['result_directory'] / cfg['experiments'][experiment]['source'] / '05pct'


def feature_origin(cfg, experiment, model):
    gcc, source, _, _ = source_inputs(cfg, experiment)
    if model == 'Scratch_v3':
        return ROOT / source['data_directory']
    name = 'GeneCompass_LoRA' if model == 'LoRA' else model
    return ROOT / gc.run_config(gcc, experiment, name)['data_directory']


def legacy_run(cfg, experiment, model, seed, fraction):
    """Exact historical heads, not historical full-joint Test metrics for raw/LoRA."""
    gcc, source, parent, info = source_inputs(cfg, experiment)
    if fraction == 0 and seed == 0 and model in ('GeneCompass', 'LoRA'):
        name = 'GeneCompass_LoRA' if model == 'LoRA' else model
        rc = gc.run_config(gcc, experiment, name)
        path = ROOT / rc['result_directory'] / 'seed0'
        expected = base.protocol(rc)
    elif info['dataset'] == 'GSE102299' and model in ('ExpressionMLP', 'Scratch_v3'):
        key = 'legacy_expression_zero' if fraction == 0 else 'legacy_expression_five'
        old = read_json(ROOT / cfg[key]); rep = info['source']; method = model+'_FiLM'
        if fraction == 0:
            rc = old_zero.run_config(old, rep, method)
            path = ROOT / rc['result_directory'] / f'seed{seed}'
            expected = base.protocol(rc)
        else:
            path = representation.run_path(old, rep, method, seed)
            expected = representation.protocol(old)
    else:
        # GSE136154 native Scratch seed0 predates the current FiLM code hashes.
        # Use current trainers for all three seeds, keeping those old heads read-only.
        return None
    done = read_json(path / 'complete.json')
    if done['protocol'] != expected or done['best_sha256'] != base.file_hash(path / 'best.pt'):
        raise ValueError(f'Historical checkpoint/protocol changed: {path}')
    if fraction == 0 and done['combination_used_for_training_or_selection']:
        raise ValueError('Historical zero-shot head used joint supervision')
    if fraction > 0:
        _, source, parent, info = source_inputs(cfg, experiment)
        _, _, _, n = fixed_split(source, parent, info)
        if done['train_combo'] != n:
            raise ValueError('Historical head used a different joint training subset size')
    return path, done


def fixed_split(source, parent, info):
    data, _, _ = base.ready(source)
    test = pd.read_csv(data / 'test/metadata.tsv.gz', sep='\t', index_col=0)
    old = ROOT / parent['data_directory'] / info['source']
    manifest = pd.read_csv(old / 'combo_manifest.tsv.gz', sep='\t', index_col=0)
    combo = test[test.condition.eq(few.combination(parent))]
    np.testing.assert_array_equal(manifest.index, combo.index)
    expected, pool, counts = few.combo_split(len(combo), parent)
    if (not np.array_equal(np.flatnonzero(manifest.role.eq('test')), expected)
            or int(manifest['05pct'].sum()) != counts['05pct']
            or (manifest['05pct'] & manifest.role.eq('test')).any()):
        raise ValueError('Historical 5% Train and 80% Test split differs')
    selected = np.zeros(len(combo), dtype=bool); selected[pool[:counts['05pct']]] = True
    np.testing.assert_array_equal(manifest['05pct'], selected)
    return old, test, manifest, counts['05pct']


def validate_scratch_features(cfg):
    """Align native cached PBS vectors; no treated expression or encoder inference."""
    checkpoint = cfg['scratch_checkpoint']
    checkpoint_hash = base.file_hash(ROOT / checkpoint)
    checks = {}
    for experiment in cfg['experiments']:
        _, source, _, info = source_inputs(cfg, experiment)
        parent, _, saved = base.ready(source)
        groups = gc.validate_roles(source, parent)
        corpus = pd.read_csv(ROOT / source['pretraining_manifest'], sep='\t', usecols=['cell_id', 'study'])
        cache = ROOT / source['feature_directory']
        if info['dataset'] == 'GSE136154':
            signature = read_json(cache / 'Scratch_v3/signature.json')
            if saved['frozen_features_signature'] != signature or signature['checkpoint']['sha256'] != checkpoint_hash:
                raise ValueError('Historical Scratch feature signature/checkpoint changed')
            if signature['checkpoint']['path'] != checkpoint or signature['dimensions'] != 384:
                raise ValueError('Wrong Scratch checkpoint or native dimension')
            paths = [(cache/'Scratch_v3/PBS_h.npy', signature['features_sha256']),
                     (cache/'PBS_metadata.tsv.gz', signature['common_assets_sha256']['PBS_metadata.tsv.gz'])]
            metadata = pd.read_csv(cache/'PBS_metadata.tsv.gz', sep='\t', index_col=0)
            features = np.load(cache/'Scratch_v3/PBS_h.npy', mmap_mode='r')
            provenance = signature['frozen_provenance']
        else:
            signature = read_json(cache/'prepare_manifest.json')
            evidence = saved['frozen_features']
            if (evidence['checkpoint_sha256'] != checkpoint_hash or evidence['encoder_updates'] != 0
                    or evidence['original_manifest_sha256'] != base.file_hash(cache/'prepare_manifest.json')
                    or signature['checkpoint_sha256']['Scratch_v3'] != checkpoint_hash):
                raise ValueError('Historical Scratch provenance/checkpoint changed')
            metadata, features, paths = [], [], []
            for role in ('train', 'validation'):
                path = cache/role/'Scratch_v3_h.npy'
                paths.append((path, signature['aligned_feature_sha256'][role]))
                m = pd.read_csv(cache/role/'metadata.tsv.gz', sep='\t', index_col=0,
                                usecols=['cell_id', 'condition'])
                metadata.append(m[m.condition.eq('PBS')])
                features.append(np.load(path, mmap_mode='r'))
            metadata, features = pd.concat(metadata), np.concatenate(features)
            provenance = signature['scratch_provenance']
        for path, digest in paths:
            if base.file_hash(path) != digest:
                raise ValueError(f'Scratch feature cache changed: {path}')
        for name in ('config.json', 'gene_vocab.tsv', 'gene_medians_train.tsv', 'split_manifest.tsv.gz'):
            path = Path(source['pretraining_manifest']).parent / name
            if base.file_hash(ROOT/path) != provenance[str(path)]:
                raise ValueError(f'Scratch frozen preprocessing changed: {path}')
        if not metadata.index.is_unique or features.shape != (len(metadata), 384):
            raise ValueError('Scratch cache cell count/dimension mismatch')
        counts = {}
        for role, m in groups.items():
            if set(m.index) & set(corpus.cell_id) or set(m.study) & set(corpus.study):
                raise ValueError('Downstream source overlaps Scratch pretraining')
            pbs = m[m.condition.eq('PBS')]
            loc = metadata.index.get_indexer(pbs.index)
            h = np.load(parent/role/'h.npy', mmap_mode='r')
            if (loc < 0).any() or h.shape != (len(pbs), 384) or h.dtype != np.float32 or not np.isfinite(h).all():
                raise ValueError('Scratch native PBS rows/dtype invalid')
            np.testing.assert_array_equal(h, features[loc])
            np.testing.assert_array_equal(metadata.iloc[loc].condition, pbs.condition)
            counts[role] = dict(PBS=len(pbs), sha256=base.file_hash(parent/role/'h.npy'))
        checks[experiment] = dict(checkpoint_sha256=checkpoint_hash, role_features=counts,
            preprocessing='Frozen native all-nonzero CP10K/Train-only medians; FP32, no mask',
            pooling='Valid-gene mean after final encoder LayerNorm', encoder_updates=0)
    return checks


def audit(cfg):
    if cfg['seeds'] != [0, 1, 2] or cfg['fractions'] != [0., .05]:
        raise ValueError('This comparison fixes three seeds and 0/5% joint supervision')
    out = folder(cfg, 'result_directory'); out.mkdir(parents=True, exist_ok=True)
    counts, plan, sources = [], [], {}
    for experiment in cfg['experiments']:
        gcc, source, fc, info = source_inputs(cfg, experiment)
        parent, _, saved = base.ready(source)
        gc.validate_roles(source, parent)
        split_dir, test, combo, n = fixed_split(source, fc, info)
        counts.append(dict(experiment=experiment, dataset=info['dataset'], train_joint_0pct=0,
            train_joint_5pct=n, Test_PBS=int(test.condition.eq('PBS').sum()),
            Test_joint=int(combo.role.eq('test').sum()), unused_joint_pool=int(combo.role.ne('test').sum())-n))
        sources[experiment] = dict(source=base.file_hash(parent / 'complete.json'),
            split=base.file_hash(split_dir / 'combo_manifest.tsv.gz'),
            draws=base.file_hash(split_dir / 'draws.npz'))
        for model in cfg['models']:
            for fraction in cfg['fractions']:
                for seed in cfg['seeds']:
                    old = legacy_run(cfg, experiment, model, seed, fraction)
                    plan.append(dict(experiment=experiment, model=model, fraction=fraction, seed=seed,
                        action='reuse' if old else 'train',
                        reason=('Historical Scratch FiLM code fingerprint differs; use current trainer' if
                                model=='Scratch_v3' and info['dataset']=='GSE136154' and seed==0 else
                                'Matched completed head' if old else 'No compatible completed head'),
                        origin=str(old[0].relative_to(ROOT)) if old else '',
                        result=str(run_dir(cfg, experiment, model, seed, fraction).relative_to(ROOT))))
    if 'Scratch_v3' in cfg['models']:
        base.atomic_json(out / 'scratch_feature_acceptance.json', validate_scratch_features(cfg))
    saved = dict(config=cfg, sources=sources, encoder_inference=False,
        five_percent_design='Independent joint-supervision retraining from matched random initialization',
        selection='Single-stimulus Validation only; shared fixed joint Test')
    path = out / 'audit.json'
    if path.exists() and read_json(path) != saved:
        raise ValueError('Source split or comparison settings changed')
    base.atomic_json(path, saved); base.atomic_json(out / 'config.json', cfg)
    pd.DataFrame(counts).to_csv(out / 'cell_counts.csv', index=False)
    pd.DataFrame(plan).to_csv(out / 'run_plan.csv', index=False)
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print('RUN PLAN', pd.DataFrame(plan).groupby(['fraction', 'action']).size().to_dict(), flush=True)


def prepare(cfg):
    audit(cfg)
    for experiment in cfg['experiments']:
        gcc, source, fc, info = source_inputs(cfg, experiment)
        parent, _, _ = base.ready(source)
        shared = folder(cfg, 'data_directory', experiment+'/features'); shared.mkdir(parents=True, exist_ok=True)
        if (shared / 'complete.json').exists():
            print('PREPARE reused', experiment, flush=True); continue
        stats = dict(np.load(parent / 'statistics.npz'))
        groups = {}
        for role in ('train', 'validation'):
            meta = pd.read_csv(parent / role / 'metadata.tsv.gz', sep='\t', index_col=0)
            y = np.load(parent / role / 'y.npy', mmap_mode='r')
            groups[role] = dict(pbs=np.array(y[meta.condition.eq('PBS').to_numpy()]), meta=meta)
        print('PCA FIT', experiment, groups['train']['pbs'].shape, '->', cfg['pca_dimensions'], flush=True)
        ref, _ = representation.fit_pca(groups['train'], stats, source, cfg)
        np.savez_compressed(shared / 'pca.npz', **ref)
        pd.Series(groups['train']['meta'].index[groups['train']['meta'].condition.eq('PBS')], name='cell_id').to_csv(
            shared / 'pca_fit_cells.tsv.gz', sep='\t', index=False)
        for role, group in groups.items():
            np.save(shared / f'{role}_PCA_h.npy', representation.transform(group['pbs'], ref))
            if info['dataset'] == 'GSE102299':
                original = read_json(ROOT / cfg['legacy_expression_zero'])
                old_rc = old_zero.run_config(original, info['source'], 'ExpressionMLP_FiLM')
                old_data = ROOT / old_rc['data_directory']
                for filename in ('metadata.tsv.gz', 'y.npy'):
                    if (old_data / role / filename).resolve() != (parent / role / filename).resolve():
                        raise ValueError('Historical expression features used other Train/Validation cells')
                gc.link(old_data / role / 'h.npy', shared / f'{role}_ExpressionMLP_h.npy')
            else:
                np.save(shared / f'{role}_ExpressionMLP_h.npy', group['pbs'])
        for model in cfg['models']:
            rc = zero_config(cfg, experiment, model)
            dest = ROOT / rc['data_directory']; dest.mkdir(parents=True, exist_ok=True)
            (ROOT / rc['result_directory']).mkdir(parents=True, exist_ok=True)
            model_stats = dict(stats)
            if model == 'ExpressionMLP':
                model_stats.update(h_mean=stats['control_mean'],
                    h_std=np.maximum(stats['control_std'], source['minimum_expression_scale']))
            elif model == 'PCA':
                model_stats.update(h_mean=ref['h_mean'], h_std=ref['h_std'])
            else:
                origin = feature_origin(cfg, experiment, model)
                original_stats = dict(np.load(origin / 'statistics.npz'))
                model_stats.update(h_mean=original_stats['h_mean'], h_std=original_stats['h_std'])
                for key in stats:
                    if key not in ('h_mean', 'h_std'):
                        np.testing.assert_array_equal(stats[key], original_stats[key])
            for role in groups:
                for filename in ('metadata.tsv.gz', 'y.npy'):
                    gc.link(parent / role / filename, dest / role / filename)
                feature = (shared / f'{role}_{model}_h.npy' if model in ('PCA', 'ExpressionMLP')
                           else origin / role / 'h.npy')
                gc.link(feature, dest / role / 'h.npy')
            for name in ('gene_panel.tsv', 'split_manifest.tsv.gz', 'distribution_reference.npz'):
                gc.link(parent / name, dest / name)
            np.savez_compressed(dest / 'statistics.npz', **model_stats)
            base.atomic_json(dest / 'config.json', rc)
            base.atomic_json(dest / 'complete.json', dict(config=rc,
                source_manifest=base.file_hash(parent / 'complete.json'), pca_sha256=base.file_hash(shared / 'pca.npz'),
                model=model, train_combination_cells=0, statistics_fit='Original single-only Train; h uses Train PBS only'))
        base.atomic_json(shared / 'complete.json', dict(config=cfg,
            source_manifest=base.file_hash(parent / 'complete.json'), pca_sha256=base.file_hash(shared / 'pca.npz')))


def check(cfg):
    audit(cfg); rows = []
    for experiment in cfg['experiments']:
        _, source, parent_few, info = source_inputs(cfg, experiment)
        parent, _, _ = base.ready(source)
        shared = folder(cfg, 'data_directory', experiment+'/features')
        reference = dict(np.load(shared / 'pca.npz'))
        meta = pd.read_csv(parent / 'train/metadata.tsv.gz', sep='\t', index_col=0)
        fit = pd.read_csv(shared / 'pca_fit_cells.tsv.gz', sep='\t').cell_id
        np.testing.assert_array_equal(fit, meta.index[meta.condition.eq('PBS')])
        for model in cfg['models']:
            rc = zero_config(cfg, experiment, model); dest, _, saved = base.ready(rc)
            if saved['pca_sha256'] != base.file_hash(shared / 'pca.npz'):
                raise ValueError('PCA reference changed')
            current = dict(np.load(dest / 'statistics.npz')); original = dict(np.load(parent / 'statistics.npz'))
            for key in original:
                if key not in ('h_mean', 'h_std'):
                    np.testing.assert_array_equal(current[key], original[key])
            for role in ('train', 'validation'):
                m = pd.read_csv(dest / role / 'metadata.tsv.gz', sep='\t', index_col=0)
                if set(m.condition) != {'PBS', *base.training_conditions(source)}:
                    raise ValueError('Joint supervision leaked into zero-shot/Validation')
                for name in ('metadata.tsv.gz', 'y.npy'):
                    if (dest / role / name).resolve() != (parent / role / name).resolve():
                        raise ValueError('Source expression/split changed')
                h = np.load(dest / role / 'h.npy', mmap_mode='r')
                dim = (len(pd.read_csv(dest / 'gene_panel.tsv', sep='\t')) if model == 'ExpressionMLP'
                       else 384 if model == 'Scratch_v3' else cfg['pca_dimensions'])
                if h.shape != (int(m.condition.eq('PBS').sum()), dim) or not np.isfinite(h).all():
                    raise ValueError('Feature/PBS alignment or dimension differs')
            genes = pd.read_csv(dest / 'gene_panel.tsv', sep='\t')
            torch.manual_seed(0)
            head = base.make_model(genes, current, rc, torch.device('cpu'))
            rows.append(dict(experiment=experiment, model=model, input_dimensions=dim,
                             parameters=sum(p.numel() for p in head.parameters()), Train_PBS=len(fit)))
            for fraction in cfg['fractions']:
                for seed in cfg['seeds']:
                    old = legacy_run(cfg, experiment, model, seed, fraction)
                    if old:
                        validate_legacy(head, old[0])
    base.atomic_json(folder(cfg, 'result_directory') / 'input_acceptance.json',
                     dict(passed=True, checks=rows, parameter_updates=0, heldout_expression_read=False))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def validate_legacy(model, origin):
    weights = torch.load(origin / 'best.pt', map_location='cpu', weights_only=False)['model']
    for name in ('h_mean', 'h_std'):
        torch.testing.assert_close(model.state_dict()[name], weights[name], rtol=0, atol=0)
    if model.get_extra_state() != weights['_extra_state']:
        raise ValueError('Historical head has different gene/stimulus semantics')
    model.load_state_dict(weights)


def reuse_head(cfg, experiment, model, seed, fraction):
    old = legacy_run(cfg, experiment, model, seed, fraction)
    if old is None:
        return False
    origin, done = old
    rc = zero_config(cfg, experiment, model); data, _, _ = base.ready(rc)
    genes = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
    stats = dict(np.load(data / 'statistics.npz'))
    torch.manual_seed(seed)
    head = base.make_model(genes, stats, rc, torch.device('cpu')); validate_legacy(head, origin)
    if fraction == 0:
        signature = base.protocol(rc)
    else:
        fc = prepare_five(cfg, experiment, model, seed)
        signature = few.protocol(fc)
    dest = run_dir(cfg, experiment, model, seed, fraction); dest.mkdir(parents=True, exist_ok=True)
    gc.link(origin / 'best.pt', dest / 'best.pt'); gc.link(origin / 'history.csv', dest / 'history.csv')
    new = dict(done, protocol=signature, reused_from=str(origin.relative_to(ROOT)),
        origin_complete_sha256=base.file_hash(origin / 'complete.json'),
        parameters=sum(p.numel() for p in head.parameters()))
    if (dest / 'complete.json').exists() and read_json(dest / 'complete.json') != new:
        raise ValueError('Reused head registration changed')
    base.atomic_json(dest / 'complete.json', new)
    print('REUSE', experiment, model, seed, fraction, flush=True)
    return True


def prepare_five(cfg, experiment, model, seed):
    fc = five_config(cfg, experiment, model, seed)
    data = ROOT / fc['data_directory']; data.mkdir(parents=True, exist_ok=True)
    (ROOT / fc['result_directory']).mkdir(parents=True, exist_ok=True)
    _, source, old, info = source_inputs(cfg, experiment)
    old_data, _, _, _ = fixed_split(source, old, info)
    rc = zero_config(cfg, experiment, model); parent, out, _ = base.ready(rc)
    done = read_json(out / f'seed{seed}/complete.json')
    if done['protocol'] != base.protocol(rc) or done['best_sha256'] != base.file_hash(out / f'seed{seed}/best.pt'):
        raise ValueError('Finish and lock zero-shot before five-percent joint training')
    for name in ('combo_manifest.tsv.gz', 'indices.npz', 'draws.npz', 'train_combo.npy'):
        gc.link(old_data / name, data / info['source'] / name)
    saved = dict(config=fc, signatures={info['source']: dict(
        source={name: base.file_hash(parent / name) for name in
                ('complete.json', 'split_manifest.tsv.gz', 'statistics.npz', 'distribution_reference.npz')},
        zero_sha256=done['best_sha256'], split_sha256=base.file_hash(old_data / 'combo_manifest.tsv.gz'),
        train_combo_sha256=base.file_hash(old_data / 'train_combo.npy'))})
    if (data / 'complete.json').exists() and read_json(data / 'complete.json') != saved:
        raise ValueError('Few-shot training source changed')
    base.atomic_json(data / 'complete.json', saved)
    return fc


def train(cfg, device):
    for experiment in cfg['experiments']:
        for model in cfg['models']:
            rc = zero_config(cfg, experiment, model)
            for seed in cfg['seeds']:
                if not reuse_head(cfg, experiment, model, seed, 0):
                    print('TRAIN zero-shot', experiment, model, seed, flush=True)
                    base.train_seed(rc, seed, device)
    for experiment in cfg['experiments']:
        for model in cfg['models']:
            for seed in cfg['seeds']:
                if not reuse_head(cfg, experiment, model, seed, .05):
                    fc = prepare_five(cfg, experiment, model, seed)
                    print('TRAIN 5% joint supervision', experiment, model, seed, flush=True)
                    few.train_one(fc, cfg['experiments'][experiment]['source'], .05, device)


def lock_models(cfg):
    locked = {}
    for experiment in cfg['experiments']:
        for model in cfg['models']:
            for fraction in cfg['fractions']:
                for seed in cfg['seeds']:
                    dest = run_dir(cfg, experiment, model, seed, fraction)
                    done = read_json(dest / 'complete.json')
                    expected = (base.protocol(zero_config(cfg, experiment, model)) if fraction == 0
                                else few.protocol(five_config(cfg, experiment, model, seed)))
                    if done['protocol'] != expected or done['best_sha256'] != base.file_hash(dest / 'best.pt'):
                        raise ValueError(f'Complete and lock all heads before Test: {dest}')
                    locked[f'{experiment}/{model}/{fraction}/{seed}'] = done
    return locked


def represented_test(cfg, experiment, model, group):
    gcc, _, _, _ = source_inputs(cfg, experiment)
    if model == 'ExpressionMLP':
        return dict(group, h=group['pbs'])
    if model == 'PCA':
        ref = dict(np.load(folder(cfg, 'data_directory', experiment+'/features/pca.npz')))
        return dict(group, h=representation.transform(group['pbs'], ref))
    original = feature_origin(cfg, experiment, model)
    return dict(group, h=np.load(original / 'test/h.npy'))


def prediction_scores(pred, truth, control, genes, reference, draws, source):
    rows, moments, panels, observed, predicted = base.population_scores(pred, truth, control, reference['functional_indices'])
    distribution = []
    for i, seed in enumerate(source['distribution_seeds']):
        pi, ti = draws['predicted_indices'][i], draws['target_indices'][i]
        values = base.distribution_scores(pred[pi], truth[ti], reference, panels, source['kl_alpha'])
        distribution.extend(dict(sample_seed=seed, **r) for r in values)
    dist = pd.DataFrame(distribution)
    average = dist.drop(columns='sample_seed').groupby('panel').mean(numeric_only=True).reset_index()
    scores = pd.DataFrame(rows).merge(average, on='panel', how='left')
    gene_table = pd.concat([genes[['gene_id','gene_symbol']], pd.DataFrame(moments)], axis=1)
    return scores, dist, gene_table


def legacy_expression_scores(cfg, experiment, model, seed, fraction, genes, truth, control):
    """Historical Expression/Scratch benchmarks used this exact 80% Holdout."""
    info = cfg['experiments'][experiment]
    if model not in ('ExpressionMLP', 'Scratch_v3') or info['dataset'] != 'GSE102299':
        return None
    old = read_json(ROOT / cfg['legacy_expression_zero' if fraction == 0 else 'legacy_expression_five'])
    root = ROOT / old['result_directory']; rep = info['source']
    run, _ = legacy_run(cfg, experiment, model, seed, fraction)
    table = pd.read_csv(run / 'test/gene_predictions.tsv.gz', sep='\t')
    np.testing.assert_array_equal(table.gene_id, genes.gene_id)
    np.testing.assert_allclose(table.observed_mean, truth.mean(0,dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(table.control_mean, control.mean(0,dtype=float), rtol=1e-12, atol=1e-12)
    values = pd.read_csv(root / 'test_metrics.csv', dtype={'replicate': str})
    dist = pd.read_csv(root / 'distribution_draws.csv', dtype={'replicate': str})
    select = lambda frame: frame[frame.replicate.eq(rep) & frame.method.eq(model+'_FiLM') & frame.seed.eq(seed)].drop(
        columns=['replicate','method','seed']).reset_index(drop=True)
    scores, draws = select(values), select(dist)
    if len(scores) != 6 or draws.sample_seed.nunique() != 5:
        raise ValueError('Historical expression evaluation is incomplete')
    return scores, draws, table


def macro_average(scores):
    """Two independent sources per dataset, equal weight within each training seed."""
    keys = ['dataset', 'fraction', 'model', 'seed', 'panel']
    if (scores.groupby(keys).experiment.nunique() != 2).any() or scores.duplicated(keys+['experiment']).any():
        raise ValueError('A macro score requires exactly one result from each of two sources')
    numeric = scores.select_dtypes(include=np.number).columns.difference(['fraction','seed','n_genes'])
    result = scores.groupby(keys, sort=False)[list(numeric)].agg(
        lambda values: np.mean(values.to_numpy(dtype=float))).reset_index()
    result['RMSE_delta'] = np.sqrt(result.MSE_delta)
    result['source_count'] = 2
    return result


def evaluate(cfg, device):
    locked = lock_models(cfg)
    out = folder(cfg, 'result_directory'); path = out / 'test_protocol.json'
    if path.exists() and read_json(path) != locked:
        raise ValueError('Selected checkpoints changed after Test scoring')
    base.atomic_json(path, locked)
    all_scores = []
    for experiment in cfg['experiments']:
        _, source, fc, info = source_inputs(cfg, experiment)
        parent, _, _ = base.ready(source)
        split_dir, _, _, n_joint = fixed_split(source, fc, info)
        group = few.load_test(fc, info['source'])
        x, truth = group['pbs'], group['targets'][few.combination(fc)]
        genes = pd.read_csv(parent / 'gene_panel.tsv', sep='\t')
        stats = dict(np.load(parent / 'statistics.npz'))
        reference = dict(np.load(parent / 'distribution_reference.npz'))
        draws = dict(np.load(split_dir / 'draws.npz'))
        joint = np.load(split_dir / 'train_combo.npy', mmap_mode='r')[:n_joint]
        baseline_shifts = dict(mean_delta_additive=(stats['target_mean']-stats['control_mean']).sum(0),
                               fewshot_mean_delta=joint.mean(0,dtype=float)-stats['control_mean'])
        for fraction in cfg['fractions']:
            methods = [('control_copy', -1), ('mean_delta_additive', -1)]
            if fraction > 0:
                methods += [('fewshot_mean_delta', -1), ('fewshot_target_mean', -1)]
            methods += [(m, s) for m in cfg['models'] for s in cfg['seeds']]
            for model, seed in methods:
                dest = out / experiment / 'test' / few.label(fraction) / model / (f'seed{seed}' if seed >= 0 else 'fixed')
                dest.mkdir(parents=True, exist_ok=True)
                token = dict(split_sha256=base.file_hash(split_dir / 'combo_manifest.tsv.gz'),
                    draws_sha256=base.file_hash(split_dir / 'draws.npz'),
                    reference_sha256=base.file_hash(parent / 'distribution_reference.npz'),
                    head_sha256=locked[f'{experiment}/{model}/{fraction}/{seed}']['best_sha256'] if seed >= 0 else None)
                marker = dest / 'evaluation_complete.json'
                if marker.exists():
                    if read_json(marker) != token:
                        raise ValueError('Scored Test inputs changed')
                    values = pd.read_csv(dest / 'test_metrics.csv')
                    print('TEST reused', experiment, fraction, model, seed, flush=True)
                else:
                    reused = legacy_expression_scores(cfg,experiment,model,seed,fraction,genes,truth,x)
                    if reused is not None:
                        values, dist, gene_table = reused
                        print('TEST historical benchmark metrics reused', experiment, fraction, model, seed, flush=True)
                    else:
                        if model == 'control_copy':
                            pred = x
                        elif model in baseline_shifts:
                            pred = np.maximum(x+baseline_shifts[model],0).astype(np.float32)
                        elif model == 'fewshot_target_mean':
                            pred = np.broadcast_to(joint.mean(0,dtype=float).astype(np.float32), x.shape)
                        else:
                            rc = zero_config(cfg, experiment, model)
                            model_stats = dict(np.load(ROOT / rc['data_directory'] / 'statistics.npz'))
                            head = base.make_model(genes, model_stats, rc, device)
                            state = torch.load(run_dir(cfg, experiment, model, seed, fraction) / 'best.pt', map_location=device, weights_only=False)
                            head.load_state_dict(state['model'])
                            inputs = represented_test(cfg, experiment, model, group)
                            pred = base.predict(head, inputs, few.combination(fc), rc, device)
                            del head
                        values, dist, gene_table = prediction_scores(pred,truth,x,genes,reference,draws,source)
                    values.to_csv(dest / 'test_metrics.csv', index=False)
                    dist.to_csv(dest / 'distribution_draws.csv', index=False)
                    gene_table.to_csv(dest / 'gene_predictions.tsv.gz', sep='\t', index=False)
                    base.atomic_json(marker, token)
                    print('TEST', experiment, fraction, model, seed, 'MSE', values.iloc[0].MSE_delta, flush=True)
                all_scores.append(values.assign(experiment=experiment, dataset=info['dataset'],
                                               fraction=fraction, model=model, seed=seed))
                pd.concat(all_scores, ignore_index=True).to_csv(out / 'metrics_by_source_seed.csv', index=False)
    source_scores = pd.concat(all_scores, ignore_index=True)
    averaged = macro_average(source_scores)
    averaged.to_csv(out / 'source_macro_by_seed.csv', index=False)
    fields = averaged.select_dtypes(include=np.number).columns.difference(['fraction','seed','source_count'])
    averaged.groupby(['dataset','fraction','model','panel'])[list(fields)].agg(['mean','std']).to_csv(out / 'source_macro_summary.csv')
    base.atomic_json(out / 'evaluation_complete.json', dict(seeds=cfg['seeds'], locked_checkpoints=locked,
        same_Test_for_zero_and_five=True, source_weight=.5, Test_used_for_selection=False,
        five_percent_design='Matched random initialization, singles+5% joint retraining',
        source_aggregation='Mean MSE then sqrt per seed; arithmetic mean PCC/R2/overlap; no cell pooling'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit','prepare','check','train','evaluate','plot','all'])
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = p.parse_args(); cfg = read_json(args.config)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    with threadpool_limits(cfg['cpu_threads']):
        if args.stage == 'audit':
            audit(cfg)
        if args.stage in ('prepare','all'):
            prepare(cfg)
        if args.stage in ('check','all'):
            check(cfg)
        if args.stage in ('train','all'):
            train(cfg, device)
        if args.stage in ('evaluate','all'):
            evaluate(cfg, device)
        if args.stage in ('plot','all'):
            subprocess.run([sys.executable, str(ROOT/'main/plot_combination_representation_3seeds.py'),
                            '--config', str(args.config)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
