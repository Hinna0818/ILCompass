"""Raw/LoRA GeneCompass: reuse the four original within-source combination tasks.

Existing PBS embeddings are reused only after checking checkpoint, cell identity,
and the exact historical counts/median Top2048 inputs. No FM extraction/training
is needed. Original expression assets, splits, FiLM trainer and metrics are shared.
"""
import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_film_multistimulus as base
from genecompass_coverage import map_genes
from posttrain_data import encode

ROOT = base.ROOT
CONFIG = ROOT / 'configs/genecompass_combination_zero_shot_v1.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def source_config(cfg, experiment):
    return read_json(ROOT / cfg['experiments'][experiment])


def run_config(cfg, experiment, model):
    # All optimization and evaluation settings come from the original task.
    return dict(source_config(cfg, experiment),
        data_directory=str(Path(cfg['data_directory']) / experiment / model),
        result_directory=str(Path(cfg['result_directory']) / experiment / model),
        checkpoint=cfg['models'][model],
        feature_directory=cfg['raw_cache'] if model == 'GeneCompass' else cfg['lora_cache'],
        workflow_script=str(Path(__file__).relative_to(ROOT)))


def link(source, target):
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError(f'Destination is not the expected historical asset: {target}')
    else:
        target.symlink_to(source)


def role_metadata(folder, role):
    return pd.read_csv(folder / role / 'metadata.tsv.gz', sep='\t', index_col=0)


def validate_roles(source, folder):
    groups = {role: role_metadata(folder, role) for role in ('train', 'validation', 'test')}
    for role, meta in groups.items():
        label = meta.author_replicate.astype(str) if source.get('within_replicate') else meta.author_batch
        expected = source.get('within_replicate') or source['within_batch']
        conditions = base.evaluation_conditions(source) if role == 'test' else base.training_conditions(source)
        if (not meta.index.is_unique or not meta.split.eq(role).all()
                or not label.eq(expected).all() or set(meta.condition) != {'PBS', *conditions}):
            raise ValueError(f'Wrong source, role or condition boundary: {folder}/{role}')
    all_ids = pd.Index(np.concatenate([m.index.to_numpy() for m in groups.values()]))
    if not all_ids.is_unique:
        raise ValueError('Train/Validation/Test overlap')
    return groups


def audit_inputs(cfg, experiments):
    """Read PBS counts only; never load held-out combination expression here."""
    raw, lora = ROOT / cfg['raw_cache'], ROOT / cfg['lora_cache']
    signature = read_json(raw / 'signature.json')
    encoded = read_json(raw / 'encoded.json')
    complete = read_json(lora / 'complete.json')
    checkpoint_hashes = {m: base.file_hash(ROOT / p) for m, p in cfg['models'].items()}
    if (signature['checkpoints']['GeneCompass'] != checkpoint_hashes['GeneCompass']
            or complete['signature']['checkpoint']['sha256'] != checkpoint_hashes['GeneCompass_LoRA']
            or 'GeneCompass' not in encoded['models'] or not encoded['frozen']
            or signature['encoding'] != 'median' or signature['max_gene_tokens'] != 2048
            or signature['pooling'] != 'masked_gene_mean' or signature['precision'] != 'float32'):
        raise ValueError('Cached encoder checkpoint or input/pooling protocol differs')
    prep = read_json(ROOT / 'configs/2prepare_traindata.json')
    lc = read_json(ROOT / cfg['lora_config'])
    if (ROOT / lc['checkpoint'] / 'pytorch_model.bin').resolve() != (ROOT / cfg['models']['GeneCompass']).resolve():
        raise ValueError('LoRA base differs from raw GeneCompass')
    for name, path in [('tokens', prep['token_dictionary']), ('medians', prep['gene_median_reference'])]:
        if base.file_hash(ROOT / path) != signature[name]:
            raise ValueError(f'GeneCompass {name} changed')
    for name in ('PBS_inputs.npz', 'PBS_metadata.tsv.gz'):
        path = raw / name
        key = str(path.relative_to(ROOT))
        if base.file_hash(path) != complete['signature']['input_sha256'][key]:
            raise ValueError(f'Raw/LoRA cached input mismatch: {path}')
    features = {'GeneCompass': raw / 'GeneCompass/PBS_h.npy',
                'GeneCompass_LoRA': lora / 'lung_h.npy'}
    if base.file_hash(features['GeneCompass_LoRA']) != complete['features']['lung']:
        raise ValueError('LoRA frozen feature file changed')
    meta = pd.read_csv(raw / 'PBS_metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
    arrays = {m: np.load(p, mmap_mode='r') for m, p in features.items()}
    if (not meta.index.is_unique or not meta.condition.eq('PBS').all()
            or any(h.shape != (len(meta), 768) or h.dtype != np.float32 or not np.isfinite(h).all()
                   for h in arrays.values())):
        raise ValueError('Invalid frozen PBS embedding shape, cell IDs or values')
    tokens = pickle.loads((ROOT / prep['token_dictionary']).read_bytes())
    medians = pickle.loads((ROOT / prep['gene_median_reference']).read_bytes())
    corpus = pd.read_csv(ROOT / lc['split_manifest'], sep='\t', usecols=['cell_id', 'study'])
    provenance, counts = {}, []
    with np.load(raw / 'PBS_inputs.npz') as archive:
        # npz indexing decompresses a whole array: load once, then slice in memory.
        cached = {key: archive[key] for key in ('ids', 'vals', 'length')}
        # Both sources share one study-level PBS cache; avoid reading each study twice.
        for source_path in dict.fromkeys(source_config(cfg, e)['source'] for e in experiments):
            selected = [e for e in experiments if source_config(cfg, e)['source'] == source_path]
            pbs_ids = []
            for e in selected:
                source = source_config(cfg, e)
                folder, _, saved = base.ready(source)
                if base.file_hash(ROOT / source_path) != saved['source_sha256']:
                    raise ValueError('Historical source counts changed')
                groups = validate_roles(source, folder)
                pbs_ids.extend(m.index[m.condition.eq('PBS')] for m in groups.values())
                for role, m in groups.items():
                    counts.extend(dict(experiment=e, split=role, condition=c, cells=int(n))
                                  for c, n in m.condition.value_counts().items())
                provenance[e] = dict(parent=str(folder.relative_to(ROOT)),
                                     manifest_sha256=base.file_hash(folder / 'complete.json'))
            ids = pd.Index(np.concatenate([v.to_numpy() for v in pbs_ids]))
            rows = meta.index.get_indexer(ids)
            if not ids.is_unique or (rows < 0).any():
                raise ValueError('Historical PBS cells missing or repeated in frozen cache')
            backed = ad.read_h5ad(ROOT / source_path, backed='r')
            a = backed[ids].to_memory(); backed.file.close()
            if (not a.var_names.is_unique or not a.obs_names.equals(ids)
                    or not a.obs.qc_version.eq('ilc2_balanced_v1').all()
                    or not a.obs.selected_ILC2.all()):
                raise ValueError('Historical PBS gene IDs, QC or cell identities differ')
            x = base._raw_counts(a.layers['counts'])
            if (a.X != x).nnz or set(ids) & set(corpus.cell_id) or set(a.obs.study) & set(corpus.study):
                raise ValueError('PBS raw counts mismatch or LoRA pretraining overlap')
            cols, med, _ = map_genes(a.var_names, tokens, medians)
            for left in range(0, len(a), 128):
                inputs = encode(x[left:left + 128], cols, med, 2048, 'median')
                loc = rows[left:left + 128]
                for key, values in zip(('ids', 'vals', 'length'), inputs):
                    np.testing.assert_array_equal(values, cached[key][loc])
            print('AUDIT exact PBS counts/median Top2048 matched:', source_path, len(a), flush=True)
            del a, x
    return meta, arrays, dict(checkpoints=checkpoint_hashes, experiments=provenance,
        feature_sha256={m: base.file_hash(p) for m, p in features.items()},
        input_sha256=base.file_hash(raw / 'PBS_inputs.npz'),
        representation='FP32 last-layer valid-gene mean; exclude species CLS/PAD; no mask',
        input='counts/gene median -> stable descending Top2048 -> log2(1+x)',
        ILC2_LoRA_corpus_overlap=0, encoder_updates=0), pd.DataFrame(counts)


def prepare(cfg, experiments):
    meta, features, provenance, counts = audit_inputs(cfg, experiments)
    out = ROOT / cfg['result_directory']; out.mkdir(parents=True, exist_ok=True)
    for e in experiments:
        source = source_config(cfg, e); parent, _, _ = base.ready(source)
        groups = validate_roles(source, parent)
        for model, array in features.items():
            rc = run_config(cfg, e, model)
            data, dest = ROOT / rc['data_directory'], ROOT / rc['result_directory']
            data.mkdir(parents=True, exist_ok=True); dest.mkdir(parents=True, exist_ok=True)
            evidence = dict(provenance, experiments={e: provenance['experiments'][e]})
            if (data / 'complete.json').exists():
                _, _, saved = base.ready(rc)
                if saved['frozen_features'] != evidence:
                    raise ValueError('Frozen features or historical source changed')
                print('PREPARE reused', e, model, flush=True); continue
            for name in ('split_manifest.tsv.gz', 'gene_panel.tsv', 'distribution_reference.npz'):
                link(parent / name, data / name)
            for role, m in groups.items():
                d = data / role; d.mkdir(exist_ok=True)
                for name in ('metadata.tsv.gz', 'y.npy'):
                    link(parent / role / name, d / name)
                for path in (parent / role).glob('*_draws.npz'):
                    link(path, d / path.name)
                ids = m.index[m.condition.eq('PBS')]
                np.save(d / 'h.npy', array[meta.index.get_indexer(ids)])
            stats = dict(np.load(parent / 'statistics.npz'))
            h = np.load(data / 'train/h.npy')
            stats.update(h_mean=h.mean(0), h_std=h.std(0))
            np.savez_compressed(data / 'statistics.npz', **stats)
            base.atomic_json(data / 'complete.json', dict(config=rc, frozen_features=evidence,
                training_conditions=list(base.training_conditions(rc)), combination_train_cells=0,
                statistics_fit='Only embedding mean/std replaced, fitted on original Train PBS',
                feature_sha256={role: base.file_hash(data / role / 'h.npy') for role in groups}))
            base.atomic_json(dest / 'config.json', rc)
    # Per-experiment files support preparing one task now and the other tasks later.
    for e in experiments:
        d = out / e; d.mkdir(exist_ok=True)
        counts[counts.experiment.eq(e)].to_csv(d / 'cell_counts.csv', index=False)
    base.atomic_json(out / 'config.json', cfg)
    print(counts.to_string(index=False), flush=True)


def check(cfg, experiments):
    checks = []
    for e in experiments:
        source = source_config(cfg, e); parent, _, _ = base.ready(source)
        groups = validate_roles(source, parent)
        for model in cfg['models']:
            rc = run_config(cfg, e, model); data, _, saved = base.ready(rc)
            original = dict(np.load(parent / 'statistics.npz'))
            stats = dict(np.load(data / 'statistics.npz'))
            for key in original:
                if key not in ('h_mean', 'h_std'):
                    np.testing.assert_array_equal(stats[key], original[key])
            for role, m in groups.items():
                for name in ('y.npy', 'metadata.tsv.gz'):
                    if (data / role / name).resolve() != (parent / role / name).resolve():
                        raise ValueError('Expression or fixed split not linked to original')
                if base.file_hash(data / role / 'h.npy') != saved['feature_sha256'][role]:
                    raise ValueError('Prepared PBS feature rows changed')
                h = np.load(data / role / 'h.npy')
                if h.shape != (int(m.condition.eq('PBS').sum()), 768) or not np.isfinite(h).all():
                    raise ValueError('Prepared PBS feature alignment differs')
                for p in (parent / role).glob('*_draws.npz'):
                    if (data / role / p.name).resolve() != p.resolve():
                        raise ValueError('Evaluation samples changed')
            h = np.load(data / 'train/h.npy')
            np.testing.assert_array_equal(stats['h_mean'], h.mean(0))
            np.testing.assert_array_equal(stats['h_std'], h.std(0))
            if (data / 'distribution_reference.npz').resolve() != (parent / 'distribution_reference.npz').resolve():
                raise ValueError('PCA/KL metric reference differs')
            genes = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
            torch.manual_seed(source['seeds'][0])
            head = base.make_model(genes, stats, rc, torch.device('cpu'))
            checks.append(dict(experiment=e, model=model, seeds=source['seeds'], genes=len(genes),
                FM_dimensions=768, parameters=sum(p.numel() for p in head.parameters()),
                combination_train_cells=0, original_expression_splits_draws=True))
        base.atomic_json(ROOT / cfg['result_directory'] / e / 'input_acceptance.json',
                         dict(passed=True, checks=checks[-len(cfg['models']):], parameter_updates=0,
                              heldout_combination_expression_read=False))
    print(pd.DataFrame(checks).to_string(index=False), flush=True)


def summarize(cfg, experiments):
    frames = []
    for e in experiments:
        for model in cfg['models']:
            rc = run_config(cfg, e, model)
            scores = pd.read_csv(ROOT / rc['result_directory'] / 'test_metrics.csv')
            scores.insert(0, 'encoder', model)
            scores.insert(0, 'experiment', e)
            frames.append(scores)
        source = source_config(cfg, e)
        old = pd.read_csv(ROOT / source['result_directory'] / 'test_metrics.csv')
        old = old[old.method.isin(['FiLM', 'FiLM_additive']) & old.seed.isin(source['seeds'])].copy()
        old.insert(0, 'encoder', 'Scratch_v3_historical'); old.insert(0, 'experiment', e)
        frames.append(old)
    out = ROOT / cfg['result_directory']
    scores = pd.concat(frames, ignore_index=True)
    # Identical expression-only baselines are listed once, not once per encoder.
    baseline = scores.method.isin(['control_copy', 'mean_delta_additive'])
    scores.loc[baseline, 'encoder'] = 'Expression_baseline'
    scores = scores.drop_duplicates(['experiment', 'encoder', 'condition', 'method', 'seed', 'panel'])
    scores.to_csv(out / 'comparison_metrics.csv', index=False)
    scores[scores.panel.isin(['All_genes', 'Top100'])].to_csv(out / 'comparison_all_top100.csv', index=False)
    print('COMPARISON', out / 'comparison_all_top100.csv', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'check', 'train', 'evaluate', 'summarize', 'all'])
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--experiment', default='all', choices=['all', 'GSE136154_C', 'GSE136154_D',
                                                        'GSE102299_rep1', 'GSE102299_rep2'])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args(); cfg = read_json(args.config)
    experiments = list(cfg['experiments']) if args.experiment == 'all' else [args.experiment]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    with threadpool_limits(4):
        if args.command in ('prepare', 'all'):
            prepare(cfg, experiments)
        if args.command in ('check', 'train', 'evaluate', 'all'):
            check(cfg, experiments)
        if args.command in ('train', 'all'):
            for e in experiments:
                for model in cfg['models']:
                    rc = run_config(cfg, e, model)
                    print('TRAIN', e, model, flush=True)
                    for seed in rc['seeds']:
                        base.train_seed(rc, seed, device)
        if args.command in ('evaluate', 'all'):
            # Finish every selected head before accessing combination Test expression.
            for e in experiments:
                for model in cfg['models']:
                    rc = run_config(cfg, e, model)
                    for seed in rc['seeds']:
                        d = ROOT / rc['result_directory'] / f'seed{seed}'
                        saved = read_json(d / 'complete.json')
                        if (saved['combination_used_for_training_or_selection']
                                or saved['protocol'] != base.protocol(rc)
                                or saved['best_sha256'] != base.file_hash(d / 'best.pt')):
                            raise ValueError('Finish and lock all selected single-only heads first')
            for e in experiments:
                for model in cfg['models']:
                    print('EVALUATE', e, model, flush=True)
                    base.evaluate_models(run_config(cfg, e, model), device)
        if args.command in ('summarize', 'evaluate', 'all'):
            summarize(cfg, experiments)


if __name__ == '__main__':
    main()
