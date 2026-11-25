"""Expression MLP versus locked frozen-v3 FiLM on exactly the same few-shot task.

Only the cell representation changes: Train-PBS-standardized full expression
replaces the frozen FM embedding. Stimulus embeddings, FiLM, decoder and losses
are shared. The wider input increases trainable parameters; this is a practical
pipeline comparison, not an isolated test of pretraining weights.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import train_film_combination_fewshot as few

base = few.base
ROOT = base.ROOT
CONFIG = ROOT / 'configs/expression_film_GSE102299_rep1_05pct_v1.json'
METHOD = 'ExpressionMLP_FiLM'


def inputs(cfg):
    fc = json.loads((ROOT / cfg['fewshot_config']).read_text())
    source = few.sources(fc)[cfg['replicate']]
    data, _, _ = base.ready(source)
    genes = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
    with np.load(data / 'statistics.npz') as f:
        stats = dict(f)
    return fc, source, genes, stats


def protocol(cfg):
    fc, _, _, _ = inputs(cfg)
    parent = few.protocol(fc)
    ref = ROOT / fc['result_directory'] / cfg['replicate'] / few.label(cfg['fraction'])
    done = json.loads((ref / 'complete.json').read_text())
    if done['protocol'] != parent or base.file_hash(ref / 'best.pt') != done['best_sha256']:
        raise ValueError('Frozen FM comparison protocol/checkpoint changed')
    return dict(config=cfg, parent=parent, fm_best_sha256=done['best_sha256'],
        code_sha256=base.file_hash(Path(__file__)),
        representation='Train PBS standardized full log1p(CP10K); no FM features used')


def expression_group(group):
    # Replace cached FM features before tensor creation or any model invocation.
    return dict(group, h=group['pbs'])


def make_model(genes, stats, source, device):
    return base.FiLMPerturbationModel(genes.gene_id.tolist(), stats['control_mean'],
        np.maximum(stats['control_std'], source['minimum_expression_scale']), **source['model']).to(device)


def train(cfg, device):
    out = ROOT / cfg['result_directory']; out.mkdir(parents=True, exist_ok=True)
    signature = protocol(cfg)
    with (out / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'complete.json').exists():
            done = json.loads((out / 'complete.json').read_text())
            if done['protocol'] != signature or done['best_sha256'] != base.file_hash(out / 'best.pt'):
                raise ValueError('Completed expression run changed')
            print('TRAIN expression completed, reused', flush=True); return
        fc, source, genes, stats = inputs(cfg)
        tr = expression_group(few.train_group(fc, cfg['replicate'], cfg['fraction']))
        va = expression_group(base.load_role(source, 'validation'))
        torch.manual_seed(fc['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(fc['seed'])
        model = make_model(genes, stats, source, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=source['learning_rate'], weight_decay=source['weight_decay'])
        tensor = base.as_tensors(tr, device); scale = torch.tensor(stats['scale'], device=device)
        rng = np.random.default_rng(fc['seed'])
        history, best, stale, start = [], float('inf'), 0, 1
        if (out / 'last.pt').exists():
            last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
            if last['protocol'] != signature or last['device'] != str(device):
                raise ValueError('Resume protocol/device changed')
            model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
            rng.bit_generator.state = last['numpy_rng']; torch.set_rng_state(last['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(last['cuda_rng'])
            history, best, stale, start = last['history'], last['best'], last['stale'], last['epoch'] + 1
        base.atomic_json(out / 'config.json', dict(config=cfg, source_training_config=source,
            architecture='G -> 256 -> 128 MLP; unchanged 32D stimulus embeddings/FiLM/128D decoder -> G',
            input_mean='Original Train PBS mean', input_std_floor=source['minimum_expression_scale'],
            trainable_parameters=sum(p.numel() for p in model.parameters()), seed=fc['seed']))
        clock = time.monotonic()
        for epoch in range(start, source['epochs'] + 1):
            if epoch > source['minimum_epochs'] and stale >= source['patience']:
                break
            batches = base.epoch_batches(tr['targets'], source, rng, conditions=tuple(tr['targets']))
            model.train(); totals = np.zeros(4)
            for condition, ids, weight in batches:
                control = rng.choice(len(tr['pbs']), len(ids), replace=True)
                x = tensor['pbs'][control]
                c = x.new_tensor(base.condition_vectors(source)[condition]).expand(len(x), -1)
                optimizer.zero_grad(set_to_none=True)
                terms = base.loss_terms(model(x, x, c), tensor['targets'][condition][ids], scale, source)
                if not torch.isfinite(terms[0]):
                    raise ValueError('Nonfinite expression objective')
                (terms[0] * weight * len(batches)).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), source['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([v.detach().item() for v in terms]) * weight
            values = base.validation_scores(model, va, source, device)
            score = float(np.mean(list(values.values())))
            if score < best:
                best, stale = score, 0
                base.atomic_torch(out / 'best.pt', dict(model=model.state_dict(), epoch=epoch, protocol=signature))
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_loss=totals[0], train_mean=totals[1],
                train_variance=totals[2], train_shift=totals[3], val_macro_MSE_delta=score,
                best=best, stale=stale, grad_norm=float(norm), **{f'val_{c}_MSE_delta': v for c, v in values.items()}))
            base.atomic_torch(out / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, protocol=signature, device=str(device), history=history, best=best, stale=stale,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(f'{METHOD} epoch={epoch} train={totals[0]:.6g} val={score:.6g} '
                  f'best={best:.6g} stale={stale} elapsed={time.monotonic()-clock:.0f}s', flush=True)
        selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        base.atomic_json(out / 'complete.json', dict(protocol=signature, best_epoch=selected['epoch'],
            epochs_run=history[-1]['epoch'], best_validation_MSE=best, best_sha256=base.file_hash(out / 'best.pt'),
            train_combo=len(tr['targets'][few.combination(fc)]), FM_features_used=False,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            selection='Original single-stimulus Validation only', combination_loss_weight=1/3))


def evaluate(cfg, device):
    out = ROOT / cfg['result_directory']; signature = protocol(cfg)
    done = json.loads((out / 'complete.json').read_text())
    if done['protocol'] != signature or done['best_sha256'] != base.file_hash(out / 'best.pt'):
        raise ValueError('Expression checkpoint must be complete and locked before Test')
    fc, source, genes, stats = inputs(cfg)
    base.atomic_json(out / 'test_protocol.json', done)
    group = expression_group(few.load_test(fc, cfg['replicate']))
    model = make_model(genes, stats, source, device)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=False)['model'])
    prediction = base.predict(model, group, few.combination(fc), source, device)
    x, truth = group['pbs'], group['targets'][few.combination(fc)]
    folder, _, _ = base.ready(source)
    with np.load(folder / 'distribution_reference.npz') as f:
        reference = dict(f)
    rows, moments, panels, observed, predicted = base.population_scores(prediction, truth, x, reference['functional_indices'])
    old_out = ROOT / fc['result_directory']
    old_genes = pd.read_csv(old_out / f'test/{cfg["replicate"]}/{few.label(cfg["fraction"])}/FiLM/gene_predictions.tsv.gz', sep='\t')
    if not genes.gene_id.equals(old_genes.gene_id):
        raise ValueError('FM and expression gene panels differ')
    np.testing.assert_allclose(moments['observed_delta'], old_genes.observed_delta, atol=1e-12, rtol=1e-12)
    distribution = []
    with np.load(ROOT / fc['data_directory'] / cfg['replicate'] / 'draws.npz') as draws:
        for i, seed in enumerate(source['distribution_seeds']):
            pi, ti = draws['predicted_indices'][i], draws['target_indices'][i]
            values = base.distribution_scores(prediction[pi], truth[ti], reference, panels, source['kl_alpha'])
            distribution.extend(dict(sample_seed=seed, **row) for row in values)
    p, d = pd.DataFrame(rows), pd.DataFrame(distribution)
    d.to_csv(out / 'distribution_draws.csv', index=False)
    d.drop(columns='sample_seed').groupby('panel').std().to_csv(out / 'distribution_sampling_sd.csv')
    scores = p.merge(d.drop(columns='sample_seed').groupby('panel').mean(), on='panel', how='left')
    scores['method'], scores['seed'] = METHOD, fc['seed']
    scores.to_csv(out / 'test_metrics.csv', index=False)
    np.save(out / 'predicted_log1p.npy', prediction)
    g = pd.concat([genes[['gene_id', 'gene_symbol']], pd.DataFrame(moments)], axis=1)
    g.to_csv(out / 'gene_predictions.tsv.gz', sep='\t', index=False)
    for name, order in [('observed', observed), ('predicted', predicted)]:
        g.iloc[order[:200]].to_csv(out / f'{name}_effect_top200.tsv.gz', sep='\t', index=False)
    old = pd.read_csv(old_out / 'test_metrics.csv', dtype={'batch': str})
    old = old[old.batch.eq(cfg['replicate']) & old.fraction.eq(cfg['fraction'])].copy()
    old.loc[old.method.eq('FiLM'), 'method'] = 'Scratch_v3_FiLM'
    old['seed'] = fc['seed']
    comparison = pd.concat([old, scores.assign(batch=cfg['replicate'], fraction=cfg['fraction'])], ignore_index=True)
    comparison.to_csv(out / 'comparison.csv', index=False)
    base.atomic_json(out / 'evaluation_complete.json', dict(protocol=signature,
        best_sha256=done['best_sha256'], same_observed_delta_as_FM=True,
        ranking='abs mean log1p(CP10K) delta; no p-values', test_PBS=len(x), test_combination=len(truth)))
    print(comparison[comparison.panel.isin(['Top100', 'All_genes'])][
        ['method', 'panel', 'MSE_delta', 'RMSE_delta', 'PCC_delta', 'R2_delta', 'common_DEGs_fraction',
         'E_distance_PCA50', 'W1_gene_mean', 'KL_hist_true_pred']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'all'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    _, source, _, _ = inputs(cfg)
    torch.set_num_threads(source['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    with threadpool_limits(source['cpu_threads']):
        if args.command in ('train', 'all'):
            train(cfg, device)
        if args.command in ('evaluate', 'all'):
            evaluate(cfg, device)


if __name__ == '__main__':
    main()
