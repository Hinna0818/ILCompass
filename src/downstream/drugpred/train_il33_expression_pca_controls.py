"""Three-seed expression/PCA controls for the four locked single-IL33 tasks.

Historical splits and target losses are retained per task. Existing raw/LoRA
results are read-only; only missing lung LoRA seeds and adaptations are trained.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import train_il33_mixed_delta as task
import train_il33_bidirectional_tissue as tissue
import train_representation_benchmark as representation
import train_film_multistimulus as film
from residual import ResidualPredictor
from il33_lung_metrics import evaluate
from effect_rank_gene_scores import score_gene_moments

ROOT = task.ROOT
CONFIG = ROOT / 'configs/il33_expression_pca_controls_v1.json'
BASELINES = ('ExpressionMLP', 'PCA')
LEARNED = ('ExpressionMLP', 'PCA', 'GeneCompass', 'LoRA')
EXPERIMENTS = {'mixed': 'IL33_lung_mixed_internal', 'C_D': 'IL33_C_to_D',
               'crossstudy': 'IL33_cross_study_target_PBS', 'adaptation': 'IL33_liver_adaptation'}


def read_json(path):
    return json.loads(Path(path).read_text())


def folder(cfg, key, name=''):
    return ROOT / cfg[key] / name


def link(source, target):
    source = source.resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError(f'Unexpected historical asset destination: {target}')
    else:
        target.symlink_to(source)


def configuration(cfg, name):
    parent = ROOT / cfg['sources'][name]
    original = read_json(parent / 'prepare_manifest.json')
    models = {m: 'expression-only' if m == 'ExpressionMLP' else 'Train-PBS PCA' for m in BASELINES}
    if name == 'lung_liver':
        models['LoRA'] = 'frozen historical GeneCompass LoRA'
    return dict(original['config'], seeds=cfg['seeds'], models=models,
        version='IL33_expression_PCA_controls_v1_' + name,
        **{key: str(folder(cfg, key, name).relative_to(ROOT))
           for key in ('data_directory', 'result_directory', 'model_directory')})


def legacy_head(cfg, model, seed):
    original = read_json(ROOT / cfg['sources']['lung_liver'] / 'config.json')
    if model in ('GeneCompass', 'ExpressionMLP'):
        return ROOT / original['model_directory'] / model / f'seed{seed}/final.pt'
    if model == 'LoRA' and seed == 0:
        return ROOT / cfg['lora_lung_results'] / 'checkpoints/GeneCompass_LoRA/seed0/final.pt'
    return None


def head_path(cfg, name, model, seed):
    if name == 'lung_liver':
        old = legacy_head(cfg, model, seed)
        if old is not None:
            return old
    return folder(cfg, 'model_directory', name) / model / f'seed{seed}' / (
        'best.pt' if name in ('mixed', 'crossstudy') else 'final.pt')


def adaptation_path(cfg, model, seed, fraction):
    pct = f'{round(100 * fraction):02d}pct'
    if seed == 0 and model in ('GeneCompass', 'LoRA'):
        old = 'raw_fewshot' if model == 'GeneCompass' else 'fewshot'
        return ROOT / 'results/ILC2_LoRA_IL33_v1' / old / 'checkpoints' / f'{pct}_step50.pt'
    return folder(cfg, 'model_directory', 'adaptation') / model / f'seed{seed}' / f'{pct}_step50.pt'


def audit(cfg):
    if cfg['seeds'] != [0, 1, 2] or cfg['adaptation_fractions'] != [.05, .1]:
        raise ValueError('This supplement fixes seeds0/1/2 and 5/10% adaptation')
    out = folder(cfg, 'result_directory'); out.mkdir(parents=True, exist_ok=True)
    counts, plan, protected = [], [], {}
    for name, path in cfg['sources'].items():
        source = ROOT / path; saved = read_json(source / 'prepare_manifest.json')
        task.prepared(saved['config'])
        protected[str(source / 'prepare_manifest.json')] = task.file_hash(source / 'prepare_manifest.json')
        ids = []
        for role in saved['cell_map_sha256']:
            m = pd.read_csv(source / role / 'metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
            if not m.index.is_unique or not m.split.eq(role).all():
                raise ValueError('Invalid locked source roles')
            ids.extend(m.index)
            counts.extend(dict(task=name, role=role, condition=c, cells=int(n))
                          for c, n in m.condition.value_counts().items())
        if len(ids) != len(set(ids)):
            raise ValueError('Historical source roles overlap')
    cd = read_json(ROOT / cfg['cd_config']); source = ROOT / cd['cache_directory']
    for role, batch in [('train', 'C'), ('validation', 'D')]:
        m = pd.read_csv(source / f'{role}_metadata.tsv.gz', sep='\t', index_col=0)
        if not m.author_batch.eq(batch).all():
            raise ValueError('C/D source labels changed')
        counts.extend(dict(task='C_D', role=role, condition=c, cells=int(n))
                      for c, n in m.condition.value_counts().items())
    for name in ('mixed', 'C_D', 'crossstudy', 'lung_liver'):
        methods = LEARNED if name == 'lung_liver' else BASELINES
        for model in methods:
            for seed in cfg['seeds']:
                path = head_path(cfg, name, model, seed)
                reuse = name == 'lung_liver' and legacy_head(cfg, model, seed) is not None
                if reuse and not path.is_file():
                    raise FileNotFoundError(path)
                plan.append(dict(task=name, model=model, seed=seed,
                                 action='reuse' if reuse else 'train', checkpoint=str(path.relative_to(ROOT))))
    for model in LEARNED:
        for seed in cfg['seeds']:
            for f in cfg['adaptation_fractions']:
                path = adaptation_path(cfg, model, seed, f)
                reuse = seed == 0 and model in ('GeneCompass', 'LoRA')
                if reuse and not path.is_file():
                    raise FileNotFoundError(path)
                plan.append(dict(task=f'adaptation_{round(100*f)}pct', model=model, seed=seed,
                                 action='reuse' if reuse else 'train', checkpoint=str(path.relative_to(ROOT))))
    for path in [ROOT / cfg['historical_metrics'], ROOT / cfg['adaptation_split']]:
        protected[str(path)] = task.file_hash(path)
    signature = dict(config=cfg, protected_sha256=protected, original_results_written=False,
                     PCA_fit='Train PBS only, independently for each task', encoder_inference=False)
    if (out / 'audit.json').exists() and read_json(out / 'audit.json') != signature:
        raise ValueError('Historical inputs or supplement configuration changed')
    task.atomic_json(out / 'audit.json', signature)
    pd.DataFrame(counts).to_csv(out / 'cell_counts.csv', index=False)
    pd.DataFrame(plan).to_csv(out / 'run_plan.csv', index=False)
    task.atomic_json(out / 'config.json', cfg)
    print('AUDIT', pd.DataFrame(plan).groupby(['task', 'action']).size().to_dict(), flush=True)


def prepare_pca(cfg, dest, roles):
    """roles contain original metadata/mmap expression; only Train PBS fits PCA."""
    m, y = roles['train']; selected = m.condition.eq('PBS').to_numpy()
    x = np.asarray(y[selected])
    stats = dict(control_mean=x.mean(0, dtype=float), control_std=x.std(0, dtype=float))
    print('PCA FIT Train PBS', dest.name, x.shape, '->', cfg['pca_dimensions'], flush=True)
    reference, _ = representation.fit_pca(dict(pbs=x), stats,
        dict(minimum_expression_scale=cfg['minimum_expression_scale']), cfg)
    np.savez_compressed(dest / 'pca.npz', **reference)
    pd.Series(m.index[selected], name='cell_id').to_csv(dest / 'pca_fit_cells.tsv.gz', sep='\t', index=False)
    hashes = {}
    for role, (meta, expression) in roles.items():
        d = dest / role; d.mkdir(exist_ok=True)
        values = np.asarray(expression[meta.condition.eq('PBS').to_numpy()])
        path = d / 'PCA_h.npy'; np.save(path, representation.transform(values, reference))
        hashes[role] = task.file_hash(path)
    return hashes


def prepare(cfg):
    audit(cfg)
    for name, path in cfg['sources'].items():
        parent = ROOT / path; rc = configuration(cfg, name)
        data, out, _ = task.paths(rc); data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
        if (data / 'prepare_manifest.json').exists():
            task.prepared(rc); print('PREPARE reused', name, flush=True); continue
        old = read_json(parent / 'prepare_manifest.json')
        link(parent / 'gene_panel.tsv', data / 'gene_panel.tsv')
        roles = {}
        for role in old['cell_map_sha256']:
            for filename in ('metadata.tsv.gz', 'y.npy'):
                link(parent / role / filename, data / role / filename)
            roles[role] = task.load_expression_split(parent, old, role)
        link(parent / 'train/statistics.npz', data / 'train/statistics.npz')
        hashes = prepare_pca(cfg, data, roles)
        if name == 'lung_liver':
            lora = ROOT / cfg['lora_lung_data']; lm = read_json(lora / 'prepare_manifest.json')
            np.testing.assert_array_equal(pd.read_csv(data / 'gene_panel.tsv', sep='\t').gene_id,
                                          pd.read_csv(lora / 'gene_panel.tsv', sep='\t').gene_id)
            for role, (m, _) in roles.items():
                om = pd.read_csv(lora / role / 'metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
                np.testing.assert_array_equal(m.index, om.index)
                if task.file_hash(lora / role / 'GeneCompass_LoRA_h.npy') != lm['aligned_feature_sha256'][role]:
                    raise ValueError('Historical LoRA features changed')
                link(lora / role / 'GeneCompass_LoRA_h.npy', data / role / 'LoRA_h.npy')
        manifest = dict(config=rc, panel_sha256=old['panel_sha256'],
            training_statistics_sha256=old['training_statistics_sha256'],
            cell_map_sha256=old['cell_map_sha256'], output_genes=old['output_genes'],
            encoder_dimensions={'PCA': cfg['pca_dimensions'], 'LoRA': 768},
            parent=str(parent), parent_sha256=task.file_hash(parent / 'prepare_manifest.json'),
            pca_sha256=task.file_hash(data / 'pca.npz'), feature_sha256=hashes,
            PCA_fit='Train PBS only; Validation/Test transformed without refitting')
        task.atomic_json(data / 'config.json', rc); task.atomic_json(data / 'prepare_manifest.json', manifest)
    # C/D retains its distinct residual objective and fixed fifteen-epoch schedule.
    cd = read_json(ROOT / cfg['cd_config']); parent = ROOT / cd['cache_directory']
    data = folder(cfg, 'data_directory', 'C_D'); data.mkdir(parents=True, exist_ok=True)
    if not (data / 'complete.json').exists():
        link(parent / 'genes.tsv', data / 'gene_panel.tsv'); roles = {}
        for role, old_role in [('train', 'train'), ('test', 'validation')]:
            link(parent / f'{old_role}_metadata.tsv.gz', data / role / 'metadata.tsv.gz')
            link(parent / f'{old_role}_y.npy', data / role / 'y.npy')
            roles[role] = (pd.read_csv(data / role / 'metadata.tsv.gz', sep='\t', index_col=0),
                           np.load(data / role / 'y.npy', mmap_mode='r'))
        hashes = prepare_pca(cfg, data, roles)
        task.atomic_json(data / 'complete.json', dict(config=cfg, source_config=cd,
            source_signature=read_json(parent / 'signature.json'), feature_sha256=hashes,
            pca_sha256=task.file_hash(data / 'pca.npz')))
    prepare_adaptation(cfg)


def features(cfg, name, role, model):
    if name == 'C_D':
        data = folder(cfg, 'data_directory', name)
        m = pd.read_csv(data / role / 'metadata.tsv.gz', sep='\t', index_col=0)
        y = np.load(data / role / 'y.npy', mmap_mode='r')
    else:
        rc = configuration(cfg, name); data, _, _, saved, _ = task.prepared(rc)
        m, y = task.load_expression_split(data, saved, role)
    pbs = m.condition.eq('PBS').to_numpy()
    if model == 'ExpressionMLP':
        h = np.asarray(y[pbs])
    elif model == 'GeneCompass':
        h = np.load(ROOT / cfg['sources']['lung_liver'] / role / 'GeneCompass_h.npy', mmap_mode='r')
    else:
        h = np.load(data / role / f'{model}_h.npy', mmap_mode='r')
    if h.shape[0] != int(pbs.sum()):
        raise ValueError('Feature/PBS row mismatch')
    return m, y, h


def prepare_adaptation(cfg):
    dest = folder(cfg, 'data_directory', 'adaptation'); dest.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(ROOT / cfg['adaptation_split'], sep='\t', index_col=0)
    m, y, _ = features(cfg, 'lung_liver', 'test', 'PCA')
    np.testing.assert_array_equal(split.index, m.index)
    pbs = m.condition.eq('PBS').to_numpy(); positions = np.full(len(m), -1)
    positions[pbs] = np.arange(pbs.sum()); smaller = None
    for f in cfg['adaptation_fractions']:
        key = f'{round(100*f):02d}pct'; selected = split[key].to_numpy()
        if split.loc[selected, 'role'].eq('test').any() or (smaller is not None and (smaller & ~selected).any()):
            raise ValueError('Adaptation must be nested and disjoint from fixed Holdout')
        smaller = selected
        pr, tr = np.flatnonzero(selected & pbs), np.flatnonzero(selected & ~pbs)
        x = np.asarray(y[pr]); mean = y[tr].mean(0, dtype=float)
        reference = ROOT / 'data/processed/ILC2_LoRA_IL33_v1/fewshot' / f'{key}.npz'
        with np.load(reference) as old:
            np.testing.assert_array_equal(x, old['control'])
            np.testing.assert_array_equal(mean, old['target_mean'])
        for model in LEARNED:
            _, _, h = features(cfg, 'lung_liver', 'test', model)
            d = dest / model; d.mkdir(exist_ok=True)
            np.savez_compressed(d / f'{key}.npz', control=x, h=np.asarray(h[positions[pr]]),
                                target_mean=mean, delta=mean-x.mean(0, dtype=float))
    link(ROOT / cfg['adaptation_split'], dest / 'split_manifest.tsv.gz')


def check(cfg):
    audit(cfg); rows = []
    for name in (*cfg['sources'], 'C_D'):
        data = folder(cfg, 'data_directory', name)
        meta = pd.read_csv(data / 'train/metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
        fit = pd.read_csv(data / 'pca_fit_cells.tsv.gz', sep='\t').cell_id
        np.testing.assert_array_equal(fit, meta.index[meta.condition.eq('PBS')])
        ref = dict(np.load(data / 'pca.npz'))
        if ref['components'].shape[0] != cfg['pca_dimensions']:
            raise ValueError('PCA must match GeneCompass 768-dimensional input')
        if name != 'C_D':
            rc = configuration(cfg, name); _, _, _, saved, _ = task.prepared(rc)
            parent = ROOT / cfg['sources'][name]
            if saved['pca_sha256'] != task.file_hash(data / 'pca.npz'):
                raise ValueError('PCA reference changed')
            for role, digest in saved['feature_sha256'].items():
                if task.file_hash(data / role / 'PCA_h.npy') != digest:
                    raise ValueError('PCA features changed')
                for file in ('y.npy', 'metadata.tsv.gz'):
                    if (data / role / file).resolve() != (parent / role / file).resolve():
                        raise ValueError('Original split/expression not reused')
            for key, value in read_json(parent / 'config.json').items():
                if key not in ('version', 'models', 'seeds', 'data_directory', 'result_directory', 'model_directory'):
                    if rc[key] != value:
                        raise ValueError(f'Training parameter changed: {name}/{key}')
        else:
            saved = read_json(data / 'complete.json')
            if (saved['config'] != cfg or saved['source_config'] != read_json(ROOT / cfg['cd_config'])
                    or saved['pca_sha256'] != task.file_hash(data / 'pca.npz')):
                raise ValueError('C/D preparation protocol changed')
            for role, digest in saved['feature_sha256'].items():
                if task.file_hash(data / role / 'PCA_h.npy') != digest:
                    raise ValueError('C/D PCA features changed')
        genes = len(pd.read_csv(data / 'gene_panel.tsv', sep='\t'))
        hidden = read_json(ROOT / cfg['cd_config'])['hidden_size'] if name == 'C_D' else configuration(cfg, name)['hidden_size']
        expression = (ExpressionResidual(genes, hidden, torch.zeros(2, genes)) if name == 'C_D'
                      else task.ExpressionDeltaMLP(genes, hidden))
        projected = (ResidualPredictor(torch.zeros(2, cfg['pca_dimensions']), torch.zeros(2, genes), hidden)
                     if name == 'C_D' else task.DirectDeltaHead(torch.zeros(2, cfg['pca_dimensions']), genes, hidden))
        rows.append(dict(task=name, train_PBS=len(fit), PCA_dimensions=cfg['pca_dimensions'],
                         PCA_train_only=True, fixed_original_targets=True,
                         expression_parameters=sum(p.numel() for p in expression.parameters()),
                         PCA_head_parameters=sum(p.numel() for p in projected.parameters())))
    task.atomic_json(folder(cfg, 'result_directory') / 'input_acceptance.json',
                     dict(passed=True, checks=rows, formal_training_started=False, encoder_inference=False))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


class ExpressionResidual(nn.Module):
    """C/D expression baseline: retain the historical mean offset and objective."""
    def __init__(self, genes, hidden, effects):
        super().__init__(); self.register_buffer('mean_effects', effects)
        self.residual = task.ExpressionDeltaMLP(genes, hidden)

    def forward(self, h, y, condition):
        return y + condition @ self.mean_effects + self.residual(h, condition)


def cd_model(model, h, effects, genes, cfg):
    return (ExpressionResidual(genes, cfg['hidden_size'], effects) if model == 'ExpressionMLP'
            else ResidualPredictor(h, effects, cfg['hidden_size']))


def train_cd(cfg, model_name, seed, device):
    cd = read_json(ROOT / cfg['cd_config'])
    path = head_path(cfg, 'C_D', model_name, seed); path.parent.mkdir(parents=True, exist_ok=True)
    signature = task.file_hash(folder(cfg, 'data_directory', 'C_D') / 'complete.json')
    with (path.parent / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (path.parent / 'complete.json').exists():
            saved = read_json(path.parent / 'complete.json')
            if saved['fingerprint'] != signature or task.file_hash(path) != saved['final_sha256']:
                raise ValueError('C/D checkpoint changed')
            print('REUSE C_D', model_name, seed, flush=True); return
        m, y, h = features(cfg, 'C_D', 'train', model_name)
        x = torch.tensor(np.asarray(y[m.condition.eq('PBS').to_numpy()]), device=device)
        target = torch.tensor(np.asarray(y[m.condition.eq('IL33').to_numpy()]), device=device)
        h = torch.tensor(np.asarray(h), device=device)
        delta = target.mean(0)-x.mean(0)
        scale = torch.cat([x, target]).std(0, unbiased=False).clamp_min(cd['minimum_expression_scale'])
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = cd_model(model_name, h, torch.stack([delta, torch.zeros_like(delta)]), x.shape[1], cd).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cd['learning_rate'], weight_decay=cd['weight_decay'])
        start, history = 1, []; latest = path.parent / 'latest.pt'
        if latest.exists():
            saved = torch.load(latest, map_location='cpu', weights_only=False)
            if saved['fingerprint'] != signature:
                raise ValueError('C/D resume source changed')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            start, history = saved['epoch']+1, saved['history']
        loss_cfg = dict(cd, shift_loss_weight=cd['residual_penalty'])
        for epoch in range(start, cd['epochs']+1):
            model.train(); order = rng.permutation(len(target)); total = 0.
            for left in range(0, len(order), cd['batch_size']):
                ti = order[left:left+cd['batch_size']]; ci = rng.choice(len(x), len(ti), replace=True)
                condition = x.new_tensor([1., 0.]).expand(len(ci), -1)
                raw = model(h[ci], x[ci], condition)
                loss, _, _, _ = film.loss_terms(dict(prediction=raw.clamp_min(0), raw_delta=raw-x[ci]-delta),
                                               target[ti], scale, loss_cfg)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite C/D objective')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cd['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); total += float(loss.detach()) * len(ti)/len(target)
            history.append(dict(epoch=epoch, train_loss=total))
            task.atomic_torch(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                fingerprint=signature, epoch=epoch, history=history, numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            pd.DataFrame(history).to_csv(path.parent / 'history.csv', index=False)
            print('C_D', model_name, seed, epoch, total, flush=True)
        task.atomic_torch(path, dict(model=model.state_dict(), epoch=cd['epochs'], seed=seed, selection='fixed_final_epoch'))
        task.atomic_json(path.parent / 'complete.json', dict(fingerprint=signature, epoch=cd['epochs'],
            final_sha256=task.file_hash(path), Test_used_for_selection=False, encoder_updates=0))


def train(cfg, device):
    for name in ('mixed', 'crossstudy'):
        rc = configuration(cfg, name)
        for model in BASELINES:
            for seed in cfg['seeds']:
                print('EXPERIMENT', name, model, f'seed{seed}', flush=True)
                task.train_one(rc, model, seed, device, resume=True, expression_input=model == 'ExpressionMLP')
    for model in BASELINES:
        for seed in cfg['seeds']:
            train_cd(cfg, model, seed, device)
    rc = configuration(cfg, 'lung_liver')
    for model in ('PCA', 'LoRA'):
        for seed in cfg['seeds']:
            if legacy_head(cfg, model, seed) is None:
                tissue.train_fixed(rc, model, seed, device, resume=True)


def lock_head(cfg, name, model, seed):
    path = head_path(cfg, name, model, seed); saved = read_json(path.parent / 'complete.json')
    key = 'best_sha256' if path.name == 'best.pt' else 'final_sha256'
    if saved[key] != task.file_hash(path):
        raise ValueError(f'Incomplete or changed checkpoint: {path}')
    if name == 'lung_liver' and legacy_head(cfg, model, seed) is not None:
        source = ROOT / (cfg['lora_lung_data'] if model == 'LoRA' else cfg['sources']['lung_liver'])
        expected = task.file_hash(source / 'prepare_manifest.json')
    else:
        expected = task.file_hash(folder(cfg, 'data_directory', name) /
                                  ('complete.json' if name == 'C_D' else 'prepare_manifest.json'))
    if saved['fingerprint'] != expected:
        raise ValueError(f'Checkpoint source/configuration differs: {path}')
    return dict(path=str(path.relative_to(ROOT)), sha256=saved[key])


def adapt(cfg, device):
    # All source heads finish before reading any target adaptation response.
    for model in LEARNED:
        for seed in cfg['seeds']:
            lock_head(cfg, 'lung_liver', model, seed)
    ac = read_json(ROOT / cfg['adaptation_config']); rc = configuration(cfg, 'lung_liver')
    genes = len(pd.read_csv(folder(cfg, 'data_directory', 'lung_liver/gene_panel.tsv'), sep='\t'))
    for model_name in LEARNED:
        for seed in cfg['seeds']:
            initial = torch.load(head_path(cfg, 'lung_liver', model_name, seed), map_location='cpu', weights_only=False)
            for fraction in cfg['adaptation_fractions']:
                path = adaptation_path(cfg, model_name, seed, fraction)
                if seed == 0 and model_name in ('GeneCompass', 'LoRA'):
                    saved = read_json(path.parent.parent / 'training_complete.json')
                    if saved['checkpoints']['checkpoints/'+path.name] != task.file_hash(path):
                        raise ValueError('Historical adaptation changed')
                    print('REUSE ADAPT', model_name, seed, fraction, flush=True); continue
                path.parent.mkdir(parents=True, exist_ok=True)
                marker = path.with_suffix('.json'); key = f'{round(100*fraction):02d}pct'
                signature = dict(source=lock_head(cfg, 'lung_liver', model_name, seed),
                    config=ac, seed=seed, fraction=fraction, split_sha256=task.file_hash(ROOT / cfg['adaptation_split']),
                    adaptation_sha256=task.file_hash(folder(cfg, 'data_directory', f'adaptation/{model_name}/{key}.npz')))
                if marker.exists():
                    saved = read_json(marker)
                    if saved['signature'] != signature or saved['sha256'] != task.file_hash(path):
                        raise ValueError('Adaptation protocol changed')
                    continue
                values = dict(np.load(folder(cfg, 'data_directory', f'adaptation/{model_name}/{key}.npz')))
                torch.manual_seed(seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)
                rng = np.random.default_rng(seed)
                model = task.restore_head(initial, rc, genes, device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=ac['learning_rate'], weight_decay=ac['weight_decay'])
                x, h = values['control'], values['h']; target = torch.tensor(values['delta'], dtype=torch.float32, device=device)
                history = []
                for step in range(1, ac['steps']+1):
                    rows = rng.choice(len(x), min(len(x), ac['batch_size']), replace=False)
                    xx = torch.tensor(x[rows], device=device); hh = torch.tensor(h[rows], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = task.delta_loss(xx, model(hh), target)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite adaptation objective')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), ac['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); history.append(dict(step=step, train_loss=float(loss.detach())))
                    if step in ac['save_steps']:
                        checkpoint = path.with_name(f'{key}_step{step}.pt')
                        task.atomic_torch(checkpoint, dict(model=model.state_dict(), seed=seed,
                            step=step, fraction=fraction, selection='fixed_step_no_test_selection'))
                for name, value in model.named_buffers():
                    torch.testing.assert_close(value.cpu(), initial['model'][name], rtol=0, atol=0)
                pd.DataFrame(history).to_csv(path.with_suffix('.csv'), index=False)
                task.atomic_json(marker, dict(signature=signature, sha256=task.file_hash(path),
                                             Test_used_for_training_or_selection=False))
                print('ADAPT COMPLETE', model_name, seed, key, flush=True)


def score_prediction(cfg, pred, truth, control, panel, info, dest):
    _, moments = evaluate(pred, truth, control, [], [], top_n=())
    table = pd.concat([panel[['gene_id', 'gene_symbol']].reset_index(drop=True), pd.DataFrame(moments)], axis=1)
    if info['experiment'] == EXPERIMENTS['adaptation']:
        reference_path = ROOT / 'results/ILC2_LoRA_IL33_v1/raw_fewshot/test/zero_shot/gene_predictions.tsv.gz'
        prefix = ''
    else:
        history = pd.read_csv(ROOT / 'results/ILC2_LoRA_IL33_v1/analysis/metrics_by_background_seed.csv').fillna({'prefix': ''})
        # The historical external evaluation has one target background, labelled "all".
        background = 'all' if info['experiment'] == EXPERIMENTS['crossstudy'] else info['background']
        references = history[(history.experiment == info['experiment']) & history.background.eq(background)
                             & history.model.eq('GeneCompass') & history.seed.eq(0)][['source', 'prefix']].drop_duplicates()
        if len(references) != 1:
            raise ValueError(f"Need one historical reference: {info['experiment']}/{background}")
        row = references.iloc[0]
        reference_path, prefix = ROOT / row.source, row.prefix
    reference = pd.read_csv(reference_path, sep='\t')
    if prefix:
        reference = reference.rename(columns={prefix+'_'+k: k for k in moments})
    np.testing.assert_array_equal(table.gene_id, reference.gene_id)
    keys = ['observed_mean', 'control_mean', 'observed_delta', 'observed_var']
    np.testing.assert_allclose(table[keys], reference[keys], rtol=1e-8, atol=1e-10)
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / 'gene_predictions.tsv.gz', sep='\t', index=False)
    functional = pd.read_csv(ROOT / cfg['functional_genes']).gene_id
    return score_gene_moments(info, table, functional)[0]


def evaluate_all(cfg, device):
    locks = {}
    for name in ('mixed', 'C_D', 'crossstudy'):
        for model in BASELINES:
            for seed in cfg['seeds']:
                locks[f'{name}/{model}/{seed}'] = lock_head(cfg, name, model, seed)
    for model in LEARNED:
        for seed in cfg['seeds']:
            locks[f'lung_liver/{model}/{seed}'] = lock_head(cfg, 'lung_liver', model, seed)
            for f in cfg['adaptation_fractions']:
                path = adaptation_path(cfg, model, seed, f)
                marker = (read_json(path.parent.parent / 'training_complete.json') if
                          seed == 0 and model in ('GeneCompass', 'LoRA') else read_json(path.with_suffix('.json')))
                digest = marker['checkpoints']['checkpoints/'+path.name] if 'checkpoints' in marker else marker['sha256']
                if task.file_hash(path) != digest:
                    raise ValueError('Finish and lock every adaptation before Holdout scoring')
                locks[f'adaptation/{model}/{seed}/{f}'] = dict(path=str(path.relative_to(ROOT)), sha256=digest)
    out = folder(cfg, 'result_directory'); protocol = out / 'test_protocol.json'
    if protocol.exists() and read_json(protocol) != locks:
        raise ValueError('Test was already scored with different heads')
    task.atomic_json(protocol, locks)
    rows = []
    for name in ('mixed', 'C_D', 'crossstudy'):
        rc = read_json(ROOT / cfg['cd_config']) if name == 'C_D' else configuration(cfg, name)
        panel = pd.read_csv(folder(cfg, 'data_directory', name) / 'gene_panel.tsv', sep='\t')
        for method in BASELINES:
            m, y, h = features(cfg, name, 'test', method)
            pm = m[m.condition.eq('PBS')]
            for seed in cfg['seeds']:
                state = torch.load(head_path(cfg, name, method, seed), map_location='cpu', weights_only=False)
                if name == 'C_D':
                    weights = state['model']; effects = weights['mean_effects'].to(device)
                    model = cd_model(method, torch.zeros(2, h.shape[1], device=device), effects, len(panel), rc).to(device)
                    model.load_state_dict(weights)
                    x = np.asarray(y[m.condition.eq('PBS').to_numpy()]); pred = []
                    model.eval()
                    with torch.no_grad():
                        for left in range(0, len(x), 128):
                            xx = torch.as_tensor(x[left:left+128], device=device)
                            hh = torch.as_tensor(np.asarray(h[left:left+128]), device=device)
                            pred.append(model(hh, xx, xx.new_tensor([1., 0.]).expand(len(xx), -1)).clamp_min(0).cpu().numpy())
                    pred = np.concatenate(pred); truth = np.asarray(y[m.condition.eq('IL33').to_numpy()])
                    rows.extend(score_prediction(cfg, pred, truth, x, panel,
                        dict(experiment=EXPERIMENTS[name], model=method, seed=seed, background='all', adaptation='none'),
                        out / name / method / f'seed{seed}/test'))
                else:
                    model = task.restore_head(state, rc, len(panel), device)
                    groups = rc['backgrounds'] if name == 'mixed' else rc['external_backgrounds']
                    for group in groups:
                        selected = pm.response_group.eq(group).to_numpy()
                        x = np.asarray(y[(m.condition.eq('PBS') & m.response_group.eq(group)).to_numpy()])
                        truth = np.asarray(y[(m.condition.eq('IL33') & m.response_group.eq(group)).to_numpy()])
                        pred = x + task.predict_delta(model, h[selected], x, rc['prediction_batch_size'], device)
                        rows.extend(score_prediction(cfg, pred, truth, x, panel,
                            dict(experiment=EXPERIMENTS[name], model=method, seed=seed, background=group, adaptation='none'),
                            out / name / method / f'seed{seed}/test' / group))
                del model
        pd.DataFrame(rows).to_csv(out / 'new_metrics_by_background_seed.csv', index=False)
    rc = configuration(cfg, 'lung_liver'); data = folder(cfg, 'data_directory', 'lung_liver')
    panel = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
    split = pd.read_csv(ROOT / cfg['adaptation_split'], sep='\t', index_col=0)
    for method in LEARNED:
        m, y, h = features(cfg, 'lung_liver', 'test', method)
        np.testing.assert_array_equal(m.index, split.index)
        pbs = m.condition.eq('PBS').to_numpy(); holdout = split.role.eq('test').to_numpy()
        x = np.asarray(y[holdout & pbs]); truth = np.asarray(y[holdout & ~pbs]); hh = np.asarray(h[holdout[pbs]])
        for seed in cfg['seeds']:
            for fraction in [0., *cfg['adaptation_fractions']]:
                path = head_path(cfg, 'lung_liver', method, seed) if fraction == 0 else adaptation_path(cfg, method, seed, fraction)
                state = torch.load(path, map_location='cpu', weights_only=False)
                model = task.restore_head(state, rc, len(panel), device)
                pred = x + task.predict_delta(model, hh, x, rc['prediction_batch_size'], device)
                label = 'Zero-shot' if fraction == 0 else f'{round(100*fraction)}%'
                rows.extend(score_prediction(cfg, pred, truth, x, panel,
                    dict(experiment=EXPERIMENTS['adaptation'], model=method, seed=seed, background='all', adaptation=label),
                    out / 'adaptation' / method / f'seed{seed}' / label))
                del model
    new = pd.DataFrame(rows); new.to_csv(out / 'new_metrics_by_background_seed.csv', index=False)
    metric_columns = ['delta_MSE', 'delta_R2', 'delta_Pearson', 'mean_R2', 'variance_R2',
                      'direction_accuracy', 'overlap', 'functional_MSE']
    macro = new.groupby(['experiment', 'model', 'seed', 'adaptation', 'panel'])[metric_columns].mean().reset_index()
    macro['delta_RMSE'] = np.sqrt(macro.delta_MSE)
    old = pd.read_csv(ROOT / cfg['historical_metrics'])
    old = old[old.experiment.isin([EXPERIMENTS[n] for n in ('mixed', 'C_D', 'crossstudy')]) &
              old.model.isin(['GeneCompass', 'LoRA', 'control_copy', 'background_mean_delta', 'C_mean_delta', 'global_mean_delta'])].copy()
    old['adaptation'] = 'none'
    combined = pd.concat([old, macro], ignore_index=True)
    combined.to_csv(out / 'metrics_by_seed.csv', index=False)
    combined.groupby(['experiment', 'adaptation', 'model', 'panel'])[metric_columns+['delta_RMSE']].agg(
        ['mean', 'std', 'count']).to_csv(out / 'metrics_summary.csv')
    for path, digest in read_json(out / 'audit.json')['protected_sha256'].items():
        if task.file_hash(path) != digest:
            raise ValueError('Historical results or split were modified')
    task.atomic_json(out / 'evaluation_complete.json', dict(locked_checkpoints=locks,
        original_results_unchanged=True, Test_used_for_selection=False, seeds=cfg['seeds'],
        effect_ranking='abs mean logCP10K delta; no p-value selection',
        aggregation='Equal background MSE then sqrt within seed; arithmetic mean across three seeds'))
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
    with threadpool_limits(cfg['cpu_threads']):
        if args.stage == 'audit':
            audit(cfg)
        if args.stage in ('prepare', 'all'):
            prepare(cfg)
        if args.stage in ('check', 'all'):
            check(cfg)
        if args.stage in ('train', 'all'):
            train(cfg, device)
        if args.stage in ('adapt', 'all'):
            adapt(cfg, device)
        if args.stage in ('evaluate', 'all'):
            evaluate_all(cfg, device)
        if args.stage in ('plot', 'all'):
            subprocess.run([sys.executable, str(ROOT / 'main/plot_il33_expression_pca_controls.py'),
                            '--config', str(args.config)], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
