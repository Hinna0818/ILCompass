"""Nested liver adaptation from the fixed Lung-trained v3 head, common holdout."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from il33_data import ROOT, task
from il33_benchmark import effect_rank_metrics
from il33_lung_metrics import correlation, differential, evaluate, r2

CONFIG = ROOT / 'scratch/configs/il33_liver_fewshot_v3_seed0.json'
TOP_N = (20, 50, 100, 200)
ENCODER = 'Scratch_v3'


def label(fraction):
    return f'{round(fraction * 100):02d}pct'


def prepare(cfg):
    data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if (data / 'manifest.json').exists():
        ready(cfg)
        return
    source = ROOT / cfg['source_data']
    parent = json.loads((source / 'prepare_manifest.json').read_text())
    meta = pd.read_csv(source / 'test/metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
    if not meta.index.is_unique or not (meta.study.eq('GSE189285') & meta.tissue.eq('Liver')).all():
        raise ValueError('Unexpected liver source identities')
    rng = np.random.default_rng(cfg['split_seed'])
    manifest = meta[['study', 'tissue', 'condition', 'library_id']].copy()
    manifest['source_row'] = np.arange(len(meta))
    manifest['role'] = 'test'
    for fraction in cfg['fractions']:
        manifest[label(fraction)] = False
    for condition in ['PBS', 'IL33']:
        order = rng.permutation(np.flatnonzero(meta.condition.eq(condition)))
        pool = order[:int(len(order) * cfg['adaptation_pool_fraction'])]
        manifest.loc[meta.index[pool], 'role'] = 'adaptation_pool'
        for fraction in cfg['fractions']:
            rows = order[:int(len(order) * fraction)]
            if not len(rows) or not set(rows).issubset(pool):
                raise ValueError('Invalid adaptation fraction')
            manifest.loc[meta.index[rows], label(fraction)] = True
    manifest.to_csv(data / 'split_manifest.tsv.gz', sep='\t', index_label='cell_id')
    feature_path = source / f'test/{ENCODER}_h.npy'
    if task.file_hash(feature_path) != parent['aligned_feature_sha256']['test']:
        raise ValueError('Frozen liver features changed')
    y = np.load(source / 'test/y.npy', mmap_mode='r')
    h = np.load(feature_path, mmap_mode='r')
    pbs_rows = np.flatnonzero(meta.condition.eq('PBS'))
    feature_rows = np.full(len(meta), -1); feature_rows[pbs_rows] = np.arange(len(pbs_rows))
    counts = []
    for fraction in cfg['fractions']:
        selected = manifest[label(fraction)].to_numpy()
        control_rows = np.flatnonzero(selected & meta.condition.eq('PBS').to_numpy())
        target_rows = np.flatnonzero(selected & meta.condition.eq('IL33').to_numpy())
        if manifest.loc[selected, 'role'].eq('test').any():
            raise ValueError('Adaptation/test overlap')
        control = np.array(y[control_rows])
        target_mean = y[target_rows].mean(0, dtype=np.float64)
        np.savez_compressed(data / f'{label(fraction)}.npz', control=control,
            h=np.asarray(h[feature_rows[control_rows]]), target_mean=target_mean,
            delta=target_mean - control.mean(0, dtype=np.float64))
        counts.append(dict(fraction=fraction, PBS=len(control_rows), IL33=len(target_rows)))
    test_counts = manifest[manifest.role.eq('test')].condition.value_counts().to_dict()
    pd.DataFrame(counts).to_csv(out / 'adaptation_cell_counts.csv', index=False)
    head = ROOT / cfg['source_results'] / f'checkpoints/{ENCODER}/seed0/final.pt'
    protected = {str(p.relative_to(ROOT)): task.file_hash(p) for p in [head,
        ROOT / cfg['source_results'] / 'comparison_seed0.csv', source / 'prepare_manifest.json']}
    task.atomic_json(data / 'manifest.json', dict(config=cfg, protected_sha256=protected,
        split_sha256=task.file_hash(data / 'split_manifest.tsv.gz'),
        adaptation_sha256={label(f): task.file_hash(data / f'{label(f)}.npz') for f in cfg['fractions']},
        test_counts=test_counts, adaptation_counts=counts,
        protocol='Fixed nested condition-stratified split; all checkpoints locked before holdout scoring',
        encoder_updates=0, head_standardization='Keep original Lung-only h_mean/h_std'))
    task.atomic_json(out / 'config.json', cfg)
    print('PREPARED', counts, 'HOLDOUT', test_counts, flush=True)


def ready(cfg):
    data, out = ROOT / cfg['data_directory'], ROOT / cfg['result_directory']
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['config'] != cfg or task.file_hash(data / 'split_manifest.tsv.gz') != manifest['split_sha256']:
        raise ValueError('Fixed adaptation configuration/split changed')
    for path, digest in manifest['protected_sha256'].items():
        if task.file_hash(ROOT / path) != digest:
            raise ValueError('Source model/result changed')
    for name, digest in manifest['adaptation_sha256'].items():
        if task.file_hash(data / f'{name}.npz') != digest:
            raise ValueError('Adaptation training data changed')
    return data, out, manifest


def source_head(cfg):
    return torch.load(ROOT / cfg['source_results'] / f'checkpoints/{ENCODER}/seed0/final.pt',
                      map_location='cpu', weights_only=False)


def train(cfg):
    data, out, manifest = ready(cfg)
    if (out / 'training_complete.json').exists():
        return
    parent = json.loads((ROOT / cfg['source_data'] / 'config.json').read_text())
    panel = pd.read_csv(ROOT / cfg['source_data'] / 'gene_panel.tsv', sep='\t')
    initial = source_head(cfg)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint_hashes, history = {}, []
    for fraction in cfg['fractions']:
        name = label(fraction)
        torch.manual_seed(cfg['training_seed']); torch.cuda.manual_seed_all(cfg['training_seed'])
        rng = np.random.default_rng(cfg['training_seed'])
        model = task.restore_head(initial, parent, len(panel), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        with np.load(data / f'{name}.npz') as values:
            x, h = values['control'], values['h']
            target = torch.tensor(values['delta'], dtype=torch.float32, device=device)
        started = time.monotonic()
        for step in range(1, cfg['steps'] + 1):
            rows = rng.choice(len(x), size=min(len(x), cfg['batch_size']), replace=False)
            xx, hh = torch.tensor(x[rows], device=device), torch.tensor(h[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = task.delta_loss(xx, model(hh), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite adaptation loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(fraction=fraction, step=step, train_loss=float(loss.detach()),
                                grad_norm=float(norm), elapsed_seconds=time.monotonic() - started))
            if step in cfg['save_steps']:
                for key in ['h_mean', 'h_std', 'mean_effects']:
                    torch.testing.assert_close(model.state_dict()[key].cpu(), initial['model'][key], rtol=0, atol=0)
                path = out / 'checkpoints' / f'{name}_step{step}.pt'
                path.parent.mkdir(exist_ok=True)
                task.atomic_torch(path, dict(model=model.state_dict(), fraction=fraction, step=step,
                    config=cfg, split_sha256=manifest['split_sha256'], selection='fixed_step_no_test_selection'))
                checkpoint_hashes[str(path.relative_to(out))] = task.file_hash(path)
                print(f'ADAPT {name} step={step} train_loss={float(loss):.7g}', flush=True)
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    task.atomic_json(out / 'training_complete.json', dict(config=cfg, checkpoints=checkpoint_hashes,
        test_used_for_training_or_selection=False, encoder_updates=0))


def score(cfg):
    data, out, manifest = ready(cfg)
    complete = json.loads((out / 'training_complete.json').read_text())
    if complete['config'] != cfg:
        raise ValueError('Training protocol changed')
    for name, digest in complete['checkpoints'].items():
        if task.file_hash(out / name) != digest:
            raise ValueError('Locked head changed')
    source = ROOT / cfg['source_data']
    parent = json.loads((source / 'config.json').read_text())
    panel = pd.read_csv(source / 'gene_panel.tsv', sep='\t')
    split = pd.read_csv(data / 'split_manifest.tsv.gz', sep='\t', index_col=0)
    meta = pd.read_csv(source / 'test/metadata.tsv.gz', sep='\t', index_col=0, low_memory=False)
    np.testing.assert_array_equal(split.index, meta.index)
    test = split.role.eq('test').to_numpy()
    pbs = meta.condition.eq('PBS').to_numpy()
    y = np.load(source / 'test/y.npy', mmap_mode='r')
    x, truth = np.array(y[test & pbs]), np.array(y[test & ~pbs])
    h = np.asarray(np.load(source / f'test/{ENCODER}_h.npy', mmap_mode='r')[test[pbs]])
    de, true_order = differential(truth, x, np.ones(len(panel), dtype=bool))
    pd.concat([panel[['gene_id', 'gene_symbol']], de], axis=1).to_csv(out / 'holdout_observed_DEG.tsv.gz', sep='\t', index=False)
    rows, top_rows, key_rows = [], [], []
    def evaluate_one(name, pred, fraction, step):
        if pred.shape != x.shape or not np.isfinite(pred).all() or pred.min() < 0:
            raise ValueError('Invalid held-out prediction')
        _, predicted_order = differential(pred, x, np.ones(len(panel), dtype=bool))
        metrics, moments = evaluate(pred, truth, x, true_order, predicted_order, top_n=TOP_N)
        metrics.update(effect_rank_metrics(moments['observed_delta'], moments['predicted_delta']))
        fi = panel.functional.to_numpy()
        metrics['functional_delta_RMSE'] = np.sqrt(np.mean((moments['observed_delta'][fi] - moments['predicted_delta'][fi]) ** 2))
        info = dict(method=name, fraction=fraction, step=step)
        rows.append(dict(**info, **metrics))
        g = pd.concat([panel[['gene_id', 'gene_symbol']], pd.DataFrame(moments)], axis=1)
        g['observed_detection'] = np.mean(truth > 0, axis=0)
        g['predicted_detection'] = np.mean(pred > 0, axis=0)
        folder = out / 'gene_predictions'; folder.mkdir(exist_ok=True)
        g.to_csv(folder / f'{name}.tsv.gz', sep='\t', index=False)
        keys = g[g.gene_symbol.isin(['Gata3', 'Id2', 'Il1rl1', 'Il13', 'Il5', 'Areg', 'Nfkbia', 'Mki67'])].copy()
        keys['method'] = name; key_rows.append(keys)
        for n in TOP_N:
            idx = true_order[:n]
            record = dict(**info, n=n)
            for target in ['mean', 'delta']:
                obs, pp = moments[f'observed_{target}'][idx], moments[f'predicted_{target}'][idx]
                record.update({f'{target}_RMSE': np.sqrt(np.mean((pp - obs) ** 2)),
                               f'{target}_R2': r2(obs, pp), f'{target}_Pearson': correlation(obs, pp)})
            top_rows.append(record)
        print(f'SCORE {name}: delta_RMSE={metrics["delta_RMSE"]:.6f} delta_R2={metrics["R2_delta_all"]:.4f}', flush=True)
    initial = source_head(cfg)
    zero = task.restore_head(initial, parent, len(panel), torch.device('cpu'))
    evaluate_one('zero_shot', x + task.predict_delta(zero, h, x, 128, torch.device('cpu')), 0, 0)
    evaluate_one('control_copy', x, 0, 0)
    for fraction in cfg['fractions']:
        name = label(fraction)
        with np.load(data / f'{name}.npz') as values:
            delta, mean = values['delta'], values['target_mean']
        evaluate_one(f'{name}_mean_delta', np.maximum(x + delta, 0).astype(np.float32), fraction, 0)
        evaluate_one(f'{name}_target_mean', np.broadcast_to(mean.astype(np.float32), x.shape), fraction, 0)
        for step in cfg['save_steps']:
            state = torch.load(out / 'checkpoints' / f'{name}_step{step}.pt', map_location='cpu', weights_only=False)
            model = task.restore_head(state, parent, len(panel), torch.device('cpu'))
            pred = x + task.predict_delta(model, h, x, 128, torch.device('cpu'))
            evaluate_one(f'{name}_step{step}', pred, fraction, step)
            if fraction == .05 and step == cfg['steps']:
                np.save(out / '05pct_step50_predicted_log1p.npy', pred)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(top_rows).to_csv(out / 'topn_expression_metrics.csv', index=False)
    pd.concat(key_rows).to_csv(out / 'key_gene_predictions.csv', index=False)
    task.atomic_json(out / 'verification.json', dict(passed=True, same_heldout_cells_for_all_methods=True,
        nested_disjoint_split=True, fitting_uses_adaptation_only=True, all_heads_locked_before_scoring=True,
        encoder_updates=0, original_Lung_standardization_unchanged=True, original_head_and_results_unchanged=True,
        heldout_counts=manifest['test_counts'], biological_replication='same pooled libraries; not new animal validation'))
    report(cfg, scores, manifest)


def report(cfg, scores, manifest):
    out = ROOT / cfg['result_directory']
    columns = ['method', 'delta_RMSE', 'R2_delta_all', 'Pearson_delta', 'R2_mean_all',
               'functional_delta_RMSE', 'DEG_overlap_100', 'effect_rank_overlap_100']
    lines = ['# v3：少量肝细胞继续适配', '',
        '保持FM冻结，原肺训练100epoch的v3预测头继续更新全部head参数；保留原肺标准化，重置AdamW，lr3e-4，固定50步。',
        '分条件以seed42固定80%肝细胞作相同Holdout；1/5/10%为总肝细胞比例，训练集合嵌套且均在剩余20%内。',
        f'Holdout={manifest["test_counts"]}；Adaptation={manifest["adaptation_counts"]}。',
        '只使用Adaptation PBS/IL33估计目标平均Δ，复用原delta_loss；无验证、无测试选步，预先固定5/20/50步全部报告。',
        '原11,537基因panel、归一化和FM特征均不改变。少量肝数据的平均Δ与IL33均值基线使用相同训练细胞。',
        '整体和TopN比较均是群体均值，不是单细胞配对反事实；表达与变化RMSE相同，R²/Pearson不同。', '',
        '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in scores[columns].itertuples(index=False, name=None):
        cells = [row[0]] + ['NA' if pd.isna(v) else f'{v:.6f}' for v in row[1:]]
        lines.append('| ' + ' | '.join(cells) + ' |')
    table = scores.set_index('method')
    zero_rmse = table.loc['zero_shot', 'delta_RMSE']
    lines += ['', '## 实际结果', '',
        '5%和10%肝监督可快速纠正平均响应；1%本轮反而增加留出误差。相同Holdout上：']
    for fraction in cfg['fractions']:
        name = label(fraction)
        final = table.loc[f'{name}_step{cfg["steps"]}']
        lines.append(f"{name}固定50步：ΔRMSE={final.delta_RMSE:.6f}，"
                     f"相对zero-shot误差降低{(1 - final.delta_RMSE / zero_rmse) * 100:.2f}%，"
                     f"ΔR²={final.R2_delta_all:.6f}，ΔPearson={final.Pearson_delta:.6f}。")
    keys = pd.read_csv(out / 'key_gene_predictions.csv')
    selected = keys[keys.gene_symbol.isin(['Gata3', 'Id2', 'Il1rl1', 'Nfkbia'])]
    pivot = selected.pivot(index='gene_symbol', columns='method', values='predicted_delta')
    truth = selected[selected.method.eq('zero_shot')].set_index('gene_symbol').observed_delta
    lines += ['', '## 关键基因平均Δ', '', '| 基因 | 真实Holdout | Zero-shot | 5%/50步 | 10%/50步 |',
              '|---|---:|---:|---:|---:|']
    for gene in pivot.index:
        lines.append(f'| {gene} | {truth.loc[gene]:.4f} | {pivot.loc[gene, "zero_shot"]:.4f} | '
                     f'{pivot.loc[gene, "05pct_step50"]:.4f} | {pivot.loc[gene, "10pct_step50"]:.4f} |')
    lines += ['', '5%/10%肝IL33均值基线的ΔRMSE分别为'
              f'{table.loc["05pct_target_mean", "delta_RMSE"]:.6f}/'
              f'{table.loc["10pct_target_mean", "delta_RMSE"]:.6f}，仍低于微调头，'
              '但它们对所有细胞输出同一向量、预测方差为零。不能据此声称恢复单细胞异质性。', '',
              '## 解释边界', '',
        '这是同一肝混池文库内的有监督目标域适配，已不属于肺→肝zero-shot；不是独立动物验证。单split/优化seed，不能评价稳定性。',
        '所有对比在同一个新Holdout上；不能把其指标与历史全肝测试直接当作同一分母。',
        '若少量肝均值基线也接近或优于微调头，只能说明目标域群体响应容易校准，不能证明FM学到了额外单细胞规律。',
        '连续偏移仍可能填满原PBS的零值；均值改善不保证检测率或细胞间分布改善。', '',
        '复现：`python scratch/src/il33_liver_fewshot_v3.py all`；prepare/train/score可分阶段执行。', '']
    (out / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'score', 'report', 'all'])
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    if args.stage in ['prepare', 'all']:
        prepare(cfg)
    if args.stage in ['train', 'all']:
        train(cfg)
    if args.stage in ['score', 'all']:
        score(cfg)
    if args.stage == 'report':
        _, out, manifest = ready(cfg)
        report(cfg, pd.read_csv(out / 'comparison.csv'), manifest)


if __name__ == '__main__':
    main()
