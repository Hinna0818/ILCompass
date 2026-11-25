"""Frozen Scratch v1 PBS -> IL33 delta: independent Lung/Liver internal benchmarks."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ilc2_mpl')

import numpy as np
import pandas as pd
import torch
from torch import nn

from il33_data import CONFIG, METHODS, ROOT, configuration, sources, paths, prepare, encode, dataset, task
from il33_lung_metrics import effect_rank_metrics


def head(method, features, genes, cfg, device):
    model = (task.DirectDeltaHead(torch.tensor(np.array(features)), genes, cfg['hidden_size'])
             if method == 'Scratch_v1' else task.ExpressionDeltaMLP(genes, cfg['hidden_size']))
    return model.to(device)


def run_path(cfg, tissue, method, seed):
    out = paths(cfg, tissue)[1] / method / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def fingerprint(cfg, tissue):
    data, _ = paths(cfg, tissue)
    return task.file_hash(data / 'prepare_manifest.json') + ':' + task.file_hash(data / 'encoding.json')


def train_one(cfg, tissue, method, seed, device):
    data, _ = paths(cfg, tissue)
    out = run_path(cfg, tissue, method, seed)
    signature = fingerprint(cfg, tissue)
    if (out / 'complete.json').exists():
        complete = json.loads((out / 'complete.json').read_text())
        if complete['fingerprint'] != signature or complete['best_sha256'] != task.file_hash(out / 'best.pt'):
            raise ValueError('Completed run/data changed')
        print(f'REUSE {tissue}/{method}/seed{seed}', flush=True); return
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    train = dataset(cfg, tissue, 'train', method)
    validation = dataset(cfg, tissue, 'validation', method)
    panel = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
    with np.load(data / 'train/statistics.npz') as f:
        statistics = dict(f)
    group = f'GSE189285_{tissue}'
    dc = dict(cfg, backgrounds=[group])
    m, y, h = train
    controls = m[m.condition.eq('PBS')]
    x = np.asarray(y[m.condition.eq('PBS').to_numpy()])
    model = head(method, h, len(panel), cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['lr_factor'],
        patience=cfg['lr_patience'], threshold=cfg['minimum_relative_improvement'],
        threshold_mode='rel', min_lr=cfg['minimum_learning_rate'])
    history, best, reference, stale, start = [], float('inf'), float('inf'), 0, 1
    if (out / 'last.pt').exists():
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if state['fingerprint'] != signature:
            raise ValueError('Resume configuration/data changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, best, reference, stale, start = (state[k] for k in ['history', 'best', 'reference', 'stale', 'next_epoch'])
    targets = {group: torch.tensor(statistics[group + '__delta'], dtype=torch.float32, device=device)}
    for epoch in range(start, cfg['max_epochs'] + 1):
        if epoch > cfg['min_epochs'] and stale >= cfg['early_stop_patience']:
            break
        batch_loss, grad = task.fit_epoch(model, optimizer, controls, h, x, targets, dc, rng, device)
        _, tr = task.score_model(model, train, dc, panel, statistics, device)
        _, va = task.score_model(model, validation, dc, panel, statistics, device)
        score = va['macro_delta_MSE']
        if not np.isfinite(score):
            raise ValueError('Nonfinite Validation error')
        scheduler.step(score)
        history.append(dict(epoch=epoch, batch_loss=batch_loss, grad_norm=grad,
            train_delta_MSE=tr['macro_delta_MSE'], validation_delta_MSE=score,
            validation_delta_Pearson=va['macro_Pearson_delta'], lr=optimizer.param_groups[0]['lr']))
        state = dict(model=model.state_dict(), epoch=epoch, fingerprint=signature, method=method,
                     seed=seed, tissue=tissue, input_dimensions=h.shape[1], output_genes=len(panel))
        if score < best:
            best = score
            task.atomic_torch(out / 'best.pt', dict(state, validation=va))
        if score < reference * (1 - cfg['minimum_relative_improvement']):
            reference, stale = score, 0
        else:
            stale += 1
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        task.atomic_torch(out / 'last.pt', dict(state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            history=history, next_epoch=epoch + 1, best=best, reference=reference, stale=stale))
        if epoch == 1 or epoch % 10 == 0:
            print(f'{tissue}/{method}/seed{seed} epoch={epoch} train={tr["macro_delta_MSE"]:.6g} '
                  f'val={score:.6g} best={best:.6g} stale={stale}', flush=True)
    selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    task.atomic_json(out / 'complete.json', dict(fingerprint=signature, epochs=len(history),
        best_epoch=selected['epoch'], best_validation=selected['validation'],
        best_sha256=task.file_hash(out / 'best.pt'), parameters=sum(p.numel() for p in model.parameters()),
        stop_reason='early_stop' if stale >= cfg['early_stop_patience'] else 'max_epochs'))


class MeanTarget(nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))

    def forward(self, x):
        return self.mean - x


def extend_deg_metrics(row, folder, group, top_n=(20, 50, 100, 200)):
    genes = pd.read_csv(folder / f'{group}_gene_predictions.tsv.gz', sep='\t')
    de = pd.read_csv(folder / f'{group}_observed_DEG.tsv.gz', sep='\t')
    eligible = de.qvalue.lt(.05) & de.delta.abs().ge(1e-8)
    ranked = de.loc[eligible].assign(abs_delta=lambda x: x.delta.abs()).sort_values(
        ['pvalue', 'abs_delta'], ascending=[True, False], kind='stable')
    order = de.index.get_indexer(ranked.index)
    dp, dt = genes.predicted_delta.to_numpy(), genes.observed_delta.to_numpy()
    predicted_effect = np.flatnonzero(np.abs(dp) >= 1e-8)
    predicted_effect = predicted_effect[np.argsort(-np.abs(dp[predicted_effect]), kind='stable')]
    for n in top_n:
        idx = order[:n]
        row[f'TopDEG_delta_RMSE_{n}'] = float(np.sqrt(np.mean((dp[idx] - dt[idx]) ** 2))) if len(idx) else None
        row[f'DEG_effect_overlap_{n}'] = len(set(idx) & set(predicted_effect[:n])) / len(idx) if len(idx) else None
    row['observed_significant_DEGs'] = int(eligible.sum())
    return row


def test(cfg, device):
    # All validation-selected heads are fixed before loading any Test expression.
    locks = {}
    for tissue in sources(cfg):
        for method in METHODS:
            for seed in cfg['seeds']:
                out = run_path(cfg, tissue, method, seed)
                complete = json.loads((out / 'complete.json').read_text())
                if complete['fingerprint'] != fingerprint(cfg, tissue) or complete['best_sha256'] != task.file_hash(out / 'best.pt'):
                    raise ValueError('Test closed: training incomplete or checkpoint changed')
                locks[f'{tissue}/{method}/seed{seed}'] = complete
    base = ROOT / cfg['result_directory']
    if (base / 'test_protocol.json').exists() and json.loads((base / 'test_protocol.json').read_text()) != locks:
        raise ValueError('Test already evaluated with different model selections')
    task.atomic_json(base / 'test_protocol.json', locks)
    rows = []
    for tissue in sources(cfg):
        data, result = paths(cfg, tissue)
        panel = pd.read_csv(data / 'gene_panel.tsv', sep='\t')
        with np.load(data / 'train/statistics.npz') as f:
            statistics = dict(f)
        group = f'GSE189285_{tissue}'; dc = dict(cfg, backgrounds=[group])
        train = dataset(cfg, tissue, 'train', 'ExpressionMLP')
        m, y, _ = train
        mean_target = np.asarray(y[m.condition.eq('IL33').to_numpy()]).mean(0, dtype=np.float64)
        for method in ('control_copy', 'train_mean_delta', 'train_target_mean', *METHODS):
            data_method = method if method in METHODS else 'ExpressionMLP'
            test_data = dataset(cfg, tissue, 'test', data_method)
            for seed in cfg['seeds'] if method in METHODS else [-1]:
                baseline = None; model = None
                if method in METHODS:
                    out = run_path(cfg, tissue, method, seed)
                    c = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
                    model = head(method, np.zeros((2, c['input_dimensions']), dtype=np.float32), len(panel), cfg, device)
                    model.load_state_dict(c['model'])
                    best_epoch = c['epoch']
                else:
                    out = result / method; out.mkdir(exist_ok=True)
                    best_epoch = None
                    if method == 'train_target_mean':
                        model = MeanTarget(mean_target).to(device)
                    else:
                        baseline = 'control_copy' if method == 'control_copy' else 'background_mean_delta'
                folder = out / 'test'
                scores, _ = task.score_model(model, test_data, dc, panel, statistics, device,
                    full_de=True, save_to=folder, baseline=baseline)
                row = dict(tissue=tissue, method=method, seed=seed, best_epoch=best_epoch, **scores[0])
                rows.append(extend_deg_metrics(row, folder, group))
                print(f'TEST {tissue}/{method}/seed{seed} delta_RMSE={row["delta_RMSE"]:.6g} '
                      f'Pearson_delta={row["Pearson_delta"]} DEG@100={row["DEG_overlap_100"]}', flush=True)
    pd.DataFrame(rows).to_csv(base / 'test_metrics.csv', index=False)
    report(cfg)


def report(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = ROOT / cfg['result_directory']
    metrics = pd.read_csv(base / 'test_metrics.csv')
    effect_rows = []
    functional_rows = []
    for row in metrics.itertuples(index=False):
        folder = base / row.tissue / row.method
        if row.method in METHODS:
            folder = folder / f'seed{row.seed}'
        genes = pd.read_csv(folder / 'test' / f'GSE189285_{row.tissue}_gene_predictions.tsv.gz', sep='\t')
        effect_rows.append(effect_rank_metrics(genes.observed_delta.to_numpy(), genes.predicted_delta.to_numpy()))
        panel = pd.read_csv(paths(cfg, row.tissue)[0] / 'gene_panel.tsv', sep='\t')
        f = genes.merge(panel[['gene_id', 'gene_symbol', 'functional']], on='gene_id', validate='one_to_one')
        f = f[f.functional].copy()
        f['tissue'], f['method'], f['seed'] = row.tissue, row.method, row.seed
        functional_rows.append(f)
    for key in effect_rows[0]:
        metrics[key] = [row[key] for row in effect_rows]
    metrics.to_csv(base / 'test_metrics.csv', index=False)
    pd.concat(functional_rows, ignore_index=True).to_csv(base / 'functional_gene_predictions.csv', index=False)
    cols = ['delta_RMSE', 'Pearson_delta', 'R2_delta_all', 'functional__delta_RMSE',
            'DEG_overlap_20', 'DEG_overlap_50', 'DEG_overlap_100', 'DEG_overlap_200',
            'DEG_direction_100', 'TopDEG_delta_RMSE_100', 'DEG_effect_overlap_100',
            *[f'effect_rank_overlap_{n}' for n in (20, 50, 100, 200)]]
    summary = metrics.groupby(['tissue', 'method'], sort=False)[cols].agg(['mean', 'std'])
    summary.to_csv(base / 'seed_summary.csv')
    lines = ['# Scratch v1：同组织内IL-33表达变化预测', '',
        '肺和肝分别独立建模；GSE189285文库内、按条件分层70/15/15，split seed42。研究不在预训练语料中；'
        '下游仍是同文库细胞留出，不能视为独立动物或文库外验证。', '',
        'PBS原始counts → 冻结预训练词表/Train-only中位数/全非零排序 → v1 epoch20 FP32 mean embedding（192维）'
        ' → train-PBS特征标准化 → IL33条件编码+128维GELU预测头 → 固定panel Δ。'
        'ExpressionMLP使用完整panel log1p(CP10K)+相同刺激编码，128维GELU预测Δ。两者均从control-copy初始化。', '',
        '表达输出为max(PBS+Δ,0)。训练loss为每批PBS预测有效Δ的均值与Train IL33均值−Train PBS均值之间的基因平均MSE；'
        '不随机配对PBS和IL33细胞。没有逐细胞真实对应，指标主要验证群体均值响应，不证明单细胞反事实准确性。', '',
        '每组织固定输出panel仅用Train PBS/IL33检测率≥1%或预定义67功能基因选择。'
        '输入与输出归一化分开：encoder为log2(1+CP10K/预训练中位数)，预测目标为log1p(CP10K)，分母都使用完整原测量GEX。', '',
        'AdamW lr3e-4、batch128、最多200 epoch、至少30 epoch、patience25，Validation平均Δ-MSE选择best；'
        '三个seed，全部模型锁定后仅评分Test。基线包含control-copy、Train平均Δ（输出非负截断）和Train IL33平均表达。', '',
        '## 本次结果判断', '']
    for tissue in sources(cfg):
        a = summary.loc[(tissue, 'Scratch_v1'), ('delta_RMSE', 'mean')]
        b = summary.loc[(tissue, 'ExpressionMLP'), ('delta_RMSE', 'mean')]
        c = summary.loc[(tissue, 'train_target_mean'), ('delta_RMSE', 'mean')]
        lines.append(f'{tissue}：v1 Δ-RMSE={a:.6f}，表达MLP={b:.6f}，Train IL33均值基线={c:.6f}。'
                     f'v1相对MLP误差变化{(a/b-1)*100:+.2f}%，相对均值基线为{a/c:.2f}倍；这些是描述性比较，不是动物级显著性检验。')
    lines += ['', '当前两种预测头数值接近，且未胜过Train IL33平均表达基线，尚无充分证据支持预训练带来实质性增益。'
              '按p值排名的DEG与按平均变化幅度排名差异明显，需同时看分布/检测率，而不能仅看均值相关性。', '']
    counts = pd.read_csv(base / 'split_counts.csv')
    lines += ['## 细胞划分', '', '| 组织 | split | 条件 | 细胞数 |', '|---|---|---|---:|']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in counts.itertuples(index=False, name=None)]
    lines.append('')
    for tissue in sources(cfg):
        data, _ = paths(cfg, tissue)
        manifest = json.loads((data / 'prepare_manifest.json').read_text())
        lines += [f'## {tissue}', '', f"固定panel：{manifest['output_genes']}基因，"
                  f"{manifest['functional_genes']}个功能候选；{manifest['in_encoder_vocab']}个在v1词表中。", '',
                  '| 方法 | Δ-RMSE ↓ | Δ-Pearson ↑ | 功能Δ-RMSE ↓ | DEG overlap@100 ↑ | 幅度排名overlap@100 ↑ |',
                  '|---|---:|---:|---:|---:|---:|']
        for method in metrics.method.unique():
            vals = []
            for col in ['delta_RMSE', 'Pearson_delta', 'functional__delta_RMSE', 'DEG_overlap_100', 'effect_rank_overlap_100']:
                mean, std = summary.loc[(tissue, method), [(col, 'mean'), (col, 'std')]]
                vals.append('—' if not np.isfinite(mean) else f'{mean:.5f} ± {std:.5f}' if np.isfinite(std) else f'{mean:.5f}')
            lines.append('| ' + method + ' | ' + ' | '.join(vals) + ' |')
        lines.append('')
    training = []
    for tissue in sources(cfg):
        for method in METHODS:
            for seed in cfg['seeds']:
                complete = json.loads((run_path(cfg, tissue, method, seed) / 'complete.json').read_text())
                training.append(dict(tissue=tissue, method=method, seed=seed,
                    epochs=complete['epochs'], best_epoch=complete['best_epoch'], parameters=complete['parameters'],
                    validation_delta_MSE=complete['best_validation']['macro_delta_MSE']))
    pd.DataFrame(training).to_csv(base / 'training_summary.csv', index=False)
    lines += ['## DEG指标与限制', '',
        '复用项目differential：细胞级Mann–Whitney U、固定panel内BH校正，q<0.05且非零平均变化的基因，按p值升序、绝对Δ降序排序。'
        'DEG overlap@N是真实/预测Top-N显著DEG交集除以真实Top-N实际基因数；不是全panel基因表达相关性。'
        '另存Top20/50/100/200的Δ误差、方向一致性。effect_rank_overlap_N在真实和预测两侧均按|平均Δ|排序，无显著性过滤；'
        'DEG_effect_overlap_N则比较真实p值排名与预测幅度排名，三种overlap不可混用。',
        '这是描述性细胞级差异分析；每组织每条件仅一个混池文库，无独立动物重复，不能作为动物级差异检验。'
        '三个seed标准差仅反映预测头优化随机性。Test两条件的细胞数不同，预测分布与实测分布的检验功效也不同。'
        '预测表达由同批PBS推导，与PBS并非独立样本，预测侧p/q值只作描述性排序，不能作为新的显著性证据。',
        'train_target_mean直接输出Train IL33平均表达；这是均值指标的强基线，但其细胞间预测方差为零。'
        '当前平均Δ监督本身不识别单细胞反应异质性，均值指标好不能证明分布恢复。',
        '仅一种刺激，刺激编码恒定；不能评价未见刺激泛化。Scratch头与表达MLP输入维度/参数量不同，差异不能单独归因于预训练。', '',
        '## 文件与运行', '',
        '`test_metrics.csv`含各seed全部指标；`seed_summary.csv`为均值/SD。各组织/方法/seed目录保存history、best/last、completion和Test表达变化/DEG/逐基因结果。',
        '`deg_overlap.png`为显著性排名，`effect_overlap.png`为两侧均按平均变化幅度排名；`functional_gene_predictions.csv`保存67候选的预测。',
        '专项检测率诊断见[deg_rank_diagnosis.md](deg_rank_diagnosis.md)，由`python main/diagnose_scratch_il33_deg_ranks.py`生成。',
        '`data`目录保留来源、固定split、panel、Train-only统计和仅PBS的冻结features；源h5ad、预训练模型未修改。', '',
        '运行：`python scratch/src/il33_benchmark.py all --device cuda`；支持prepare/encode/train/test/report阶段。已有完成结果会复用，未完成训练从last.pt恢复。', '']
    (base / 'report.md').write_text('\n'.join(lines))
    for prefix, label, filename in [('DEG_overlap', 'P-value-ranked DEG overlap', 'deg_overlap.png'),
                                     ('effect_rank_overlap', 'Absolute mean-delta rank overlap', 'effect_overlap.png')]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, tissue in zip(axes, sources(cfg)):
            rows = metrics[metrics.tissue.eq(tissue)]
            for method in rows.method.unique():
                m = rows[rows.method.eq(method)]
                ax.plot([20, 50, 100, 200], [m[f'{prefix}_{n}'].mean() for n in [20, 50, 100, 200]], 'o-', label=method)
            ax.set(title=tissue, xlabel='Top N genes', ylabel=label, ylim=(0, 1.03))
            if prefix == 'DEG_overlap':
                upper = metrics[[f'{prefix}_{n}' for n in (20, 50, 100, 200)]].max().max()
                ax.set_ylim(0, max(.05, float(upper) * 1.15))
            ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.suptitle('GSE189285: within-tissue held-out cells; mean over head seeds')
        fig.tight_layout(); fig.savefig(base / filename, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tissue in zip(axes, sources(cfg)):
        for method in METHODS:
            for seed in cfg['seeds']:
                history = pd.read_csv(run_path(cfg, tissue, method, seed) / 'history.csv')
                ax.plot(history.epoch, history.validation_delta_MSE, alpha=.65,
                        label=f'{method}/seed{seed}')
        ax.set(title=tissue, xlabel='Epoch', ylabel='Validation mean-delta MSE', yscale='log')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(base / 'training_curves.png', dpi=180); plt.close(fig)
    print(f'REPORT {base / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'encode', 'train', 'test', 'report', 'all'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args(); cfg = configuration()
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(args.device)
    if args.stage == 'report':
        report(cfg); return
    if args.stage in ('prepare', 'all'):
        prepare(cfg)
    if args.stage in ('encode', 'all'):
        encode(cfg)
    if args.stage in ('train', 'test', 'all') and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.stage in ('train', 'all'):
        for tissue in sources(cfg):
            for method in METHODS:
                for seed in cfg['seeds']:
                    train_one(cfg, tissue, method, seed, device)
    if args.stage in ('test', 'all'):
        test(cfg, device)


if __name__ == '__main__':
    main()
