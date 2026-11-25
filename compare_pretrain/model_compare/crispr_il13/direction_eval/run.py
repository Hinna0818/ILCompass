"""Locked Il13 masked-value direction benchmark. No fitting, target selection or resampling."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src/downstream/geneko'))
from evaluate_il13_screen_ko import Inputs, screen_effects, shared
from pilot_il13_masked_expression import predict, logcp10k
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

OUT = ROOT / 'results/GSE262028_IL13_direction_eval_v1'
MODELS = ['raw', 'lora']
SCORES = ['delta_target_native', 'excess_delta_native', 'delta_random_mean_native']


def direction_metrics(predicted, experiment, epsilon):
    """Each gene has one vote. Numerical abstentions stay in the accuracy denominator."""
    x, y = np.asarray(predicted, dtype=float), np.asarray(experiment, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite gene effects; do not silently drop genes')
    keep = y != 0
    x, y = x[keep], y[keep]
    sign = np.where(np.abs(x) > epsilon, np.sign(x), 0)
    truth = np.sign(y)
    correct, called = sign == truth, sign != 0
    n = len(x)
    up, down = truth > 0, truth < 0
    r_up = float(correct[up].mean()) if up.any() else np.nan
    r_down = float(correct[down].mean()) if down.any() else np.nan
    can_correlate = n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
    return dict(n_genes=n, n_experimental_zero=int((~keep).sum()), n_exp_up=int(up.sum()),
        n_exp_down=int(down.sum()), n_pred_up=int((sign > 0).sum()), n_pred_down=int((sign < 0).sum()),
        n_abstain=int((~called).sum()), n_correct=int(correct.sum()),
        accuracy=float(correct.mean()) if n else np.nan,
        called_fraction=float(called.mean()) if n else np.nan,
        accuracy_called_only=float(correct[called].mean()) if called.any() else np.nan,
        recall_up=r_up, recall_down=r_down, balanced_accuracy=(r_up+r_down)/2,
        always_up_accuracy=float(up.mean()) if n else np.nan,
        always_down_accuracy=float(down.mean()) if n else np.nan,
        majority_accuracy=float(max(up.sum(), down.sum())/n) if n else np.nan,
        spearman_signed_effect=float(spearmanr(x, y).statistic) if can_correlate else np.nan,
        pearson_signed_effect=float(pearsonr(x, y).statistic) if can_correlate else np.nan)


def signature(inputs, name, batch_size):
    files = [Path(__file__), ROOT / 'src/downstream/geneko/pilot_il13_masked_expression.py',
        ROOT / 'src/downstream/geneko/evaluate_il13_screen_ko.py',
        ROOT / 'compare_pretrain/model_compare/shared.py', ROOT / 'src/genecompass_compat.py',
        ROOT / shared.MODELS['raw'] / 'pytorch_model.bin', ROOT / shared.MODELS['raw'] / 'config.json',
        ROOT / 'compare_pretrain/genepass_lora/config.json']
    if name == 'lora':
        files += [ROOT / shared.MODELS[name], ROOT / 'compare_pretrain/genepass_lora/model.py']
    return dict(model=name, batch_size=batch_size, precision='FP32, TF32 off',
        manifest_sha256=inputs.design['manifest_sha256'], pair_ids=inputs.pairs.pair_id.tolist(),
        source_sha256=inputs.design['source_sha256'], torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        hashes={str(p.relative_to(ROOT)): shared.file_hash(p) for p in files})


def check_signature(inputs, name, folder, batch_size, create=False):
    expected = signature(inputs, name, batch_size)
    path = folder / 'run_signature.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'Cached inputs/code/checkpoint changed: {path}. Use a separate result directory/version.')
    elif create:
        shared.write_json(path, expected)
    else:
        raise FileNotFoundError(path)


def tokens_for(inputs, pairs):
    return np.unique(pairs[['target_token_id', *inputs.random_columns]].to_numpy(dtype=np.int64))


def run_model(inputs, name, folder, batch_size):
    cache = folder / 'cells'
    cache.mkdir(parents=True, exist_ok=True)
    check_signature(inputs, name, folder, batch_size, create=True)
    pending = [row for row in inputs.groups if not (cache / f'{row:06d}.npz').exists()]
    print(f'{name}: pending {len(pending)}/{len(inputs.groups)} reference cells', flush=True)
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. --check-only can run on CPU.')
    model = shared.load(name).float().eval().requires_grad_(False)
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    start = time.monotonic()
    done_variants = 0
    for k, row in enumerate(pending):
        n = int(inputs.lengths[row])
        seq = (inputs.ids[row, :n].astype(np.int64), inputs.values[row, :n].copy())
        regenerated = inputs.sequence(inputs.counts[row].toarray().ravel()/inputs.scale)
        if not all(np.array_equal(a, b) for a, b in zip(seq, regenerated)):
            raise ValueError('Baseline differs from official frozen preprocessing')
        tokens = tokens_for(inputs, inputs.groups[row])
        baseline = float(predict(model, [seq], [None], inputs.readout, name)[0])
        # Record checks per cell: batch-size roundoff matters at these very small effects.
        sham = predict(model, [seq]*batch_size, [None]*batch_size, inputs.readout, name)
        sham_error = float(np.max(np.abs(sham-baseline)))
        if sham_error > 1e-4:
            raise ValueError(f'Sham repeat mismatch: {sham_error}')
        if k == 0:
            altered = (seq[0].copy(), seq[1].copy())
            q = int(np.flatnonzero(seq[0] == inputs.readout)[0])
            altered[1][q] = 999.
            hidden = predict(model, [seq, altered], [None, None], inputs.readout, name)
            leak_error = float(abs(hidden[0]-hidden[1]))
            if leak_error > 1e-6:
                raise ValueError('Hidden Il13 expression leaked into query')
            shared.write_json(folder / 'engineering_check.json', dict(cell_row=int(row),
                hidden_value_invariance_max_abs=leak_error, cls_offset=1,
                mask='Il13 value=-1; ID retained; target slot PAD/value0/attention0; no reorder/refill'))
        after = []
        for left in range(0, len(tokens), batch_size):
            tt = tokens[left:left+batch_size]
            after.extend(predict(model, [seq]*len(tt), list(tt), inputs.readout, name))
        final = cache / f'{row:06d}.npz'
        temp = cache / f'{row:06d}.tmp.npz'
        np.savez_compressed(temp, cell_row=row, tokens=tokens, baseline=baseline,
                            after=np.asarray(after), sham_error=sham_error)
        temp.replace(final)
        done_variants += len(tokens)
        if k == 0 or (k+1) % 20 == 0 or k+1 == len(pending):
            elapsed = time.monotonic()-start
            print(f'{name}: cells={len(inputs.groups)-len(pending)+k+1}/{len(inputs.groups)} '
                  f'new_variants={done_variants} elapsed={elapsed/60:.1f}min '
                  f'ETA_by_cells={elapsed/(k+1)*(len(pending)-k-1)/60:.1f}min', flush=True)
    del model
    torch.cuda.empty_cache()


def summarize(inputs, out, effects, batch_size, smoke):
    # Keep the original full-library denominator for all counterfactuals.
    ref = ROOT / inputs.design['config']['reference_directory']
    a = ad.read_h5ad(ref / 'GSE126924_Lung_IL13positive_ILC2_1864.h5ad')
    totals = np.asarray(a.X.sum(axis=1)).ravel()
    del a
    scale = inputs.scale[inputs.column_by_token[inputs.readout]]
    tables, errors = {}, []
    for name in MODELS:
        check_signature(inputs, name, out / name, batch_size)
        pieces = []
        for row, pairs in inputs.groups.items():
            path = out / name / 'cells' / f'{row:06d}.npz'
            if not path.exists():
                raise FileNotFoundError(f'Inference incomplete: {path}')
            with np.load(path) as z:
                tokens, after = z['tokens'], z['after']
                if int(z['cell_row']) != row or not np.array_equal(tokens, tokens_for(inputs, pairs)):
                    raise ValueError('Cached cell/target identity mismatch')
                before, error = float(z['baseline']), float(z['sham_error'])
            errors.append(error)
            d = pairs[['pair_id', 'target_gene_id', 'target_symbol', 'cell_row', 'cell_id',
                       'target_rank', 'Il13_rank', 'target_in_token_top_half']].copy()
            for unit, values, base in [('native', after, before),
                    ('logCP10K', logcp10k(after, scale*10000/totals[row]),
                     float(logcp10k(np.array([before]), scale*10000/totals[row])[0]))]:
                mapping = dict(zip(tokens, values))
                d[f'pred_before_{unit}'] = base
                d[f'delta_target_{unit}'] = pairs.target_token_id.map(mapping)-base
                for j in range(3):
                    d[f'delta_random{j}_{unit}'] = pairs[f'random{j}_token_id'].map(mapping)-base
                random_cols = [f'delta_random{j}_{unit}' for j in range(3)]
                d[f'delta_random_mean_{unit}'] = d[random_cols].mean(axis=1)
                d[f'excess_delta_{unit}'] = d[f'delta_target_{unit}']-d[f'delta_random_mean_{unit}']
            d['model'] = name
            pieces.append(d)
        tables[name] = pd.concat(pieces).sort_values('pair_id').reset_index(drop=True)
        if not np.isfinite(tables[name].filter(regex='^(delta|excess|pred)_').to_numpy()).all():
            raise ValueError('Incomplete/nonfinite model predictions')
    pd.testing.assert_series_equal(tables['raw'].pair_id, tables['lora'].pair_id)
    # One common threshold for both models, based solely on numerical repeat checks.
    epsilon = max(1e-6, 10*max(errors))
    gene_tables = []
    for name, d in tables.items():
        d.to_csv(out / f'{name}_cell_predictions.csv.gz', index=False)
        grouped = d.groupby(['target_gene_id', 'target_symbol'], sort=True)
        cols = [f'{score}_{unit}' for unit in ['native', 'logCP10K']
                for score in ['delta_target', 'delta_random_mean', 'excess_delta']]
        g = grouped[cols].mean()
        g['n_cells'] = grouped.size()
        for score in SCORES:
            d[score+'_up'] = d[score] > epsilon
            d[score+'_down'] = d[score] < -epsilon
            g[score+'_fraction_up'] = grouped[score+'_up'].mean()
            g[score+'_fraction_down'] = grouped[score+'_down'].mean()
            g[score+'_sign'] = np.where(g[score].abs() > epsilon, np.sign(g[score]), 0).astype(int)
        g = g.reset_index().merge(effects, left_on='target_gene_id', right_on='ensembl_gene_id',
                                 validate='one_to_one')
        if len(g) != len(inputs.targets):
            raise ValueError('Targets lost during screen merge')
        for screen in ['S17', 'S25']:
            for score in SCORES:
                g[f'{score}_correct_{screen}'] = (g[score+'_sign'] == np.sign(g[screen+'_effect'])) & g[screen+'_effect'].ne(0)
        g['model'] = name
        gene_tables.append(g)
    genes = pd.concat(gene_tables, ignore_index=True)
    genes.to_csv(out / 'gene_predictions.csv', index=False)
    records = []
    for name, g in genes.groupby('model', sort=False):
        concordant = g.screen_sign_agreement & g.S17_effect.ne(0) & g.S25_effect.ne(0)
        for scope, subset, endpoint in [
            ('all_S17', g, 'S17_effect'), ('all_S25', g, 'S25_effect'),
            ('concordant_primary', g[concordant], 'mean_signed_effect'),
            ('all_mean_secondary', g, 'mean_signed_effect')]:
            for score in SCORES:
                records.append(dict(model=name, scope=scope, score=score, epsilon_native=epsilon,
                    **direction_metrics(subset[score], subset[endpoint], epsilon)))
    metrics = pd.DataFrame(records)
    metrics.to_csv(out / 'direction_metrics.csv', index=False)
    summary = dict(smoke_only=smoke, targets=len(inputs.targets), cells=len(inputs.groups),
        target_cell_pairs=len(inputs.pairs), numerical_threshold_shared=epsilon,
        max_sham_native_error=max(errors), models=MODELS,
        screen_concordant_genes=int((effects.screen_sign_agreement & effects.S17_effect.ne(0) & effects.S25_effect.ne(0)).sum()))
    shared.write_json(out / 'summary.json', summary)
    rows = []
    for r in metrics[metrics.score.ne('delta_random_mean_native')].itertuples():
        rows.append(f'|{r.model}|{r.scope}|{r.score}|{r.n_correct}/{r.n_genes}|{r.accuracy:.1%}|'
                    f'{r.n_abstain}|{r.balanced_accuracy:.3f}|{r.spearman_signed_effect:.3f}|{r.pearson_signed_effect:.3f}|')
    report = f'''# Il13 masked-value direction evaluation{' — 工程小测试，不能用于总体结论' if smoke else ''}

实际完成 {len(inputs.targets)} 个目标、{len(inputs.groups)} 个细胞、{len(inputs.pairs):,} 个目标–细胞组合，raw/LoRA 使用完全相同的细胞及每组合3个随机基因。无训练，无真实KO表达谱拟合。

Il13 表达输入=-1，保留基因ID；目标槽位替换为PAD、value0、attention_mask0，其他位置不变，不重新排序、不补位。调用冻结模型原始 cls4value 头。
目标直接Δ = 目标移除后的Il13预测 − 未移除时的Il13预测；额外Δ = 直接Δ − 三个排名匹配随机移除的平均Δ。随机平均Δ也单独评估于 direction_metrics.csv。

每个基因先在自己的锁定细胞中取平均，再按基因等权评价；不按细胞数给高覆盖基因更大权重。两模型共用数值容差 {epsilon:.6g}（至少1e-6，且为全部实测sham误差上界的10倍）。绝对值不超过容差视为不确定，仍保留在总命中率分母；另报非零预测比例和仅确定预测的命中率。该容差不是生物学显著性阈值。

真实参照复用现有S17/S25处理：每条guide log2(P/N)减去该screen非靶向guide中位数，再取基因内guide中位数。正负分别表示偏向Il13报告阳性/阴性，不是Il13 RNA表达logFC。所有入选基因保留，没有按模型表现或screen效应强弱二次筛选。
两实验方向一致且非零的 {summary['screen_concordant_genes']} 个基因构成 concordant_primary；all_S17/all_S25各自独立报告；all_mean_secondary包含方向冲突基因，只作辅助。实验效应恰为0者不纳入方向分母。没有显著性或因果标签。

|模型|实验参照集合|预测量|命中/总数|命中率|不确定数|平衡命中率|有符号效应Spearman|有符号效应Pearson|
|---|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

请同时查看 direction_metrics.csv 中总猜上调、总猜下调、两者较优的多数类基线，以及随机扰动方向表现；方向命中率不能脱离正负标签比例解释。Spearman/Pearson比较预测有符号连续效应与实验有符号连续效应，并非直接计算二元方向标签的相关系数。直接Δ和额外Δ回答不同问题，不根据结果择优称为主指标。

局限：原始Il13 rank/存在性依然可见，value-only mask不等于训练时ID/value联合遮蔽；冻结proxy median预处理沿用旧流程；参考为肺ILC2而screen生物学背景不同；匹配随机基因可能有功能，不等于实验NT guide；不同目标复用细胞，不能视为独立动物重复。此前Tcf12/Maf试点已被查看，本轮为锁定全集的探索性验证，不是完全未见的外部测试。结果不证明因果调控或真实KO表达变化幅度。

逐细胞原始结果：raw_cell_predictions.csv.gz、lora_cell_predictions.csv.gz；逐基因结果：gene_predictions.csv；实验参照：screen_descriptive_effects.csv；配置和检查：audit_config.json、summary.json、raw/lora下run_signature.json及engineering_check.json。
'''
    (out / 'report.md').write_text(report)
    shared.write_json(out / 'complete.json', summary)
    print(metrics.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['both', *MODELS], default='both')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--smoke', action='store_true', help='Tcf12/Maf/Junb, first 2 eligible cells each; separate output')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    inputs = Inputs(smoke=False)
    if args.smoke:
        inputs.pairs = pd.concat([inputs.pairs[inputs.pairs.target_symbol.eq(g)].head(2)
                                  for g in ['Tcf12', 'Maf', 'Junb']]).sort_values('pair_id').copy()
        inputs.groups = dict(tuple(inputs.pairs.groupby('cell_row', sort=True)))
        inputs.targets = inputs.targets[inputs.targets.author_symbol.isin(inputs.pairs.target_symbol)].copy()
        if len(inputs.pairs) != 6 or len(inputs.targets) != 3:
            raise ValueError('Smoke targets unavailable')
    out = OUT.with_name(OUT.name+'_smoke') if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    effects, screen_provenance = screen_effects(inputs.targets)
    n_variants = sum(len(tokens_for(inputs, pairs)) for pairs in inputs.groups.values())
    audit = dict(protocol='fixed_slot_Il13_value_mask', targets=len(inputs.targets),
        cells=len(inputs.groups), target_cell_pairs=len(inputs.pairs), unique_deletions_per_model=n_variants,
        baseline_sequences_per_model=len(inputs.groups), extra_sham_sequences_per_cell=args.batch_size,
        batch_size=args.batch_size, smoke_only=args.smoke, mask_value=-1, precision='FP32',
        target_random_controls='locked design_v1; no resampling',
        output_directory=str(out), design_directory='compare_pretrain/model_compare/crispr_il13/design_v1',
        manifest_sha256=inputs.design['manifest_sha256'], screen_provenance=screen_provenance,
        primary_scope='screen concordant nonzero genes; direct and random-corrected delta reported separately',
        score_units='native log2(1+count/proxy_median); logCP10K secondary, fixed original library size',
        numeric_threshold='shared max(1e-6, 10*maximum sham error of either model across all cells)')
    for s in ['S17', 'S25']:
        audit['screen_provenance'][s]['nt_list_sha256'] = shared.file_hash(
            ROOT / f'data/processed/GSE262028_screen_inputs_v1/IL13/{s}_nt_author50.txt')
    config_path = out / 'audit_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != audit:
        raise ValueError('Audit configuration changed; do not mix result versions')
    shared.write_json(config_path, audit)
    effects.to_csv(out / 'screen_descriptive_effects.csv', index=False)
    print(json.dumps(audit, indent=2), flush=True)
    if args.check_only:
        return
    if not args.summarize_only:
        for name in MODELS if args.model == 'both' else [args.model]:
            run_model(inputs, name, out / name, args.batch_size)
    if args.summarize_only or args.model == 'both':
        summarize(inputs, out, effects, args.batch_size, args.smoke)
    else:
        print('One model finished. Complete the other, then run --summarize-only.', flush=True)


if __name__ == '__main__':
    main()
