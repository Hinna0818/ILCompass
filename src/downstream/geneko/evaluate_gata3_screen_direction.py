"""Frozen GeneCompass/LoRA Gata3 value reconstruction under matched TF removals.

Reuse the Il13 reference tokenizer, inference helper and direction metrics. S31/S35
candidates and random controls are locked independently, before model inference.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/downstream/geneko'))
from evaluate_il13_screen_ko import Inputs, shared
from pilot_il13_masked_expression import predict, logcp10k
import anndata as ad
import numpy as np
import pandas as pd
import torch

# Do not edit the historical Il13 runner: its completed caches hash that exact source.
IL13_RUNNER = ROOT / 'compare_pretrain/model_compare/crispr_il13/direction_eval/run.py'
spec = importlib.util.spec_from_file_location('il13_direction_metrics', IL13_RUNNER)
il13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(il13)
direction_metrics = il13.direction_metrics
CONFIG = ROOT / 'configs/gse262028_gata3_direction_v1.json'
MODELS = ['raw', 'lora']
SCORES = ['delta_target_native', 'excess_delta_native', 'delta_random_mean_native']


def screen_effects(cfg):
    """Same descriptive NT-centred guide median as Il13, using S31/S35 only."""
    folder = ROOT / cfg['screen_directory'] / cfg['screen_endpoint']
    tables = []
    for screen in cfg['screens']:
        d = pd.read_csv(folder / f'{screen}_counts_author50.tsv', sep='\t')
        nt = set((folder / f'{screen}_nt_author50.txt').read_text().splitlines())
        if nt != set(d.loc[d.Gene.eq('__NT__'), 'sgRNA']) or not nt:
            raise ValueError('Screen NT guide list mismatch')
        if (d[[f'{screen}_P', f'{screen}_N']] < 50).any().any():
            raise ValueError('Expected previously prepared author50 inputs')
        ratio = np.log2(d[f'{screen}_P'] / d[f'{screen}_N'])
        d['effect'] = ratio - np.median(ratio[d.sgRNA.isin(nt)])
        t = d[~d.sgRNA.isin(nt)].groupby('Gene').effect.agg(['median', 'size'])
        tables.append(t.rename(columns={'median': f'{screen}_effect', 'size': f'{screen}_n_guides_author50'}))
    effects = pd.concat(tables, axis=1).rename_axis('author_symbol').reset_index()
    mapping = pd.read_csv(ROOT / cfg['screen_directory'] / 'gene_map.tsv', sep='\t')
    effects = mapping.merge(effects, on='author_symbol', how='left', validate='one_to_one')
    cols = [s+'_effect' for s in cfg['screens']]
    effects['mean_signed_effect'] = effects[cols].mean(axis=1, skipna=False)
    effects['screen_sign_agreement'] = (np.sign(effects[cols[0]]) == np.sign(effects[cols[1]])) & effects[cols].ne(0).all(axis=1)
    effects['screen_weight'] = effects[cols].abs().min(axis=1, skipna=False)
    return effects


def make_pairs(targets, ref, readout, rules):
    """Same rank-window sampling rule as prepare_il13_screen_design, seed locked once."""
    rng = np.random.default_rng(rules['random_seed'])
    records = []
    readout_present = (ref.ids == readout).any(axis=1)
    for t in targets.sort_values('ensembl_gene_id').itertuples():
        token = ref.tokens[t.ensembl_gene_id]
        selected = np.flatnonzero((ref.ids == token).any(axis=1) & readout_present)
        for r in selected:
            n = int(ref.lengths[r])
            ids = ref.ids[r, :n]
            p = int(np.flatnonzero(ids == token)[0])
            q = int(np.flatnonzero(ids == readout)[0])
            w = rules['random_rank_window']
            choices = np.arange(max(0, p-w), min(n, p+w+1))
            choices = choices[(ids[choices] != token) & (ids[choices] != readout)]
            controls = rng.choice(choices, rules['random_deletions_per_target_cell'], replace=False)
            row = dict(pair_id=len(records), target_gene_id=t.ensembl_gene_id,
                target_symbol=t.author_symbol, target_token_id=int(token), cell_row=int(r),
                cell_id=ref.cells.cell_id.iloc[r], input_length=n, target_rank=p+1,
                Gata3_rank=q+1, target_in_token_top_half=p+1 <= n//2)
            for j, position in enumerate(controls):
                row.update({f'random{j}_token_id': int(ids[position]), f'random{j}_rank': int(position+1)})
            records.append(row)
    return pd.DataFrame(records)


def prepare(cfg, ref, readout):
    out = ROOT / cfg['design_directory']
    rules = ref.design['config']
    parent = json.loads((ROOT / cfg['reference_design']).read_text())
    if parent != ref.design:
        raise ValueError('Reference adapter and declared reference design differ')
    sources = [CONFIG, ROOT / cfg['reference_design'], Path(__file__), IL13_RUNNER,
        ROOT / cfg['functional_panel'], ROOT / cfg['screen_directory'] / 'gene_map.tsv']
    screen_dir = ROOT / cfg['screen_directory'] / cfg['screen_endpoint']
    sources += [screen_dir / 'samples.tsv']
    for s in cfg['screens']:
        sources += [screen_dir / f'{s}_counts_author50.tsv', screen_dir / f'{s}_nt_author50.txt']
    hashes = {str(p.relative_to(ROOT)): shared.file_hash(p) for p in sources}
    if (out / 'design.json').exists():
        design = json.loads((out / 'design.json').read_text())
        if design['source_sha256'] != hashes or design['reference_source_sha256'] != ref.design['source_sha256']:
            raise ValueError('Frozen design sources changed; use a new result version')
        for name, digest in design['artifacts_sha256'].items():
            if shared.file_hash(out / name) != digest:
                raise ValueError(f'Frozen design changed: {name}')
        return design
    effects = screen_effects(cfg)
    readout_present = (ref.ids == readout).any(axis=1)
    rows = []
    for t in effects.itertuples():
        token = ref.tokens.get(t.ensembl_gene_id)
        in_space = t.ensembl_gene_id in ref.genes
        n = int(((ref.ids == token).any(axis=1) & readout_present).sum()) if in_space else 0
        reasons = []
        if t.mapping_status != 'unique_exact_symbol':
            reasons.append('nonunique_or_missing_gene_mapping')
        if not in_space:
            reasons.append('outside_reference_core_vocabulary')
        if t.author_symbol == cfg['readout_symbol']:
            reasons.append('readout_not_a_perturbation_target')
        for screen in cfg['screens']:
            count = getattr(t, screen+'_n_guides_author50')
            if not np.isfinite(count) or count < rules['minimum_guides_each_screen']:
                reasons.append(screen+'_insufficient_guides')
        if n < rules['minimum_joint_cells']:
            reasons.append('insufficient_joint_token_cells')
        rows.append(dict(author_symbol=t.author_symbol, joint_Gata3_token_cells=n,
                         selected=not reasons, exclusion_reason=';'.join(reasons)))
    coverage = effects.merge(pd.DataFrame(rows), on='author_symbol', validate='one_to_one')
    targets = coverage[coverage.selected].sort_values('ensembl_gene_id').reset_index(drop=True)
    if targets.empty or not targets.ensembl_gene_id.is_unique:
        raise ValueError('Empty or ambiguous candidate set')
    pairs = make_pairs(targets, ref, readout, rules)
    out.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out / 'candidate_coverage.csv', index=False)
    targets.to_csv(out / 'targets.csv', index=False)
    pairs.to_csv(out / 'target_cell_controls.tsv.gz', sep='\t', index=False)
    panel = pd.read_csv(ROOT / cfg['functional_panel'])[['gene_symbol']]
    panel = panel.merge(coverage, left_on='gene_symbol', right_on='author_symbol', how='left', validate='one_to_one')
    panel.to_csv(out / 'functional_panel_coverage.csv', index=False)
    design = dict(config=cfg, inherited_rules={k: rules[k] for k in ['max_length', 'random_seed',
        'random_rank_window', 'random_deletions_per_target_cell', 'minimum_guides_each_screen', 'minimum_joint_cells']},
        reference_cells=len(ref.ids), Gata3_in_top2048_cells=int(readout_present.sum()),
        target_genes=len(targets), used_cells=int(pairs.cell_row.nunique()), target_cell_pairs=len(pairs),
        screen_concordant_genes=int(targets.screen_sign_agreement.sum()),
        functional_panel_selected=int(panel.selected.eq(True).sum()),
        functional_panel_concordant=int((panel.selected.eq(True) & panel.screen_sign_agreement.eq(True)).sum()),
        readout_token=int(readout), source_sha256=hashes, reference_source_sha256=ref.design['source_sha256'],
        artifacts_sha256={n: shared.file_hash(out/n) for n in ['candidate_coverage.csv', 'targets.csv',
            'target_cell_controls.tsv.gz', 'functional_panel_coverage.csv']},
        limitation='Same Il13-positive mature lung reference as Il13 benchmark, not unbiased progenitor differentiation reference; S31/S35 are lineage-associated reporter screens, not RNA logFC.')
    shared.write_json(out / 'design.json', design)
    return design


def validate_pairs(ref, pairs, targets, readout, rules):
    if pairs.empty or pairs.duplicated(['target_gene_id', 'cell_row']).any():
        raise ValueError('Empty or duplicated target-cell pairs')
    r = pairs.cell_row.to_numpy(int)
    if not np.array_equal(ref.cells.cell_id.iloc[r].to_numpy(), pairs.cell_id):
        raise ValueError('Cell identity mismatch')
    for token_col, rank_col in [('target_token_id', 'target_rank')] + [(f'random{j}_token_id', f'random{j}_rank') for j in range(3)]:
        pos = pairs[rank_col].to_numpy(int)-1
        if (pos < 0).any() or (pos >= ref.lengths[r]).any() or not np.array_equal(ref.ids[r, pos], pairs[token_col]):
            raise ValueError('Token/rank mismatch')
    if not (ref.ids[r, pairs.Gata3_rank.to_numpy(int)-1] == readout).all():
        raise ValueError('Readout absent')
    dropped = pairs[['target_token_id', *ref.random_columns]].to_numpy()
    if (dropped == readout).any() or (np.diff(np.sort(dropped, axis=1), axis=1) == 0).any():
        raise ValueError('Readout deletion or repeated control')
    for j in range(3):
        if ((pairs[f'random{j}_rank'] - pairs.target_rank).abs() > rules['random_rank_window']).any():
            raise ValueError('Random rank mismatch')
    expected = targets.set_index('ensembl_gene_id').joint_Gata3_token_cells.sort_index()
    actual = pairs.groupby('target_gene_id').size().sort_index()
    if not expected.index.equals(actual.index) or not np.array_equal(expected, actual):
        raise ValueError('Eligible target cells lost')


def token_set(pairs, ref):
    return np.unique(pairs[['target_token_id', *ref.random_columns]].to_numpy(np.int64))


def run_model(ref, name, out, signature, batch_size):
    folder = out / name
    cache = folder / 'cells'
    cache.mkdir(parents=True, exist_ok=True)
    sign = dict(signature, model=name, checkpoint=shared.provenance(name))
    path = folder / 'run_signature.json'
    if path.exists() and json.loads(path.read_text()) != sign:
        raise ValueError('Inference configuration/checkpoint changed; do not mix cached results')
    shared.write_json(path, sign)
    pending = [r for r in ref.groups if not (cache / f'{r:06d}.npz').exists()]
    if not pending:
        print(f'{name}: already complete', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; --check-only and --prepare-only work on CPU')
    model = shared.load(name).float().eval().requires_grad_(False)
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    start = time.monotonic()
    for k, r in enumerate(pending):
        n = int(ref.lengths[r])
        seq = (ref.ids[r, :n].astype(np.int64), ref.values[r, :n].copy())
        regenerated = ref.sequence(ref.counts[r].toarray().ravel()/ref.scale)
        if not all(np.array_equal(x, y) for x, y in zip(seq, regenerated)):
            raise ValueError('Frozen tokenizer baseline changed')
        before = float(predict(model, [seq], [None], ref.readout, name)[0])
        sham = predict(model, [seq]*batch_size, [None]*batch_size, ref.readout, name)
        error = float(np.max(np.abs(sham-before)))
        if error > 1e-4:
            raise ValueError('Large sham numerical error')
        if k == 0:
            altered = (seq[0].copy(), seq[1].copy())
            altered[1][seq[0] == ref.readout] = 999.
            q = predict(model, [seq, altered], [None, None], ref.readout, name)
            if abs(q[0]-q[1]) > 1e-6:
                raise ValueError('Masked Gata3 value leaked into prediction')
            shared.write_json(folder / 'engineering_check.json', dict(readout='Gata3', cell_row=int(r),
                hidden_value_invariance_max_abs=float(abs(q[0]-q[1])), cls_offset=1,
                protocol='readout value=-1, retain ID; target PAD/value0/attention0; no reorder/refill'))
        tokens = token_set(ref.groups[r], ref)
        after = []
        for left in range(0, len(tokens), batch_size):
            tt = tokens[left:left+batch_size]
            after.extend(predict(model, [seq]*len(tt), list(tt), ref.readout, name))
        temp = cache / f'{r:06d}.tmp.npz'
        np.savez_compressed(temp, cell_row=r, tokens=tokens, baseline=before,
                            after=np.asarray(after), sham_error=error)
        temp.replace(cache / f'{r:06d}.npz')
        if k == 0 or (k+1) % 20 == 0 or k+1 == len(pending):
            elapsed = time.monotonic()-start
            print(f'{name}: cells={len(ref.groups)-len(pending)+k+1}/{len(ref.groups)} '
                  f'elapsed={elapsed/60:.1f}min ETA={elapsed/(k+1)*(len(pending)-k-1)/60:.1f}min', flush=True)
    del model
    torch.cuda.empty_cache()


def summarize(ref, cfg, out, signature, smoke):
    source = ROOT / ref.design['config']['reference_directory'] / 'GSE126924_Lung_IL13positive_ILC2_1864.h5ad'
    a = ad.read_h5ad(source)
    totals = np.asarray(a.X.sum(axis=1)).ravel()
    del a
    scale = ref.scale[ref.column_by_token[ref.readout]]
    tables, errors = {}, []
    for name in MODELS:
        saved = json.loads((out/name/'run_signature.json').read_text())
        if saved != dict(signature, model=name, checkpoint=shared.provenance(name)):
            raise ValueError('Cached inference signature differs')
        pieces = []
        for r, pairs in ref.groups.items():
            with np.load(out/name/'cells'/f'{r:06d}.npz') as z:
                tokens, after = z['tokens'], z['after']
                if int(z['cell_row']) != r or not np.array_equal(tokens, token_set(pairs, ref)):
                    raise ValueError('Cached perturbation identity mismatch')
                before = float(z['baseline'])
                errors.append(float(z['sham_error']))
            d = pairs.copy()
            for unit, vals, base in [('native', after, before),
                    ('logCP10K', logcp10k(after, scale*10000/totals[r]),
                     float(logcp10k(np.array([before]), scale*10000/totals[r])[0]))]:
                lookup = dict(zip(tokens, vals))
                d[f'pred_before_{unit}'] = base
                d[f'delta_target_{unit}'] = d.target_token_id.map(lookup)-base
                for j in range(3):
                    d[f'delta_random{j}_{unit}'] = d[f'random{j}_token_id'].map(lookup)-base
                d[f'delta_random_mean_{unit}'] = d[[f'delta_random{j}_{unit}' for j in range(3)]].mean(axis=1, skipna=False)
                d[f'excess_delta_{unit}'] = d[f'delta_target_{unit}']-d[f'delta_random_mean_{unit}']
            d['model'] = name
            pieces.append(d)
        tables[name] = pd.concat(pieces).sort_values('pair_id').reset_index(drop=True)
        if not np.isfinite(tables[name].filter(regex='^(delta|excess|pred)_').to_numpy()).all():
            raise ValueError('Nonfinite predictions')
    pd.testing.assert_series_equal(tables['raw'].pair_id, tables['lora'].pair_id)
    epsilon = max(1e-6, 10*max(errors))
    gene_tables = []
    for name, d in tables.items():
        d.to_csv(out/f'{name}_cell_predictions.csv.gz', index=False)
        cols = [f'{score}_{unit}' for unit in ['native', 'logCP10K']
                for score in ['delta_target', 'delta_random_mean', 'excess_delta']]
        for score in SCORES:
            d[score+'_fraction_up'] = (d[score] > epsilon).astype(float)
            d[score+'_fraction_down'] = (d[score] < -epsilon).astype(float)
            cols += [score+'_fraction_up', score+'_fraction_down']
        grouped = d.groupby(['target_gene_id', 'target_symbol'])
        g = grouped[cols].mean()
        g['n_cells'] = grouped.size()
        g = g.reset_index().merge(ref.targets, left_on='target_gene_id', right_on='ensembl_gene_id', validate='one_to_one')
        if len(g) != len(ref.targets):
            raise ValueError('Targets lost in merge')
        g['model'] = name
        gene_tables.append(g)
    genes = pd.concat(gene_tables, ignore_index=True)
    genes.to_csv(out/'gene_predictions.csv', index=False)
    panel = set(pd.read_csv(ROOT/cfg['functional_panel']).gene_symbol)
    records = []
    for name, g in genes.groupby('model', sort=False):
        for group, base in [('all_targets', g), ('prior_ILC2_panel', g[g.target_symbol.isin(panel)])]:
            scopes = [(f'all_{s}', base, s+'_effect') for s in cfg['screens']]
            scopes += [('concordant_primary', base[base.screen_sign_agreement], 'mean_signed_effect'),
                       ('all_mean_secondary', base, 'mean_signed_effect')]
            for scope, sub, truth in scopes:
                for score in SCORES:
                    stats = direction_metrics(sub[score], sub[truth], epsilon)
                    if scope == 'concordant_primary':
                        weights = sub.screen_weight.to_numpy()
                        pred = np.where(sub[score].abs() > epsilon, np.sign(sub[score]), 0)
                        correct = pred == np.sign(sub[truth].to_numpy())
                        stats.update(weighted_accuracy=float(np.average(correct, weights=weights)) if weights.sum() else np.nan,
                            total_weight=float(weights.sum()), effective_n_weights=float(weights.sum()**2/(weights**2).sum()) if weights.sum() else np.nan)
                    records.append(dict(model=name, group=group, scope=scope, score=score, epsilon_native=epsilon, **stats))
    metrics = pd.DataFrame(records)
    metrics.to_csv(out/'direction_metrics.csv', index=False)
    primary = metrics[(metrics.scope == 'concordant_primary') & metrics.score.ne('delta_random_mean_native')]
    lines = [f'|{r.group}|{r.model}|{r.score}|{r.n_correct}/{r.n_genes}|{r.accuracy:.1%}|{r.weighted_accuracy:.1%}|{r.spearman_signed_effect:.3f}|{r.pearson_signed_effect:.3f}|' for r in primary.itertuples()]
    report = f'''# Gata3 masked-value direction evaluation{' — smoke工程测试，不作总体结论' if smoke else ''}

{len(ref.targets)}个TF，{len(ref.groups)}个不同参考细胞，{len(ref.pairs):,}个目标–细胞组合。raw/LoRA冻结、不训练；同一目标共享细胞和3个排名匹配随机对照。继承Il13值遮蔽流程：Gata3 ID保留、value=-1；目标或随机槽位PAD/value0/attention0，不排序、不补位。调用原始cls4value头，CLS偏移+1。

直接Δ=目标移除后的Gata3预测−基线预测；额外Δ=直接Δ−三个随机Δ均值。先按目标在细胞内平均，再按基因计算命中率，不按细胞数加权。共用数值容差{epsilon:.6g}；不确定预测仍在分母，实验效应恰为0不计方向。各screen独立及冲突基因辅助结果均在CSV。等权主指标与screen效应加权辅助指标并列：w=min(|S31|,|S35|)，非生物学重要性/显著性权重。

|集合|模型|预测量|正确/总数|等权准确率|效应加权准确率|Spearman|Pearson|
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

S31/S35参照沿用描述性log2(P/N)减NT中位数、目标内guide中位数。不是caRpools统计复现，不计算命中显著性。P=GATA3/ILC2相关群体，N=TBET/NK相关群体；正效应表示guide偏向P群，而非直接证明Gata3 RNA升高。作者分组沿用已整理samples.tsv，GEO标题/表型等元信息冲突未被消除。

参考仍是Il13阳性成熟肺ILC2，并非祖细胞分化体系。实验效应可能来自分化、身份、存活或群体比例改变，模型读出只测Gata3 RNA条件重构敏感性。不能称ILC2特异调控或真实KO表达预测已验证。原Gata3存在性和排名仍可见；value-only遮蔽不同于预训练联合遮蔽；随机基因可有功能，不能等同实验NT。相关性衡量有符号连续效应，不是把两个二元方向直接相关。对两个模型比较时关注总猜上/下基线、随机对照及不确定比例；没有独立动物/训练seed误差条。

已有67基因panel按事先名单取交集，不根据模型对错筛选；覆盖和排除明细见设计目录。完整输出包含gene_predictions.csv、两模型cell_predictions.csv.gz、direction_metrics.csv、固定screen效应和运行配置。
'''
    (out/'report.md').write_text(report)
    summary = dict(smoke_only=smoke, targets=len(ref.targets), cells=len(ref.groups),
        target_cell_pairs=len(ref.pairs), numerical_threshold_shared=epsilon, max_sham_native_error=max(errors),
        screen_concordant_genes=int(ref.targets.screen_sign_agreement.sum()), models=MODELS)
    shared.write_json(out/'complete.json', summary)
    print(primary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['both', *MODELS], default='both')
    p.add_argument('--batch-size', type=int, default=2)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--summarize-only', action='store_true')
    p.add_argument('--smoke', action='store_true', help='First 3 stable gene IDs, 2 eligible cells each; separate output')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    cfg = json.loads(CONFIG.read_text())
    ref = Inputs(smoke=False)
    source = ROOT/ref.design['config']['reference_directory']/'GSE126924_Lung_IL13positive_ILC2_1864.h5ad'
    a = ad.read_h5ad(source, backed='r')
    ids = a.var_names[a.var.gene_symbol.eq(cfg['readout_symbol'])]
    if len(ids) != 1 or ids[0] not in ref.genes:
        raise ValueError('Gata3 not unique in reference core vocabulary')
    readout = ref.tokens[ids[0]]
    a.file.close()
    design = prepare(cfg, ref, readout)
    folder = ROOT/cfg['design_directory']
    ref.targets = pd.read_csv(folder/'targets.csv')
    ref.pairs = pd.read_csv(folder/'target_cell_controls.tsv.gz', sep='\t')
    ref.readout = readout
    validate_pairs(ref, ref.pairs, ref.targets, readout, design['inherited_rules'])
    if args.smoke:
        keep = ref.targets.ensembl_gene_id.head(3)
        ref.pairs = ref.pairs[ref.pairs.target_gene_id.isin(keep)].groupby('target_gene_id', sort=True).head(2).copy()
        ref.targets = ref.targets[ref.targets.ensembl_gene_id.isin(keep)].copy()
    ref.groups = dict(tuple(ref.pairs.groupby('cell_row', sort=True)))
    out = ROOT/cfg['output_directory']
    if args.smoke:
        out = out.with_name(out.name+'_smoke')
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), IL13_RUNNER, ROOT/'src/downstream/geneko/evaluate_il13_screen_ko.py',
        ROOT/'src/downstream/geneko/pilot_il13_masked_expression.py', ROOT/'compare_pretrain/model_compare/shared.py',
        ROOT/'src/genecompass_compat.py', ROOT/'compare_pretrain/genepass_lora/model.py',
        ROOT/'compare_pretrain/genepass_lora/config.json', ROOT/shared.MODELS['raw']/'config.json']
    signature = dict(design_sha256=shared.file_hash(folder/'design.json'),
        batch_size=args.batch_size, smoke_only=args.smoke, pair_ids=ref.pairs.pair_id.tolist(),
        precision='FP32, TF32 off', base_checkpoint=shared.provenance('raw'),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        code_sha256={str(f.relative_to(ROOT)): shared.file_hash(f) for f in files})
    audit = dict(readout=cfg['readout_symbol'], screens=cfg['screens'], targets=len(ref.targets),
        cells=len(ref.groups), target_cell_pairs=len(ref.pairs),
        unique_deletions_per_model=sum(len(token_set(pairs, ref)) for pairs in ref.groups.values()),
        batch_size=args.batch_size, smoke_only=args.smoke, design_directory=str(folder),
        output_directory=str(out), protocol='fixed-slot Gata3 value mask; target/random removal; no reordering/refill')
    print(json.dumps(audit, indent=2), flush=True)
    if args.prepare_only or args.check_only:
        shared.write_json(out/'input_validation.json', audit)
        return
    previous = out/'audit_config.json'
    if previous.exists() and json.loads(previous.read_text()) != signature:
        raise ValueError('Run inputs/settings changed; do not reuse output directory')
    shared.write_json(previous, signature)
    ref.targets.to_csv(out/'screen_descriptive_effects.csv', index=False)
    if not args.summarize_only:
        for name in MODELS if args.model == 'both' else [args.model]:
            run_model(ref, name, out, signature, args.batch_size)
    if args.model == 'both' or args.summarize_only:
        summarize(ref, cfg, out, signature, args.smoke)
    else:
        print('One model finished; finish the other and use --summarize-only.', flush=True)


if __name__ == '__main__':
    main()
