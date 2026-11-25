"""Paper-style quantitative Nmur1 intervention with the original GeneCompass Base.

CLS + top-half eligibility + re-tokenization + random-reference cosine gain.
Public human-only median fallback and a labelled mouse proxy are kept separate.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/ilc2_numba'

import argparse
import ast
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'external/GeneCompass'))
import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from genecompass_compat import load_genecompass
from posttrain_data import file_hash
from qc_utils import _raw_counts

CONFIG = ROOT / 'configs/nmur1_genecompass_original_hdm_v1.json'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def official_tokenizer():
    # The upstream script has file-reading side effects; execute only its pure function.
    source = ROOT / 'external/GeneCompass/preprocess/preprocess.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tokenize_cell')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['tokenize_cell']


def prepare(cfg):
    out, data = ROOT / cfg['result_directory'], ROOT / cfg['data_directory']
    out.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(ROOT / cfg['source'])
    a = a[a.obs.condition.eq(cfg['condition'])].copy()
    counts = _raw_counts(a.layers['counts'])
    if (not a.obs_names.is_unique or not a.var_names.is_unique or (a.X != counts).nnz
            or not a.obs.qc_version.eq('ilc2_balanced_v1').all()
            or not a.obs.selected_ILC2.all() or not a.obs.tissue.eq('Lung').all()
            or not a.obs.study.eq('GSE102299').all()):
        raise ValueError('Expected the fixed balanced_v1 HDM cohort')
    tokens = pickle.loads((ROOT / cfg['token_dictionary']).read_bytes())
    keep = a.var.is_core.to_numpy() & a.var_names.isin(tokens)
    gene_ids = a.var_names[keep].to_numpy()
    columns_by_token = {tokens[g]: j for j, g in enumerate(gene_ids)}
    x = counts[:, keep].tocsr()
    target_column = int(np.flatnonzero(gene_ids == cfg['target'])[0])
    if a.var.loc[cfg['target'], 'gene_symbol'] != cfg['symbol']:
        raise ValueError('Target gene ID/symbol mismatch')
    tokenize = official_tokenizer()
    meta = a.obs.copy(); meta['group'] = meta.genotype.astype(str).map({'WT': 'WT', 'Nmur1 KO': 'KO'})
    if meta.group.isna().any():
        raise ValueError('Unknown genotype')
    for recipe, reference in cfg['references'].items():
        folder = data / recipe; folder.mkdir(exist_ok=True)
        median = pickle.loads((ROOT / reference).read_bytes())
        scale = np.array([median.get(g, 1.) for g in gene_ids])
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid median reference')
        info, ids, values, offsets, active_metadata, variants = [], [], [], [0], [], []
        rng = np.random.default_rng(cfg['random_seed'])
        for row in range(len(meta)):
            v = x[row].toarray().ravel().astype(float)/scale
            ii, vv = tokenize(np.log2(1+v), gene_ids, tokens)
            rank = ii.index(tokens[cfg['target']])+1 if tokens[cfg['target']] in ii else 0
            length = min(len(ii), cfg['max_length'])
            half_valid = rank > 0 and rank <= length//2
            half_full = rank > 0 and rank <= len(ii)//2
            original = dict(source_row=row, cell_id=str(meta.index[row]),
                target_raw_count=float(x[row, target_column]), target_rank=rank,
                full_nonzero_genes=len(ii), input_length=length,
                in_top2048=0 < rank <= cfg['max_length'], top_half_valid=half_valid, top_half_full=half_full)
            info.append(original)
            # The full-ranking half includes the primary valid-sequence-half subset.
            if meta.group.iloc[row] != 'KO' and not half_full:
                continue
            slot = len(active_metadata)
            active_metadata.append({**meta.iloc[row].to_dict(), **original})
            def append_sequence(token_ids, expression_values):
                token_ids, expression_values = token_ids[:cfg['max_length']], expression_values[:cfg['max_length']]
                ids.extend(token_ids); values.extend(expression_values); offsets.append(len(ids))
            append_sequence(ii, vv)
            variants.append(dict(active_row=slot, operator='before', fraction=1., deleted_token=-1,
                                 target_in_input=tokens[cfg['target']] in ii[:cfg['max_length']]))
            if meta.group.iloc[row] == 'WT':
                p = rank-1
                positions = np.arange(max(0, p-cfg['random_rank_window']), min(len(ii), p+cfg['random_rank_window']+1))
                positions = positions[positions != p]
                random_positions = rng.choice(positions, cfg['random_deletions'], replace=len(positions)<cfg['random_deletions'])
                interventions = [(f'Nmur1_{fraction:g}', target_column, fraction) for fraction in cfg['fractions']]
                interventions += [(f'random{k}', columns_by_token[ii[pos]], 0.)
                                  for k, pos in enumerate(random_positions)]
                for operator, column, fraction in interventions:
                    changed = v.copy(); changed[column] *= fraction
                    new_ids, new_values = tokenize(np.log2(1+changed), gene_ids, tokens)
                    append_sequence(new_ids, new_values)
                    variants.append(dict(active_row=slot, operator=operator, fraction=fraction,
                        deleted_token=int(tokens[gene_ids[column]]),
                        target_in_input=tokens[cfg['target']] in new_ids[:cfg['max_length']]))
        obs = pd.DataFrame(active_metadata).set_index('cell_id')
        obs.to_csv(folder / 'cells.tsv.gz', sep='\t')
        full = meta.join(pd.DataFrame(info).set_index('cell_id').drop(columns='source_row'))
        full.to_csv(folder / 'coverage_cells.tsv.gz', sep='\t', index_label='cell_id')
        pd.DataFrame(variants).to_csv(folder / 'variants.tsv.gz', sep='\t', index=False)
        for name, arr in [('gene_ids', np.array(ids, dtype=np.int32)), ('values', np.array(values, dtype=np.float32)),
                          ('offsets', np.array(offsets, dtype=np.int64))]:
            np.save(folder / f'{name}.npy', arr)
        coverage = full.groupby(['author_replicate', 'group'], observed=True).agg(cells=('gsm', 'size'),
            detected=('target_raw_count', lambda s: s.gt(0).sum()), in_top2048=('in_top2048', 'sum'),
            top_half_valid=('top_half_valid', 'sum'), top_half_full=('top_half_full', 'sum'))
        coverage.to_csv(out / f'{recipe}_coverage.csv')
        save_json(folder / 'preparation.json', dict(config=cfg, recipe=recipe, measured_core_vocab_genes=len(gene_ids),
            median_matched_genes=sum(g in median for g in gene_ids), reference_sha256=file_hash(ROOT / reference),
            source_sha256=file_hash(ROOT / cfg['source']), checkpoint_sha256=file_hash(ROOT / cfg['checkpoint'] / 'pytorch_model.bin'),
            preprocess='Released Normalized: counts/gene median (missing=1), log2(1+x); released tokenize_cell quicksort; Top2048',
            model_inputs='species=1 for mouse; PAD masked; special cell token retained',
            no_CP10K=True, no_target_reference_fitting=True, manipulation='Scale normalized linear expression, re-rank entire measured core/vocab genes then Top2048; refill possible',
            exact_original_mouse_training_reference_available=False,
            partial_reproduction_reason='Official public reference is human-only; proxy is project-generated, not an original mouse training reference'))
        print(recipe+'\n'+coverage.to_string(), flush=True)


def run(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    model, load_report = load_genecompass(ROOT / cfg['checkpoint'])
    if load_report['missing'] or load_report['unexpected']:
        raise ValueError(load_report)
    model = model.cuda().eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if not model.bert.use_cls_token:
        raise ValueError('Paper cell embedding requires special token')
    for recipe in cfg['references']:
        folder = ROOT / cfg['data_directory'] / recipe
        saved = json.loads((folder / 'preparation.json').read_text())
        if (saved['config'] != cfg or saved['checkpoint_sha256'] != file_hash(ROOT / cfg['checkpoint'] / 'pytorch_model.bin')
                or saved['source_sha256'] != file_hash(ROOT / cfg['source'])
                or saved['reference_sha256'] != file_hash(ROOT / cfg['references'][recipe])):
            raise ValueError('Changed checkpoint/config')
        ids, values, offsets = [np.load(folder / f'{name}.npy', mmap_mode='r') for name in ('gene_ids','values','offsets')]
        lengths = np.diff(offsets); order = np.argsort(lengths, kind='stable')
        cls, mean = np.full((len(lengths), model.config.hidden_size), np.nan, dtype=np.float32), np.full((len(lengths), model.config.hidden_size), np.nan, dtype=np.float32)
        def forward(rows):
            n = int(lengths[rows].max())
            ii, vv = np.zeros((len(rows), n), dtype=np.int64), np.zeros((len(rows), n), dtype=np.float32)
            for j, row in enumerate(rows):
                ii[j, :lengths[row]] = ids[offsets[row]:offsets[row+1]]
                vv[j, :lengths[row]] = values[offsets[row]:offsets[row+1]]
            mask = torch.as_tensor(np.arange(n)[None, :] < lengths[rows, None], device='cuda')
            hh = model.bert(input_ids=torch.as_tensor(ii, device='cuda'), values=torch.as_tensor(vv, device='cuda'),
                attention_mask=mask, species=torch.ones((len(rows),1), device='cuda', dtype=torch.long),
                emb_warmup_alpha=1., return_dict=True).last_hidden_state
            return hh[:, 0], (hh[:, 1:]*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        start = time.monotonic()
        with torch.inference_mode():
            for left in range(0, len(order), cfg['batch_size']):
                rows = order[left:left+cfg['batch_size']]
                cc, mm = forward(rows)
                if left == 0:
                    repeated, _ = forward(rows)
                    sham = float((repeated-cc).abs().max())
                    single, _ = forward(rows[:1])
                    alignment = float((single-cc[:1]).abs().max())
                cls[rows], mean[rows] = cc.cpu().numpy(), mm.cpu().numpy()
                if left%512 == 0:
                    print(recipe, left, '/', len(order), 'seconds', round(time.monotonic()-start), flush=True)
        if not np.isfinite(cls).all() or not np.isfinite(mean).all() or sham > 1e-5 or alignment > 1e-4:
            raise ValueError('Nonfinite embedding or failed repeated/batched inference check')
        np.save(folder / 'cls_embeddings.npy', cls); np.save(folder / 'mean_embeddings.npy', mean)
        save_json(folder / 'inference.json', dict(load_report=load_report, model_updates=0, dimensions=model.config.hidden_size,
            sham_max_abs=sham, batch_alignment_max_abs=alignment, special_token_cell_embedding=True,
            precision='FP32; TF32/dropout off', seconds=time.monotonic()-start, source=saved,
            operator_code_sha256=file_hash(Path(__file__))))


def score(cfg):
    out, data = ROOT / cfg['result_directory'], ROOT / cfg['data_directory']
    rows, cells = [], []
    for recipe in cfg['references']:
        folder = data / recipe
        obs = pd.read_csv(folder / 'cells.tsv.gz', sep='\t', index_col=0, dtype={'author_replicate': str})
        variants = pd.read_csv(folder / 'variants.tsv.gz', sep='\t')
        variants['sequence_row'] = np.arange(len(variants))
        for pooling in ['cls', 'mean']:
            h = np.load(folder / f'{pooling}_embeddings.npy').astype(float)
            h /= np.linalg.norm(h, axis=1, keepdims=True)
            baseline = variants[variants.operator.eq('before')].set_index('active_row').sequence_row
            for rep in cfg['replicates']:
                real = np.flatnonzero(obs.author_replicate.eq(rep) & obs.group.eq('KO'))
                for scope, flag in [('paper_top_half_valid', 'top_half_valid'), ('sensitivity_top_half_full', 'top_half_full')]:
                    source = np.flatnonzero(obs.author_replicate.eq(rep) & obs.group.eq('WT') & obs[flag])
                    if not len(source) or len(source)>len(real):
                        raise ValueError('Invalid random pair sampling population')
                    for seed in cfg['pairing_seeds']:
                        target = np.random.default_rng(seed).choice(real, len(source), replace=False)
                        reference = h[baseline.loc[target].to_numpy()]
                        original = h[baseline.loc[source].to_numpy()]
                        initial = (original*reference).sum(1)
                        for operator in variants.operator.unique():
                            if operator == 'before': continue
                            selected = variants[variants.operator.eq(operator)].set_index('active_row').loc[source]
                            gain = (h[selected.sequence_row.to_numpy()]*reference).sum(1)-initial
                            rows.append(dict(recipe=recipe,pooling=pooling,replicate=rep,scope=scope,pairing_seed=seed,
                                operator=operator,cells=len(source),mean_cosine_gain=float(gain.mean()),
                                positive_gain_fraction=float((gain>0).mean()),mean_cosine_before=float(initial.mean())))
                            if pooling=='cls' and scope=='paper_top_half_valid':
                                cells.extend(dict(recipe=recipe,replicate=rep,pairing_seed=seed,cell_id=obs.index[s],
                                    real_KO_cell_id=obs.index[t],operator=operator,cosine_gain=float(g)) for s,t,g in zip(source,target,gain))
    draws=pd.DataFrame(rows); draws.to_csv(out/'cosine_pairing_draws.csv',index=False)
    draws.groupby(['recipe','pooling','replicate','scope','operator'],sort=False)[['mean_cosine_gain','positive_gain_fraction','cells']].mean().to_csv(out/'cosine_summary.csv')
    per_cell=pd.DataFrame(cells); per_cell.to_csv(out/'cell_cosine_gains.tsv.gz',sep='\t',index=False)
    comparison=[]
    for (recipe,rep,seed), group in per_cell.groupby(['recipe','replicate','pairing_seed']):
        matrix=group.pivot(index='cell_id',columns='operator',values='cosine_gain')
        random=matrix[[f'random{k}' for k in range(cfg['random_deletions'])]].mean(1)
        for op in ['Nmur1_0.5','Nmur1_0.25','Nmur1_0']:
            comparison.append(dict(recipe=recipe,replicate=rep,pairing_seed=seed,operator=op,cells=len(matrix),
                mean_cosine_gain=matrix[op].mean(),positive_gain_fraction=matrix[op].gt(0).mean(),
                random_mean_gain=random.mean(),random_positive_fraction=random.gt(0).mean(),
                target_beats_random_fraction=matrix[op].gt(random).mean(),paired_excess_gain=(matrix[op]-random).mean()))
    pd.DataFrame(comparison).to_csv(out/'primary_random_comparison.csv',index=False)
    print(pd.DataFrame(comparison).groupby(['recipe','replicate','operator']).mean(numeric_only=True).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','score','all'])
    p.add_argument('--config',type=Path,default=CONFIG);args=p.parse_args();cfg=json.loads(args.config.read_text())
    torch.set_num_threads(4)
    with threadpool_limits(4):
        if args.stage in ['prepare','all']:prepare(cfg)
        if args.stage in ['run','all']:run(cfg)
        if args.stage in ['score','all']:score(cfg)
