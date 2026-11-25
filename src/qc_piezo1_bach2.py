"""Shared balanced_v1 QC for the authorized Piezo1 and Bach2 lung cohorts."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/ilc2_numba'
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'
import argparse
import gc
import json
import pickle
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from threadpoolctl import threadpool_limits
from qc_utils import _raw_counts, _gene_ids, qc_sample, read_gene_annotation
from ilc2_qc import add_evidence, apply_standard
from qc_gse153748 import cluster
from genecompass_coverage import map_genes
from posttrain_data import encode, file_hash
import qc_gse136154 as plots

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {'GSE279338': 'Piezo1', 'OMIX009712': 'Bach2'}


def read_library(row):
    if row['format'] == '10x_h5':
        a = sc.read_10x_h5(ROOT / row['relative_path'], gex_only=True)
    elif row['format'] == '10x_mtx':
        a = sc.read_10x_mtx(ROOT / row['relative_path'], var_names='gene_symbols', make_unique=False, gex_only=True)
    else:
        raise ValueError('Input format must be inspected before adding a reader')
    a.X = _raw_counts(a.X)
    a.var['gene_symbol'] = a.var_names.to_numpy()
    a.var['gene_id_original'] = a.var.gene_ids.astype(str).to_numpy()
    a.var_names = _gene_ids(a.var.gene_id_original)
    a.var.index.name = 'gene_id'
    if not a.var_names.is_unique or not a.obs_names.is_unique:
        raise ValueError('Gene IDs and source barcodes must be unique')
    a.obs['barcode'] = a.obs_names
    a.obs_names = row['study'] + ':' + row['sample'] + ':' + a.obs_names
    a.obs.index.name = 'cell_id'
    for key, value in row.items():
        a.obs[key] = str(value)
    for key, value in dict(library_id=row['sample'], author_batch=row['capture_library'],
                           species='Mouse', role='downstream_only', split='unassigned').items():
        a.obs[key] = value
    return a


def main(study):
    data, out = ROOT / f'data/processed/{study}_qc', ROOT / f'results/{study}_qc'
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'configs/1QC.json').read_text())
    q, policy = cfg['basic_qc'], cfg['identity']
    reference = cfg['dataset_overrides'][study]
    annotation = read_gene_annotation(ROOT / reference['ensembl_annotation'])
    prep = json.loads((ROOT / 'configs/2prepare_traindata.json').read_text())
    tokens = pickle.loads((ROOT / prep['token_dictionary']).read_bytes())
    medians = pickle.loads((ROOT / prep['gene_median_reference']).read_bytes())
    manifest_path = ROOT / f'data/metadata/{study}_sample_metadata.tsv'
    manifest = pd.read_csv(manifest_path, sep='\t', keep_default_na=False)
    if manifest['sample'].duplicated().any() or not manifest.tissue.eq('Lung').all() or not manifest.study.eq(study).all():
        raise ValueError('Expected unique lung-only source libraries')
    parts, decisions, summaries = [], [], []
    for row in manifest.to_dict('records'):
        a = read_library(row)
        mapped = annotation.reindex(a.var_names)
        if mapped.gene_biotype.isna().any():
            raise ValueError('QC annotation must cover all original gene IDs')
        if parts and not a.var_names.equals(parts[0].var_names):
            raise ValueError('Source gene spaces differ; do not silently discard features')
        a.var['mt'] = mapped.is_mt.to_numpy(bool)
        a.var['genecompass_token'], _, a.var['has_median_reference'] = map_genes(a.var_names, tokens, medians)
        a.var['in_genecompass_vocab'] = a.var.genecompass_token.ge(0)
        sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=None, inplace=True)
        candidate = (a.obs.n_genes_by_counts >= q['min_detected_genes']) & (a.obs.total_counts >= policy['min_candidate_UMI'])
        d = a.obs.copy()
        d['candidate_pass'] = candidate
        result = qc_sample(a[candidate].copy(), annotation, min_genes=q['min_detected_genes'],
                           min_cells=q['min_cells_per_sample'], min_coding_mirna=q['min_detected_protein_coding_or_miRNA'],
                           max_mt_pct=q['max_mt_pct_inclusive'], n_genes_upper_sd=q['n_genes_upper_sd'])
        full, core = result['full'], result['core']
        sc.pp.scrublet(core, expected_doublet_rate=policy['scrublet_expected_rate'],
                       threshold=policy['scrublet_review_threshold'], n_prin_comps=30,
                       random_state=cfg['random_seed'], verbose=False)
        if not np.isfinite(core.obs.doublet_score).all():
            raise ValueError('Nonfinite doublet scores')
        full.obs = core.obs.copy()
        parts.append(full)
        d = d.join(result['decisions'].drop(columns=d.columns.intersection(result['decisions'].columns)))
        d['basic_qc_pass'] = d.index.isin(full.obs_names)
        decisions.append(d)
        summaries.append(dict(sample=row['sample'], genotype=row['genotype'], condition=row['condition'],
                              input_cells=a.n_obs, candidate_cells=int(candidate.sum()), basic_qc_cells=full.n_obs,
                              upper_gene_limit=result['summary']['n_genes_upper_limit']))
        print(summaries[-1], flush=True)
        del a, result, core
        gc.collect()
    full = ad.concat(parts, merge='same')
    del parts
    missing = add_evidence(full)
    analysis = full[:, full.var.is_core].copy()
    print('Joint clustering and identity review', full.shape, flush=True)
    cluster(analysis)
    full.obs['leiden'] = analysis.obs.leiden
    full.obs['library_cluster'] = full.obs.library_id.astype(str) + ':' + full.obs.leiden.astype(str)
    apply_standard(full, policy)
    full.obs['qc_version'] = policy['preset']
    full.obs['identity_confidence_method'] = 'computational_ilc2_balanced_v1_not_experimental_validation'
    if study == 'OMIX009712':
        full.obs['identity_review_status'] = 'provisional_CD90_lymphocyte_candidates'
    full.obs['low_RNA_review'] = ~full.obs.additional_quality_pass
    symbol = TARGETS[study]
    target = full.var_names[full.var.gene_symbol.eq(symbol)]
    if len(target) != 1 or target[0] not in tokens:
        raise ValueError('Expected one target gene represented in the model vocabulary')
    ids, values, lengths = encode(full.layers['counts'], *map_genes(full.var_names, tokens, medians)[:2],
                                  prep['max_gene_tokens'], prep['encoding'])
    full.obs['target_counts'] = full[:, target].layers['counts'].toarray().ravel()
    full.obs['target_in_top2048'] = ((ids == tokens[target[0]]) & (np.arange(ids.shape[1])[None, :] < lengths[:, None])).any(1)
    del ids, values, lengths
    full.uns.clear()
    full.uns['qc_version'] = policy['preset']
    full.uns['provenance_json'] = json.dumps(dict(basic_qc=q, identity=policy, reference=reference,
        manifest_sha256=file_hash(manifest_path), missing_markers=missing, target_gene=symbol,
        target_gene_id=target[0], target_token=int(tokens[target[0]]), encoding=prep['encoding'],
        max_gene_tokens=prep['max_gene_tokens'], note='3SD and Scrublet per capture; joint clustering only for review; no batch correction'))
    selected = full[full.obs.selected_ILC2].copy()
    if (selected.X != selected.layers['counts']).nnz or not full.obs_names.is_unique:
        raise ValueError('Counts or source identity integrity failed')
    d = pd.concat(decisions)
    d = d.join(full.obs.drop(columns=d.columns.intersection(full.obs.columns)))
    d['selected_ILC2'] = d.index.isin(selected.obs_names)
    d['final_reason'] = np.select([~d.candidate_pass, ~d.basic_qc_pass,
        d.index.isin(full.obs_names[~full.obs.identity_positive]),
        d.index.isin(full.obs_names[full.obs.alternative_program])],
        ['below_candidate_RNA', 'basic_qc_fail', 'weak_identity', 'alternative_program'], default='retained')
    d.to_csv(out / 'cell_decisions.tsv.gz', sep='\t')
    d.groupby(['sample', 'final_reason']).size().rename('cells').to_csv(out / 'filter_reasons.tsv', sep='\t')
    coverage = []
    for summary in summaries:
        s = selected[selected.obs['sample'].eq(summary['sample'])]
        b = full[full.obs['sample'].eq(summary['sample'])]
        summary.update(retained_ILC2=s.n_obs, retention_pct=100*s.n_obs/summary['input_cells'],
                       doublet_review=int(s.obs.doublet_review.sum()), low_RNA_review=int(s.obs.low_RNA_review.sum()),
                       median_UMI=float(s.obs.total_counts.median()), median_genes=float(s.obs.n_genes_by_counts.median()),
                       median_mt_pct=float(s.obs.pct_counts_mt.median()))
        for scope, obj in [('basic_qc', b), ('ILC2', s)]:
            coverage.append(dict(sample=summary['sample'], genotype=summary['genotype'], scope=scope, gene=symbol,
                cells=obj.n_obs, detected=int(obj.obs.target_counts.gt(0).sum()), top2048=int(obj.obs.target_in_top2048.sum()),
                detected_pct=100*obj.obs.target_counts.gt(0).mean(), top2048_pct=100*obj.obs.target_in_top2048.mean()))
    summary = pd.DataFrame(summaries)
    coverage = pd.DataFrame(coverage)
    summary.to_csv(out / 'sample_summary.tsv', sep='\t', index=False)
    coverage.to_csv(out / 'target_coverage.tsv', sep='\t', index=False)
    full.obs.groupby(['sample', 'leiden'], observed=True).agg(cells=('selected_ILC2','size'),
        retained=('selected_ILC2','sum'), alternative=('alternative_program','sum'),
        T_program=('lineage_T','sum'), NK_ILC1_program=('lineage_NK_ILC1','sum')).to_csv(out / 'cluster_review.tsv', sep='\t')
    full.write_h5ad(data / f'{study}_basic_qc.h5ad', compression='gzip')
    selected.write_h5ad(data / f'{study}_ILC2.h5ad', compression='gzip')
    selected.obs.to_csv(data / f'{study}_metadata.tsv.gz', sep='\t')
    analysis.obs = full.obs.copy()
    sc.tl.umap(analysis, random_state=cfg['random_seed'])
    pd.DataFrame(analysis.obsm['X_umap'], index=analysis.obs_names, columns=['UMAP1','UMAP2']).join(
        full.obs[['sample','genotype','selected_ILC2','leiden']]).to_csv(out / 'umap_coordinates.tsv.gz', sep='\t')
    plots.OUT = out
    # Short display labels prevent legends from covering the adjacent panel; saved metadata stays literal.
    analysis.obs['genotype'] = np.where(analysis.obs.condition.eq('control'), 'Control', symbol+' cKO')
    if study == 'OMIX009712':
        analysis.obs['sample'] = analysis.obs['sample'].astype(str).str.replace('Lung_Il7rCre+', 'Ctrl ', regex=False).str.replace('Lung_Bach2_Cko_', 'cKO ', regex=False)
    plots.plot_qc(analysis, d, selected, umap_colors=['genotype','sample','leiden','qc_identity'])
    (out / 'QC_report.md').write_text(f'# {study} / {symbol} 肺数据QC\n\n'
        '统一ilc2_balanced_v1，未调整数值阈值。候选RNA、基础QC、身份与状态分开；每捕获文库单独计算3SD和Scrublet，双细胞分数仅标记。'
        '联合聚类只用于身份复核，不做批次校正。高置信度仅为计算身份支持。\n\n'
        '```tsv\n'+summary.to_csv(sep='\t',index=False)+'```\n\n'
        '## 目标基因覆盖\n\n在原始整数counts上按当前median/top2048编码，未做模型KO推理。\n\n'
        '```tsv\n'+coverage.to_csv(sep='\t',index=False)+'```\n\n'
        f'全部原始{full.n_vars}个GEX基因保留于输出X和counts层；完整基础QC群体另存，便于检查KO导致身份变化的筛选影响。'
        '已知来源标签保留，未知动物/重复不推断；role=downstream_only，split=unassigned。'
        '配置中的参考注释覆盖全部输入ID，但不据此声称确认作者使用的注释release。\n')
    print(summary.to_string(index=False), '\n', coverage.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=list(TARGETS), required=True)
    with threadpool_limits(limits=4):
        main(parser.parse_args().study)
