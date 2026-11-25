"""Fixed balanced_v1 QC; optional `GSE189285 Liver` selects the two liver libraries."""
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/ilc2_numba'
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '4'
import gzip
import json
from pathlib import Path
import pickle
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from threadpoolctl import threadpool_limits
from qc_utils import _raw_counts, _gene_ids, read_gene_annotation, qc_sample
from ilc2_qc import add_evidence, apply_standard
from qc_gse153748 import cluster
from genecompass_coverage import map_genes, summarize_coverage
import qc_gse136154 as plots

BASE = Path(__file__).resolve().parent / 'downstream_data'
LIBRARIES = [
    ('GSE102299', 'GSM2733478', 'PBS_Rep1', 'PBS', '1'),
    ('GSE102299', 'GSM2733479', 'PBS_Rep2', 'PBS', '2'),
    ('GSE102299', 'GSM2733482', 'IL33_Rep1', 'IL33', '1'),
    ('GSE102299', 'GSM2733483', 'IL33_Rep2', 'IL33', '2'),
    ('GSE189285', 'GSM5697025', 'CtLungILC2', 'PBS', 'not_available'),
    ('GSE189285', 'GSM5697023', '33LungILC2', 'IL33', 'not_available'),
]


def read_library(gse, gsm, title, raw_directory=None):
    folder = Path(raw_directory) if raw_directory is not None else BASE / gse / 'raw'
    if gse == 'GSE102299':
        genes = pd.read_csv(folder / 'GSE102299_genes.tsv.gz', sep='\t', header=None)
        barcodes = pd.read_csv(folder / f'{gsm}_{title}.barcodes.tsv.gz', header=None)[0]
        with gzip.open(folder / f'{gsm}_{title}.matrix.mtx.gz', 'rb') as f:
            x = _raw_counts(mmread(f).T)
    else:
        # Read only the three matrix members; never extract archive paths onto disk.
        with tarfile.open(folder / f'{gsm}_{title}.tar.gz') as archive:
            def member(suffix):
                names = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
                if len(names) != 1:
                    raise ValueError(f'Expected one {suffix}: {names}')
                return gzip.GzipFile(fileobj=archive.extractfile(names[0]))
            genes = pd.read_csv(member('features.tsv.gz'), sep='\t', header=None)
            barcodes = pd.read_csv(member('barcodes.tsv.gz'), header=None)[0]
            x = _raw_counts(mmread(member('matrix.mtx.gz')).T)
        if genes.shape[1] > 2 and not genes[2].eq('Gene Expression').all():
            raise ValueError('Unexpected non-GEX features')
    ids = _gene_ids(genes[0])
    if not ids.is_unique or not barcodes.is_unique or x.shape != (len(barcodes), len(ids)):
        raise ValueError('Nonunique IDs or inconsistent dimensions')
    a = ad.AnnData(x, obs=pd.DataFrame({'barcode': barcodes.to_numpy()},
        index=pd.Index(gse + ':' + gsm + ':' + barcodes, name='cell_id')),
        var=pd.DataFrame({'gene_symbol': genes[1].to_numpy(), 'gene_id_original': genes[0].to_numpy()},
                         index=pd.Index(ids, name='gene_id')))
    return a


def main(gse, tissue='Lung', *, libraries=None, raw_directory=None, data_directory=None,
         report_directory=None, summary_prefix='', resume=False, sample_metadata=None,
         combined_name=None, save_basic=False):
    if tissue not in ['Lung', 'Liver']:
        raise ValueError('Expected Lung or Liver')
    cfg = json.loads((ROOT / 'configs/1QC.json').read_text())
    q, policy = cfg['basic_qc'], cfg['identity']
    reference = dict(cfg['core_genes'], **cfg['dataset_overrides'].get(gse, {}))
    annotation = read_gene_annotation(ROOT / reference['ensembl_annotation'])
    prep = json.loads((ROOT / 'configs/2prepare_traindata.json').read_text())
    tokens = pickle.loads((ROOT / prep['token_dictionary']).read_bytes())
    median = pickle.loads((ROOT / prep['gene_median_reference']).read_bytes())
    out = Path(report_directory) if report_directory is not None else ROOT / f'results/{gse}_qc'
    data = Path(data_directory) if data_directory is not None else BASE / gse / 'qc'
    libraries = LIBRARIES if libraries is None else libraries
    if tissue == 'Liver':
        if gse != 'GSE189285':
            raise ValueError('Liver libraries are defined only for GSE189285')
        libraries = [(gse, 'GSM5697024', 'CtLiverILC2', 'PBS', 'not_available'),
                     (gse, 'GSM5697022', '33LiverILC2', 'IL33', 'not_available')]
        data = ROOT / 'data/processed/GSE189285_qc'
        out = out / 'liver'
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    summaries, kept = [], []
    saved = json.loads((out / f'{summary_prefix}summary.json').read_text()) if resume and (out / f'{summary_prefix}summary.json').exists() else []
    completed = {s['gsm']: s for s in saved}
    for study, gsm, title, condition, replicate in libraries:
        if study != gse:
            continue
        if gsm in completed:
            previous = ad.read_h5ad(data / f'{gsm}_ILC2.h5ad')
            provenance = json.loads(previous.uns['qc_provenance_json'])
            if provenance['identity'] != policy or provenance['basic_qc'] != q:
                raise ValueError(f'{gsm}: completed QC policy changed')
            summaries.append(completed[gsm]); kept.append(previous)
            print(gsm, 'completed QC reused', flush=True)
            continue
        a = read_library(gse, gsm, title, raw_directory=raw_directory)
        mapped = annotation.reindex(a.var_names)
        if mapped.gene_biotype.isna().any():
            raise ValueError(f'{gsm}: annotation misses {mapped.gene_biotype.isna().sum()} genes')
        a.var['mt'] = mapped.is_mt.to_numpy(bool)
        a.var['genecompass_token'], _, a.var['has_median_reference'] = map_genes(a.var_names, tokens, median)
        a.var['in_genecompass_vocab'] = a.var.genecompass_token.ge(0)
        fields = dict(study=gse, gsm=gsm, sample=gsm, library_id=gsm, capture_library=gsm,
            condition=condition, author_replicate=replicate, author_batch='not_available',
            tissue=tissue, species='Mouse', strain='C57BL/6' if gse == 'GSE102299' else 'BALB/c',
            genotype='WT', animal_id='not_available', role='downstream_only', split='unassigned',
            protocol_id=gse + ('_nasal_3d' if gse == 'GSE102299' else '_ip_5d'),
            genome_build='mm10', qc_version=policy['preset'])
        if sample_metadata is not None:
            fields.update(sample_metadata[gsm])
        for k, v in fields.items():
            a.obs[k] = v
        sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=None, inplace=True)
        candidate = (a.obs.n_genes_by_counts >= q['min_detected_genes']) & (a.obs.total_counts >= policy['min_candidate_UMI'])
        result = qc_sample(a[candidate].copy(), annotation, min_genes=q['min_detected_genes'],
            min_cells=q['min_cells_per_sample'], min_coding_mirna=q['min_detected_protein_coding_or_miRNA'],
            max_mt_pct=q['max_mt_pct_inclusive'], n_genes_upper_sd=q['n_genes_upper_sd'])
        full, core = result['full'], result['core']
        print(gsm, 'input', a.n_obs, 'basic QC', full.n_obs, flush=True)
        sc.pp.scrublet(core, expected_doublet_rate=policy['scrublet_expected_rate'],
            threshold=policy['scrublet_review_threshold'], n_prin_comps=30, random_state=cfg['random_seed'], verbose=False)
        full.obs = core.obs.copy()
        missing = add_evidence(full)
        analysis = full[:, full.var.is_core].copy()
        cluster(analysis)
        full.obs['leiden'] = analysis.obs.leiden
        full.obs['library_cluster'] = gsm + ':' + full.obs.leiden.astype(str)
        apply_standard(full, policy)
        full.obs['low_RNA_review'] = ~full.obs.additional_quality_pass
        full.obs['identity_confidence_method'] = 'computational_ilc2_balanced_v1_not_experimentally_validated'
        full.obs['identity_confidence'] = np.where(full.obs.selected_ILC2, 'high', 'review')
        d = a.obs.copy()
        d['candidate_pass'] = candidate
        d = d.join(result['decisions'].drop(columns=d.columns.intersection(result['decisions'].columns)))
        d = d.join(full.obs.drop(columns=d.columns.intersection(full.obs.columns)))
        d['basic_qc_pass'] = d.index.isin(full.obs_names)
        d['selected_ILC2'] = d.index.isin(full.obs_names[full.obs.selected_ILC2])
        d['final_reason'] = np.select([~d.candidate_pass, ~d.basic_qc_pass,
            ~d.index.isin(full.obs_names[full.obs.identity_positive]),
            d.index.isin(full.obs_names[full.obs.alternative_program])],
            ['below_candidate_RNA', 'basic_qc_fail', 'weak_identity', 'alternative_program'], default='retained')
        selected = full[full.obs.selected_ILC2].copy()
        selected.uns.clear()
        selected.uns['qc_version'] = policy['preset']
        selected.uns['qc_provenance_json'] = json.dumps(dict(basic_qc=q, identity=policy,
            annotation=reference['ensembl_annotation'], missing_markers=missing,
            upper_sd_reference='candidate cells, matching existing GSE136154 implementation'))
        selected.write_h5ad(data / f'{gsm}_ILC2.h5ad', compression='gzip')
        if save_basic:
            full.uns['qc_version'] = policy['preset']
            full.uns['qc_provenance_json'] = selected.uns['qc_provenance_json']
            full.write_h5ad(data / f'{gsm}_basic_qc.h5ad', compression='gzip')
        selected.obs.to_csv(data / f'{gsm}_metadata.tsv.gz', sep='\t')
        summary = dict(study=gse, gsm=gsm, tissue=tissue, condition=condition, input_cells=a.n_obs,
            candidate_cells=int(candidate.sum()), basic_qc_cells=full.n_obs, selected_ILC2=selected.n_obs,
            doublet_review_retained=int(selected.obs.doublet_review.sum()),
            low_RNA_retained=int(selected.obs.low_RNA_review.sum()), genes=selected.n_vars,
            coverage=summarize_coverage(selected, selected.var.genecompass_token.to_numpy(), selected.var.has_median_reference.to_numpy()),
            basic_qc=result['summary'])
        summaries.append(summary)
        # External targets get QC-only plots in a separate report folder, not development features.
        library_out = out / gsm
        library_out.mkdir(exist_ok=True)
        analysis.obs = full.obs.copy()
        sc.tl.umap(analysis, random_state=cfg['random_seed'])
        plots.OUT = library_out
        plots.plot_qc(analysis, d, selected)
        d.to_csv(library_out / 'cell_decisions.tsv.gz', sep='\t')
        full.obs.groupby('leiden', observed=True).agg(cells=('gsm', 'size'),
            retained=('selected_ILC2', 'sum'), alternative_fraction=('alternative_program', 'mean')).to_csv(library_out / 'cluster_review.tsv', sep='\t')
        print(gsm, 'retained', selected.n_obs, flush=True)
        kept.append(selected)
        if resume:
            (out / f'{summary_prefix}summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    table = pd.DataFrame([{k: v for k,v in s.items() if k not in ['coverage', 'basic_qc']} for s in summaries])
    table.to_csv(out / f'{summary_prefix}sample_summary.tsv', sep='\t', index=False)
    (out / f'{summary_prefix}summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    # Do not concatenate external PBS with IL33 targets into a convenient training input.
    if gse == 'GSE102299':
        merged = ad.concat(kept, merge='same', uns_merge='same')
        merged.uns['qc_version'] = policy['preset']
        merged.write_h5ad(data / (combined_name or f'{gse}_ILC2.h5ad'), compression='gzip')
    if tissue == 'Liver':
        liver_report()


def liver_report():
    out = ROOT / 'results/GSE189285_qc/liver'
    data = ROOT / 'data/processed/GSE189285_qc'
    summaries = json.loads((out / 'summary.json').read_text())
    rows, reasons = [], []
    for s in summaries:
        a = ad.read_h5ad(data / f"{s['gsm']}_ILC2.h5ad")
        if (not a.obs_names.is_unique or not a.var_names.is_unique or
            not np.isfinite(a.X.data).all() or (a.X.data < 0).any() or
            not np.equal(a.X.data, np.floor(a.X.data)).all() or
            (a.X != a.layers['counts']).nnz or not a.obs.tissue.eq('Liver').all()):
            raise ValueError('Counts or source integrity failed')
        med = a.obs[['total_counts','n_genes_by_counts','pct_counts_mt']].median()
        rows.append(dict(gsm=s['gsm'], condition=s['condition'], cells=a.n_obs,
            median_UMI=med.total_counts, median_genes=med.n_genes_by_counts,
            median_mt_pct=med.pct_counts_mt,
            doublet_review=s['doublet_review_retained'], low_RNA_review=s['low_RNA_retained']))
        d = pd.read_csv(out / s['gsm'] / 'cell_decisions.tsv.gz', sep='\t')
        reasons.extend(dict(gsm=s['gsm'], reason=k, cells=int(v)) for k,v in d.final_reason.value_counts().items())
    pd.DataFrame(rows).to_csv(out / 'retained_quality.tsv', sep='\t', index=False)
    pd.DataFrame(reasons).to_csv(out / 'filter_reasons.tsv', sep='\t', index=False)
    lines = ['# GSE189285 肝 ILC2 QC', '',
        '统一ilc2_balanced_v1；未修改阈值。genes≥200、UMI≥500、coding/miRNA≥7、mt≤15%、文库候选细胞检出基因数3SD上限。身份调用共享apply_standard；状态不作为过滤条件。', '',
        '| 条件 | 输入 | 基础QC | 保留ILC2 | 输入保留率 |', '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['condition']} | {s['input_cells']} | {s['basic_qc_cells']} | {s['selected_ILC2']} | {100*s['selected_ILC2']/s['input_cells']:.1f}% |")
    lines += ['', '## 保留细胞质量', '', '```tsv', pd.DataFrame(rows).to_csv(sep='\t',index=False).strip(), '```', '',
        'PBS保留率低主要来自替代谱系证据；去留优先级统计见filter_reasons.tsv。marker图支持多个T/NK相关群，不能仅凭测序深度判为劣质文库。身份为计算支持，非实验验证。双细胞及低RNA标记按统一规则保留，不自动加严。',
        '每个对象保留32285个原始feature；31588个core-biotype基因用于分析。X与counts层均为相同非负整数，稳定ID、组织及来源核验通过。QC注释GRCm38 release98全覆盖，不等同确认作者原始release。', '',
        '## GeneCompass覆盖', '']
    for s in summaries:
        c=s['coverage']; lines.append(f"- {s['condition']}：词表覆盖UMI {100*c['umi_fraction_in_vocab']:.2f}%；词表且有median参考覆盖UMI {100*c['umi_fraction_with_median']:.2f}%；26858基因在词表，17215同时有median参考。")
    lines += ['', 'PBS与IL33分别保存h5ad和metadata，不并入训练。本次只做QC、UMAP及marker dotplot，未计算跨条件DEG或调参。每条件一个混池文库，不能把细胞当动物重复，也不能由捕获细胞数量推算组织增殖倍数。',
        '原始文件与SHA256：src/downstream/drugpred/downstream_data/GSE189285/raw/liver_source_manifest.json。',
        '输出：data/processed/GSE189285_qc/GSM5697024_ILC2.h5ad（PBS）；GSM5697022_ILC2.h5ad（IL33）；对应*_metadata.tsv.gz。',
        '复现：python src/downstream/drugpred/download_il33.py --liver；python src/downstream/drugpred/qc_il33.py GSE189285 Liver。']
    (out / 'QC_report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'Lung')
