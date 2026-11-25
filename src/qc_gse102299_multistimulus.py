"""Download six audited IL25/NMU libraries and reuse the project's balanced_v1 QC."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/ilc2_numba'
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/downstream/drugpred'))
import qc_il33 as shared
from posttrain_data import file_hash
from qc_utils import _raw_counts

RAW = ROOT / 'data/raw/GSE102299'
DATA = ROOT / 'data/processed/GSE102299_qc'
OUT = ROOT / 'results/GSE102299_qc'
LEGACY = ROOT / 'src/downstream/drugpred/downstream_data/GSE102299'
INVENTORY = ROOT / 'results/GSE102299_IL25_NMU_inventory_v1'
SAMPLES = [
    ('GSE102299', 'GSM2733480', 'IL25_Rep1', 'IL25', '1'),
    ('GSE102299', 'GSM2733481', 'IL25_Rep2', 'IL25', '2'),
    ('GSE102299', 'GSM2733484', 'NMU_Rep1', 'NMU', '1'),
    ('GSE102299', 'GSM2733485', 'NMU_Rep2', 'NMU', '2'),
    ('GSE102299', 'GSM2733486', 'NMU_IL25_Rep1', 'IL25_NMU', '1'),
    ('GSE102299', 'GSM2733487', 'NMU_IL25_Rep2', 'IL25_NMU', '2'),
]


def checked_gzip(path):
    # Read through the trailer to detect incomplete/HTML downloads, not just a gzip header.
    with gzip.open(path, 'rb') as handle:
        while handle.read(1024 * 1024):
            pass


def download_file(row):
    path = RAW / row['filename']
    if not path.exists():
        print('DOWNLOAD', row['filename'], flush=True)
        temporary = path.with_name(path.name + '.part')
        # This GEO endpoint rejects Range requests, so replace incomplete .part files.
        with urllib.request.urlopen(row['url'], timeout=120) as response:
            expected = int(response.headers['Content-Length'])
            with temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
        if temporary.stat().st_size != expected:
            raise ValueError(f'Incomplete download: {temporary.name}; rerun to download again')
        checked_gzip(temporary); temporary.replace(path)
    else:
        checked_gzip(path)
    return dict(file=path.name, url=row['url'], bytes=path.stat().st_size, sha256=file_hash(path))


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    known = {r['file']: r for r in json.loads((LEGACY / 'raw/source_manifest.json').read_text())}
    records = []
    for name in ('GSE102299_genes.tsv.gz', 'GSE102299_family.soft.gz'):
        source = LEGACY / 'raw' / name
        if file_hash(source) != known[name]['sha256']:
            raise ValueError(f'Existing reference file changed: {name}')
        destination = RAW / name
        if not destination.exists():
            shutil.copy2(source, destination)
        if file_hash(destination) != known[name]['sha256']:
            raise ValueError(f'Canonical reference differs: {name}')
        records.append(known[name])
    required = pd.read_csv(INVENTORY / 'required_files.csv')
    required = required[required.gsm.isin([s[1] for s in SAMPLES])]
    if len(required) != 12 or required.filename.duplicated().any():
        raise ValueError('Expected exactly matrix/barcode files for the six audited libraries')
    with ThreadPoolExecutor(max_workers=3) as pool:
        records.extend(pool.map(download_file, required.to_dict('records')))
    (RAW / 'source_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    metadata = pd.read_csv(ROOT / 'results/GSE102299_GSE136154_GSE272724_feasibility/GSE102299_samples.tsv', sep='\t')
    metadata[metadata.gsm.isin([s[1] for s in SAMPLES])].to_csv(RAW / 'geo_samples.tsv', sep='\t', index=False)
    print('DOWNLOAD complete:', len(required), 'new data files', flush=True)


def qc():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW / 'source_manifest.json').read_text())
    for entry in manifest:
        if file_hash(RAW / entry['file']) != entry['sha256']:
            raise ValueError(f'Download changed: {entry["file"]}')
    shared.main('GSE102299', libraries=SAMPLES, raw_directory=RAW,
                data_directory=DATA, report_directory=OUT, summary_prefix='IL25_NMU_', resume=True)
    cfg = json.loads((ROOT / 'configs/1QC.json').read_text())
    metadata = pd.read_csv(RAW / 'geo_samples.tsv', sep='\t').set_index('gsm')
    by_name = {r['file']: r for r in manifest}
    parts = []
    for _, gsm, title, condition, replicate in SAMPLES:
        path = DATA / f'{gsm}_ILC2.h5ad'; a = ad.read_h5ad(path)
        row = metadata.loc[gsm]
        for key, value in dict(author_condition=row.characteristics.split('treatment: ')[1].split(' | ')[0],
            author_characteristics=row.characteristics, treatment_protocol=row.treatment_protocol,
            experimental_context='in_vivo', treatment_duration='3 consecutive days', author_timepoint='not_available',
            source_matrix_file=f'{gsm}_{title}.matrix.mtx.gz',
            source_matrix_url=by_name[f'{gsm}_{title}.matrix.mtx.gz']['url'],
            qc_annotation_release=str(cfg['core_genes']['ensembl_release'])).items():
            a.obs[key] = value
        a.uns['source_files_json'] = json.dumps([r for r in manifest if r['file'].startswith(gsm)])
        a.write_h5ad(path, compression='gzip')
        a.obs.to_csv(DATA / f'{gsm}_metadata.tsv.gz', sep='\t', index_label='cell_id')
        parts.append(a)
    # Keep the four historical PBS/IL33 objects unchanged; copy their exact balanced_v1 outputs.
    existing = []
    for gsm in ('GSM2733478', 'GSM2733479', 'GSM2733482', 'GSM2733483'):
        original = LEGACY / f'qc/{gsm}_ILC2.h5ad'; a = ad.read_h5ad(original)
        provenance = json.loads(a.uns['qc_provenance_json'])
        if (not a.obs.qc_version.eq(cfg['identity']['preset']).all()
                or provenance['identity'] != cfg['identity'] or provenance['basic_qc'] != cfg['basic_qc']):
            raise ValueError('Historical samples are not the same balanced_v1 policy')
        shutil.copy2(original, DATA / original.name)
        a.obs.to_csv(DATA / f'{gsm}_metadata.tsv.gz', sep='\t', index_label='cell_id')
        existing.append(dict(gsm=gsm, source=str(original.relative_to(ROOT)), sha256=file_hash(original)))
        parts.append(a)
    if any(not a.var_names.equals(parts[0].var_names) for a in parts):
        raise ValueError('Measured gene IDs/order differ; do not silently intersect or zero-fill')
    merged = ad.concat(parts, join='inner', merge='same', uns_merge='same')
    if not merged.obs_names.is_unique:
        raise ValueError('Source cell IDs overlap')
    merged.uns['qc_version'] = cfg['identity']['preset']
    merged.uns['qc_provenance_json'] = json.dumps(dict(basic_qc=cfg['basic_qc'], identity=cfg['identity'],
        annotation=cfg['core_genes'], new_inputs=manifest, reused_objects=existing,
        author_batch_and_animal='not_available; replicate labels retained without inferring pairing',
        counts='Original integers; all measured features retained; normalization used only for QC plots'))
    temporary = DATA / 'GSE102299_ILC2.tmp.h5ad'
    merged.write_h5ad(temporary, compression='gzip'); temporary.replace(DATA / 'GSE102299_ILC2.h5ad')
    merged.obs.to_csv(DATA / 'GSE102299_ILC2_metadata.tsv.gz', sep='\t', index_label='cell_id')
    fields = ['gsm', 'condition', 'author_replicate', 'author_batch', 'tissue', 'species', 'strain',
              'genotype', 'animal_id', 'role', 'split', 'qc_version']
    merged.obs[fields].drop_duplicates().to_csv(DATA / 'GSE102299_sample_metadata.tsv', sep='\t', index=False)
    print('MERGED', merged.shape, 'all ten libraries; historical objects untouched', flush=True)


def validate():
    cfg = json.loads((ROOT / 'configs/1QC.json').read_text())
    new_summary = json.loads((OUT / 'IL25_NMU_summary.json').read_text())
    old_summary = json.loads((OUT / 'summary.json').read_text())
    summary = {s['gsm']: s for s in [*old_summary, *new_summary]}
    rows, verification, reasons = [], [], []
    source_map = {s[1]: s for s in SAMPLES}
    parts = []
    for gsm, entry in sorted(summary.items()):
        a = ad.read_h5ad(DATA / f'{gsm}_ILC2.h5ad')
        counts = _raw_counts(a.X)
        if (not a.var_names.is_unique or not a.obs_names.is_unique or (counts != a.layers['counts']).nnz
                or not a.obs.gsm.eq(gsm).all() or not a.obs.selected_ILC2.all()
                or not a.obs.qc_version.eq(cfg['identity']['preset']).all()
                or not a.obs.author_batch.eq('not_available').all() or not a.obs.animal_id.eq('not_available').all()):
            raise ValueError(f'Invalid counts/identity/source boundary: {gsm}')
        np.testing.assert_array_equal(np.asarray(counts.sum(1)).ravel(), a.obs.total_counts)
        np.testing.assert_array_equal(np.diff(counts.indptr), a.obs.n_genes_by_counts)
        if not np.isfinite(a.obs.doublet_score).all():
            raise ValueError(f'Nonfinite doublet scores: {gsm}')
        if gsm in source_map:
            raw = shared.read_library(*source_map[gsm][:3], raw_directory=RAW)
            loc = raw.obs_names.get_indexer(a.obs_names)
            if (loc < 0).any() or not raw.var_names.equals(a.var_names) or (raw.X[loc] != counts).nnz:
                raise ValueError(f'Retained counts differ from raw source: {gsm}')
            check = 'Every retained gene/cell compared with original Matrix Market counts'
        else:
            original = ad.read_h5ad(LEGACY / f'qc/{gsm}_ILC2.h5ad')
            if not original.obs_names.equals(a.obs_names) or (original.X != counts).nnz:
                raise ValueError(f'Historical retained cells changed: {gsm}')
            check = 'Exact retained cell/count equality with verified historical balanced_v1 object'
        medians = a.obs[['total_counts', 'n_genes_by_counts', 'pct_counts_mt']].median()
        rows.append(dict(gsm=gsm, condition=str(a.obs.condition.iloc[0]), replicate=str(a.obs.author_replicate.iloc[0]),
            input_cells=entry['input_cells'], candidate_cells=entry['candidate_cells'], basic_qc_cells=entry['basic_qc_cells'],
            selected_ILC2=a.n_obs, retention_pct=100 * a.n_obs / entry['input_cells'], genes=a.n_vars,
            median_UMI=medians.total_counts, median_genes=medians.n_genes_by_counts, median_mt_pct=medians.pct_counts_mt,
            doublet_review_retained=int(a.obs.doublet_review.sum()), low_RNA_retained=int(a.obs.low_RNA_review.sum())))
        verification.append(dict(gsm=gsm, passed=True, validation=check))
        decision = pd.read_csv(OUT / gsm / 'cell_decisions.tsv.gz', sep='\t', usecols=['final_reason'])
        reasons.extend(dict(gsm=gsm, reason=reason, cells=int(n)) for reason, n in decision.final_reason.value_counts().items())
        parts.append(a)
    merged = ad.read_h5ad(DATA / 'GSE102299_ILC2.h5ad')
    expected = ad.concat(parts, join='inner', merge='same')
    if (not merged.obs_names.is_unique or set(expected.obs_names) != set(merged.obs_names)
            or not merged.var_names.equals(expected.var_names)
            or (merged.X != expected[merged.obs_names].X).nnz or (merged.X != merged.layers['counts']).nnz):
        raise ValueError('Merged corpus lost/duplicated cells, genes or counts')
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'all_conditions_sample_summary.tsv', sep='\t', index=False)
    condition = table.groupby('condition')[['input_cells', 'basic_qc_cells', 'selected_ILC2']].sum()
    condition.to_csv(OUT / 'all_conditions_summary.tsv', sep='\t')
    pd.DataFrame(reasons).to_csv(OUT / 'all_conditions_filter_reasons.tsv', sep='\t', index=False)
    validation = dict(passed=True, libraries=verification, merged_cells=merged.n_obs, merged_genes=merged.n_vars,
        raw_integer_counts=True, unique_source_ids=True, unchanged_historical_cells=True,
        code_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in
            [Path(__file__), Path(shared.__file__), ROOT / 'src/qc_utils.py', ROOT / 'src/ilc2_qc.py', ROOT / 'configs/1QC.json']},
        software={p: importlib.metadata.version(p) for p in ['scanpy', 'anndata', 'numpy', 'scipy', 'scrublet']})
    (OUT / 'IL25_NMU_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    lines = ['# GSE102299 多刺激 ILC2 balanced_v1 QC', '',
        '新补齐IL25、NMU、IL25+NMU六文库；按GSM独立基础QC、Scrublet和身份规则，阈值完全沿用configs/1QC.json，无额外加严。',
        '四个历史PBS/IL33文库经版本、阈值及counts一致性验证后原样复用；原路径对象和旧四文库summary保持不变。', '',
        '```tsv\n' + table.to_csv(sep='\t', index=False).strip() + '\n```', '',
        '## 条件汇总', '', '```tsv\n' + condition.to_csv(sep='\t').strip() + '\n```', '',
        f'合并对象：{merged.n_obs:,}细胞×{merged.n_vars:,}共同实测基因。X和counts层均为原始整数；六新文库每个保留细胞/基因均与原始矩阵核对。',
        '基因≥200、UMI≥500、coding/miRNA≥7、mt≤15%；候选细胞均值+3SD上限按文库计算。身份使用共享apply_standard；效应/增殖/干扰素状态不作为独立删除规则。',
        'doublet_review和low_RNA_review按标准保留/标记，不等于实验确认singlet或前体；高置信度仅表示计算身份支持。',
        'GRCm38 release84为本项目QC注释；作者使用修改过Nmur1 3端的mm10参考，原始注释release未确认。',
        'Replicate 1/2、GSM和原始barcode保留；author_batch/animal_id仍缺失，不推断动物配对或独立批次。',
        '新文库只生成qc.png、umap.png、marker_dotplot.png；每细胞去留、群复核、软件/来源记录保留。归一化和聚类仅用于QC展示，不作为下游训练特征。', '',
        'Marker图已人工查看，复核记录见IL25_NMU_marker_review.tsv：主要保留群有ILC2核心标记支持，同时存在CD3相关/混合群，按单细胞规则排除或标记，没有额外整群剔除；效应与增殖差异不参与身份否决。仍可能残留背景或同型双细胞，不能称为实验确认的纯ILC2。', '',
        '合并counts：data/processed/GSE102299_qc/GSE102299_ILC2.h5ad；各GSM对象和metadata位于同目录。',
        '原始文件及下载URL/SHA256：data/raw/GSE102299/source_manifest.json。验证：results/GSE102299_qc/IL25_NMU_validation.json。',
        '复现：python src/qc_gse102299_multistimulus.py download；python src/qc_gse102299_multistimulus.py qc；python src/qc_gse102299_multistimulus.py validate。', '']
    (OUT / 'QC_report.md').write_text('\n'.join(lines))
    print(table.to_string(index=False)); print('VALIDATED', merged.shape, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'qc', 'validate', 'all'])
    args = parser.parse_args()
    with threadpool_limits(4):
        if args.command in ('download', 'all'):
            download()
        if args.command in ('qc', 'all'):
            qc()
        if args.command in ('qc', 'validate', 'all'):
            validate()


if __name__ == '__main__':
    main()
