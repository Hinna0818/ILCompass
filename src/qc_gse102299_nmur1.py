"""Download and balanced_v1 QC the eight Nmur1 WT/KO, PBS/HDM libraries."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/ilc2_numba'
os.environ['MPLCONFIGDIR'] = '/tmp/ilc2_mpl'

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/downstream/drugpred'))
import qc_il33 as shared
from qc_gse102299_multistimulus import download_file, checked_gzip
from posttrain_data import file_hash
from qc_utils import _raw_counts

RAW = ROOT / 'data/raw/GSE102299'
DATA = ROOT / 'data/processed/GSE102299_qc'
OUT = ROOT / 'results/GSE102299_qc'
PREFIX = 'Nmur1_KO_'
COMBINED = 'GSE102299_Nmur1_KO_ILC2.h5ad'
TARGET = 'ENSMUSG00000026237'


def libraries():
    geo = pd.read_csv(ROOT / 'results/GSE102299_GSE136154_GSE272724_feasibility/GSE102299_samples.tsv', sep='\t').set_index('gsm')
    samples, metadata = [], {}
    for number in range(2733488, 2733496):
        gsm = f'GSM{number}'; row = geo.loc[gsm]
        fields = dict(s.split(': ', 1) for s in row.characteristics.split(' | ') if ': ' in s)
        urls = row.supplementary_files.split(' | ')
        matrix = next(url.rsplit('/', 1)[1] for url in urls if url.endswith('.matrix.mtx.gz'))
        title = matrix.removeprefix(gsm+'_').removesuffix('.matrix.mtx.gz')
        if fields['organ'] != 'Lung' or fields['scrna-seq method'] != '10x':
            raise ValueError(f'Unexpected library: {gsm}')
        samples.append(('GSE102299', gsm, title, 'HDM' if fields['treatment'].startswith('House') else 'PBS', fields['replicate'].split()[-1]))
        metadata[gsm] = dict(genotype=fields['genotype'], author_genotype_label=fields['genotype'],
            author_condition=fields['treatment'], author_characteristics=row.characteristics,
            treatment_protocol=row.treatment_protocol, protocol_id='GSE102299_HDM_days_0_7_8_9',
            experimental_context='in_vivo', source_matrix_file=matrix,
            source_matrix_url=next(url for url in urls if url.endswith('.matrix.mtx.gz')).replace('ftp://', 'https://', 1),
            author_timepoint='one day after last treatment', pool_scope='lungs pooled within treatment group',
            author_reference_note='mm10 with corrected Nmur1 3-prime end; retain author counts')
    return samples, metadata, geo.loc[[s[1] for s in samples]].reset_index()


def download():
    samples, _, geo = libraries()
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = []
    for _, row in geo.iterrows():
        for url in row.supplementary_files.split(' | '):
            jobs.append(dict(filename=url.rsplit('/', 1)[1], url=url.replace('ftp://', 'https://', 1)))
    records = [download_matrix(row) if row['filename'].endswith('.matrix.mtx.gz') else download_file(row)
               for row in jobs]
    for name in ('GSE102299_genes.tsv.gz', 'GSE102299_family.soft.gz'):
        path = RAW / name
        records.append(dict(file=name, bytes=path.stat().st_size, sha256=file_hash(path),
            url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE102nnn/GSE102299/{"soft" if "soft" in name else "suppl"}/{name}'))
    (RAW / f'{PREFIX}source_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    geo.to_csv(RAW / f'{PREFIX}geo_samples.tsv', sep='\t', index=False)
    print('DOWNLOAD complete:', len(samples), 'libraries', flush=True)


def import_archive(path):
    """Import only the 16 audited regular files; never extract archive paths."""
    _, _, geo = libraries()
    urls = {url.rsplit('/', 1)[1]: url.replace('ftp://', 'https://', 1)
            for value in geo.supplementary_files for url in value.split(' | ')}
    records = []
    RAW.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        for name, url in urls.items():
            matches = [m for m in members if m.isfile() and Path(m.name).name == name]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one archive member: {name}')
            member = matches[0]
            destination = RAW / name
            temporary = RAW / (name+'.upload')
            with archive.extractfile(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            if temporary.stat().st_size != member.size:
                raise ValueError(f'Incomplete archive member: {name}')
            checked_gzip(temporary)
            digest = file_hash(temporary)
            if destination.exists():
                if file_hash(destination) != digest:
                    raise ValueError(f'Uploaded archive differs from existing file: {name}')
                temporary.unlink()
            else:
                temporary.replace(destination)
            records.append(dict(file=name, url=url, bytes=member.size, sha256=digest,
                archive_member=member.name))
            print('IMPORTED', name, member.size, flush=True)
    for name in ('GSE102299_genes.tsv.gz', 'GSE102299_family.soft.gz'):
        reference = RAW / name
        checked_gzip(reference)
        records.append(dict(file=name, bytes=reference.stat().st_size, sha256=file_hash(reference)))
    (RAW / f'{PREFIX}source_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    (RAW / f'{PREFIX}archive_import.json').write_text(json.dumps(dict(
        archive=str(path.resolve()), archive_sha256=file_hash(path), imported_files=len(urls)), indent=2)+'\n')
    geo.to_csv(RAW / f'{PREFIX}geo_samples.tsv', sep='\t', index=False)


def download_matrix(row):
    """NCBI direct HTTPS supports ranges; small parallel ranges avoid slow long streams."""
    path = RAW / row['filename']
    if path.exists():
        return download_file(row)
    with urllib.request.urlopen(urllib.request.Request(row['url'], method='HEAD'), timeout=30) as response:
        total = int(response.headers['Content-Length'])
        modified = response.headers['Last-Modified']
    temporary = path.with_name(path.name + '.part')
    start = temporary.stat().st_size if temporary.exists() else 0
    if start > total:
        raise ValueError(f'Partial file larger than remote source: {temporary}')
    chunk = 64 * 1024

    def fetch(offset):
        end = min(offset+chunk, total)-1
        request = urllib.request.Request(row['url'], headers={
            'Range': f'bytes={offset}-{end}', 'If-Range': modified})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{total}':
                        raise ValueError('NCBI range response does not match requested bytes')
                    block = response.read()
                break
            except (urllib.error.URLError, TimeoutError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 502, 503, 504):
                    raise
                if attempt == 2:
                    raise
                time.sleep(5*(attempt+1))
        if len(block) != end-offset+1:
            raise ValueError('Incomplete range response')
        return block

    print('RANGE DOWNLOAD', path.name, f'{start}/{total} bytes', flush=True)
    # Append only in order, so an interrupted .part is always a contiguous prefix.
    with ThreadPoolExecutor(max_workers=8) as pool, temporary.open('ab') as output:
        for block in pool.map(fetch, range(start, total, chunk)):
            output.write(block)
    if temporary.stat().st_size != total:
        raise ValueError(f'Incomplete file: {temporary}')
    checked_gzip(temporary)
    temporary.replace(path)
    print('DOWNLOADED', path.name, total, flush=True)
    return dict(file=path.name, url=row['url'], bytes=total, sha256=file_hash(path))


def qc():
    samples, metadata, _ = libraries()
    manifest = json.loads((RAW / f'{PREFIX}source_manifest.json').read_text())
    for record in manifest:
        if file_hash(RAW / record['file']) != record['sha256']:
            raise ValueError(f'Changed source: {record["file"]}')
    shared.main('GSE102299', libraries=samples, raw_directory=RAW, data_directory=DATA,
        report_directory=OUT, summary_prefix=PREFIX, resume=True, sample_metadata=metadata,
        combined_name=COMBINED, save_basic=True)


def validate():
    samples, metadata, _ = libraries()
    cfg = json.loads((ROOT / 'configs/1QC.json').read_text())
    summaries = {r['gsm']: r for r in json.loads((OUT / f'{PREFIX}summary.json').read_text())}
    rows, reasons, checks, parts = [], [], [], []
    for study, gsm, title, condition, replicate in samples:
        raw = shared.read_library(study, gsm, title, raw_directory=RAW)
        basic = ad.read_h5ad(DATA / f'{gsm}_basic_qc.h5ad')
        a = ad.read_h5ad(DATA / f'{gsm}_ILC2.h5ad')
        for obj in (basic, a):
            counts = _raw_counts(obj.X)
            indices = raw.obs_names.get_indexer(obj.obs_names)
            if ((indices < 0).any() or not obj.var_names.equals(raw.var_names) or (raw.X[indices] != counts).nnz
                    or (obj.X != obj.layers['counts']).nnz or not obj.obs_names.is_unique
                    or not obj.obs.gsm.eq(gsm).all() or not obj.obs.genotype.eq(metadata[gsm]['genotype']).all()
                    or not obj.obs.condition.eq(condition).all() or not obj.obs.qc_version.eq(cfg['identity']['preset']).all()):
                raise ValueError(f'Counts/source/QC boundary failed: {gsm}')
        if not a.obs.selected_ILC2.all() or not np.isfinite(basic.obs.doublet_score).all():
            raise ValueError(f'Invalid identity/scrublet result: {gsm}')
        summary = summaries[gsm]
        med = a.obs[['total_counts', 'n_genes_by_counts', 'pct_counts_mt']].median()
        target = np.asarray(a[:, [TARGET]].X.toarray()).ravel()
        rows.append(dict(gsm=gsm, genotype=metadata[gsm]['genotype'], condition=condition, replicate=replicate,
            input_cells=summary['input_cells'], candidate_cells=summary['candidate_cells'], basic_qc_cells=basic.n_obs,
            ILC2_cells=a.n_obs, retained_pct=100*a.n_obs/summary['input_cells'], genes=a.n_vars,
            median_UMI=med.total_counts, median_genes=med.n_genes_by_counts, median_mt_pct=med.pct_counts_mt,
            Nmur1_detected=int((target > 0).sum()), Nmur1_detected_pct=100*(target > 0).mean(),
            Nmur1_mean_UMI=float(target.mean()), doublet_review=int(a.obs.doublet_review.sum()),
            low_RNA_review=int(a.obs.low_RNA_review.sum())))
        d = pd.read_csv(OUT / gsm / 'cell_decisions.tsv.gz', sep='\t')
        reasons.extend(dict(gsm=gsm, reason=k, cells=int(v)) for k, v in d.final_reason.value_counts().items())
        checks.append(dict(gsm=gsm, raw_counts_exact_match=True, basic_cells=basic.n_obs, ILC2_cells=a.n_obs))
        parts.append(a)
    merged = ad.read_h5ad(DATA / COMBINED)
    expected = ad.concat(parts, merge='same')
    if (not merged.obs_names.is_unique or set(merged.obs_names) != set(expected.obs_names)
            or not merged.var_names.equals(expected.var_names) or (merged.X != expected[merged.obs_names].X).nnz):
        raise ValueError('Merged object lost/duplicated cells or counts')
    merged.uns['source_files_json'] = (RAW / f'{PREFIX}source_manifest.json').read_text()
    merged.uns['experiment'] = 'Nmur1 WT/KO by PBS/HDM; separate from alarmin experiment'
    merged.write_h5ad(DATA / COMBINED, compression='gzip')
    merged.obs.to_csv(DATA / 'GSE102299_Nmur1_KO_metadata.tsv.gz', sep='\t', index_label='cell_id')
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f'{PREFIX}sample_summary.tsv', sep='\t', index=False)
    grouped = table.groupby(['genotype', 'condition'])[['input_cells', 'basic_qc_cells', 'ILC2_cells', 'Nmur1_detected']].sum()
    grouped['Nmur1_detected_pct'] = 100*grouped.Nmur1_detected/grouped.ILC2_cells
    grouped.to_csv(OUT / f'{PREFIX}group_summary.tsv', sep='\t')
    pd.DataFrame(reasons).to_csv(OUT / f'{PREFIX}filter_reasons.tsv', sep='\t', index=False)
    validation = dict(passed=True, libraries=checks, merged_cells=merged.n_obs, genes=merged.n_vars,
        qc_preset=cfg['identity']['preset'], threshold_overrides=False, counts='Original nonnegative integers',
        software={p: importlib.metadata.version(p) for p in ('scanpy', 'anndata', 'numpy', 'scipy', 'scrublet')},
        code_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__), Path(shared.__file__), ROOT/'src/ilc2_qc.py', ROOT/'configs/1QC.json')})
    (OUT / f'{PREFIX}validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    report = ['# GSE102299 Nmur1 KO balanced_v1 QC', '',
        '8个10x肺ILC文库，WT/Nmur1 KO×PBS/HDM，每组2个作者重复。沿用configs/1QC.json和共享apply_standard，无阈值覆盖。', '',
        '```tsv', table.to_csv(sep='\t', index=False).strip(), '```', '', '## 分组汇总', '',
        '```tsv', grouped.to_csv(sep='\t').strip(), '```', '',
        f'独立合并对象：{merged.n_obs:,}个ILC2×{merged.n_vars:,}个实测基因；X和counts层为相同原始整数，基础QC对象和保留细胞均与作者原始矩阵逐项核对。',
        '原有42,541细胞的细胞因子实验对象未覆盖；本次KO对象独立命名。保留各GSM完整basic_qc对象、去留表和群复核，便于复核KO相关组成变化。',
        '基因≥200、UMI≥500、coding/miRNA≥7、mt≤15%，候选细胞的检出基因均值+3SD上限按文库独立计算；Scrublet每文库运行，高分及低RNA只标记，状态不作为身份否决。',
        'QC注释为项目GRCm38 release84；作者使用修正Nmur1 3′端的mm10参考，保留作者counts，不重新注释或把KO RNA零值作为身份门槛。',
        '每组重复为作者混池文库，不虚构动物级配对；animal_id和author_batch仍缺失。ILC2数量是测序保留数，不代表组织绝对丰度。高置信度表示计算身份支持，不等于独立实验验证。',
        '各GSM只生成QC、UMAP和marker dotplot三类图。尚未进行KO推理或模型训练。', '',
        '## 图形复核及限制', '',
        '8个文库的marker dotplot已逐一复核，观察记录见[Nmur1_KO_marker_review.tsv](Nmur1_KO_marker_review.tsv)。主要保留群有Gata3/Il7r/Id2/Rora等ILC2身份支持；其他群可见CD3、NK或髓系程序。复核未追加过滤，也未将整群直接删除；混合群中仍按共享逐细胞规则保留候选，因此不能宣称绝对纯净。',
        'KO+HDM Rep2保留率最低（65.62%），主要排除原因为替代谱系程序（1,175个），而非基础QC（69个）。这提示文库组成差异，不能仅凭测序保留数认定KO导致ILC2减少。',
        'GSM2733492（KO+PBS Rep1）Scrublet运行时整体双细胞率估计发生无效除法警告；已核验所有基础QC细胞的doublet_score为有限值。该整体率不用于解释或过滤，高分细胞仍只标记。本组仅925个ILC2，后续模型评价需注意重复间细胞数不均衡。',
        'WT中Nmur1检测为正的细胞比例约22%–23%，KO中低于0.3%，是本数据中的转录层面支持，不等价于逐细胞基因组敲除验证。检测为零也不代表受体功能必然不存在；后续token删除实验仅能直接改变目标基因实际进入序列的WT细胞。', '',
        '复现（使用本次上传归档）：`/root/.conda/envs/ilc2_model/bin/python src/qc_gse102299_nmur1.py all --archive scratch/geneko_scratch/data/GSE102299_RAW.tar`；仅重核验用`python src/qc_gse102299_nmur1.py validate`。']
    (OUT / f'{PREFIX}QC_report.md').write_text('\n'.join(report)+'\n')
    print(table.to_string(index=False), flush=True)
    print('VALIDATED', merged.shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['download', 'import', 'qc', 'validate', 'all'])
    parser.add_argument('--archive', type=Path, help='Uploaded GSE102299_RAW.tar; import without network access')
    args = parser.parse_args()
    with threadpool_limits(4):
        if args.stage == 'import' and args.archive is None:
            parser.error('import requires --archive')
        if args.stage == 'import' or (args.stage == 'all' and args.archive is not None):
            import_archive(args.archive)
        elif args.stage in ('download', 'all'):
            download()
        if args.stage in ('qc', 'all'): qc()
        if args.stage in ('validate', 'all'): validate()
