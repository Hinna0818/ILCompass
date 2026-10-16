"""Read-only inspection of the five GEO matrices; does not filter or normalize."""
from pathlib import Path
import csv, gc, gzip, json, os, platform
from datetime import datetime

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT.parent / "ILC2_rotation_proposal/GSE117567_initial_inspection"
SRC = PROJECT / 'data/GSE117567_RAW'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/ilc2_inspection_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
from threadpoolctl import threadpool_limits
import psutil

SAMPLES = [
    ('GSM3303970_BAM_', 'Bone marrow', '骨髓'),
    ('GSM3303971_Adult_Lung_', 'Lung', '肺'),
    ('GSM3303972_Adult_skin_', 'Skin', '皮肤'),
    ('GSM3303973_Adult_gut_', 'Small intestine LP', '小肠LP'),
    ('GSM3303974_FAT_', 'Visceral adipose', '内脏脂肪'),
]
MARKERS = {
    'ILC2-associated': ['Gata3', 'Rora', 'Il7r', 'Il1rl1', 'Icos', 'Klrg1', 'Arg1', 'Il5', 'Il13', 'Areg'],
    'Immature/state-associated': ['Tcf7', 'Zbtb16', 'Kit', 'Il18r1', 'Mki67'],
    'Other-lineage-associated': ['Cd3d', 'Cd3e', 'Trac', 'Ms4a1', 'Cd79a', 'Lyz2', 'Csf1r', 'Rorc', 'Tbx21', 'Ncr1', 'Nkg7', 'Eomes', 'Epcam', 'Pecam1'],
}
OUT.mkdir(parents=True, exist_ok=True)
resources = {'cpu_count': os.cpu_count(), 'available_memory_GB': psutil.virtual_memory().available / 1e9,
             'free_disk_GB': psutil.disk_usage(OUT).free / 1e9,
             'strategy': 'sequential sparse matrices; no dense full matrix; no GPU'}
(OUT / 'resources.json').write_text(json.dumps(resources, indent=2))
summaries, qc_frames, marker_rows, checks = [], [], [], []
reference_genes = None

for prefix, tissue, tissue_cn in SAMPLES:
    print(f'Reading {prefix}', flush=True)
    files = {kind: SRC / (prefix + kind + '.gz') for kind in ('matrix.mtx', 'genes.tsv', 'barcodes.tsv')}
    assert all(f.is_file() for f in files.values()), files
    with gzip.open(files['genes.tsv'], 'rt') as handle:
        gene_rows = list(csv.reader(handle, delimiter='\t'))
    assert all(len(r) == 2 for r in gene_rows), 'Expected Ensembl ID and gene symbol'
    gene_ids = np.array([r[0] for r in gene_rows])
    symbols = np.array([r[1] for r in gene_rows])
    with gzip.open(files['barcodes.tsv'], 'rt') as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    assert len(barcodes) == len(set(barcodes)), 'Duplicate barcodes within library'
    with gzip.open(files['matrix.mtx'], 'rt') as handle:
        header = next(handle).strip()
        dimensions = next(line for line in handle if not line.startswith('%')).strip()
    with threadpool_limits(limits=2):
        counts = mmread(files['matrix.mtx']).tocsc()
    assert counts.shape == (len(gene_rows), len(barcodes)), (counts.shape, len(gene_rows), len(barcodes))
    assert np.isfinite(counts.data).all() and (counts.data >= 0).all()
    assert np.equal(counts.data, np.floor(counts.data)).all(), 'Not integer counts'
    original_nnz = counts.nnz
    counts.sum_duplicates()
    counts.eliminate_zeros()
    totals = np.asarray(counts.sum(axis=0)).ravel()
    genes_detected = np.diff(counts.indptr)
    mito = np.char.startswith(np.char.lower(symbols), 'mt-')
    mt_counts = np.asarray(counts[mito, :].sum(axis=0)).ravel()
    mt_pct = np.divide(100 * mt_counts, totals, out=np.full(len(totals), np.nan), where=totals > 0)
    sample = prefix[:10]
    q = pd.DataFrame({'sample': sample, 'tissue': tissue, 'barcode': barcodes,
                      'cell_id': [sample + ':' + b for b in barcodes],
                      'total_counts': totals, 'n_genes_by_counts': genes_detected,
                      'pct_counts_mt': mt_pct})
    qc_frames.append(q)
    row = {'sample': sample, 'tissue': tissue, 'tissue_cn': tissue_cn,
           'n_genes_in_matrix': counts.shape[0], 'n_barcodes': counts.shape[1],
           'nonzero_entries': counts.nnz, 'density_pct': 100 * counts.nnz / np.prod(counts.shape),
           'total_UMI': int(totals.sum()), 'median_UMI': float(np.median(totals)),
           'median_genes': float(np.median(genes_detected)), 'median_mt_pct': float(np.nanmedian(mt_pct)),
           'zero_count_barcodes': int((totals == 0).sum()),
           'barcodes_lt200_genes': int((genes_detected < 200).sum()),
           'barcodes_lt500_genes': int((genes_detected < 500).sum()),
           'barcodes_mt_gt10_pct': int((mt_pct > 10).sum()),
           'barcodes_mt_gt20_pct': int((mt_pct > 20).sum()),
           'illustrative_genes200_mt20_n': int(((genes_detected >= 200) & (mt_pct < 20)).sum()),
           'illustrative_genes500_mt20_n': int(((genes_detected >= 500) & (mt_pct < 20)).sum()),
           'mitochondrial_gene_rows': int(mito.sum()),
           'duplicate_gene_ID_rows': len(gene_ids) - len(set(gene_ids)),
           'duplicate_symbol_rows': len(symbols) - len(set(symbols)),
           'integer_nonnegative_counts': True,
           'gene_order_same_as_BM': reference_genes is None or gene_rows == reference_genes}
    for column, values in [('UMI', totals), ('genes', genes_detected), ('mt_pct', mt_pct)]:
        for quantile, value in zip([1, 5, 25, 75, 95, 99], np.nanpercentile(values, [1, 5, 25, 75, 95, 99])):
            row[f'{column}_p{quantile}'] = float(value)
    summaries.append(row)
    checks.append({'sample': sample, 'matrix_header': header, 'matrix_dimensions': dimensions,
                   'nnz_before_cleanup': original_nnz, 'nnz_after_cleanup': counts.nnz,
                   'files': {k: {'path': str(f), 'bytes': f.stat().st_size} for k, f in files.items()}})
    for category, markers in MARKERS.items():
        for marker in markers:
            hit = np.flatnonzero(symbols == marker)
            values = np.asarray(counts[hit, :].sum(axis=0)).ravel() if len(hit) else None
            marker_rows.append({'sample': sample, 'tissue': tissue, 'category': category, 'gene': marker,
                                'gene_rows': len(hit), 'detected_cells': int((values > 0).sum()) if values is not None else None,
                                'detected_pct': float(100 * (values > 0).mean()) if values is not None else None,
                                'mean_raw_UMI': float(values.mean()) if values is not None else None})
    if reference_genes is None:
        reference_genes = gene_rows
        pd.DataFrame(gene_rows, columns=['gene_id', 'gene_symbol']).to_csv(OUT / 'gene_reference.tsv', sep='\t', index=False)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    del counts
    gc.collect()

summary = pd.DataFrame(summaries)
qc = pd.concat(qc_frames, ignore_index=True)
markers = pd.DataFrame(marker_rows)
assert len(qc) == int(summary.n_barcodes.sum()) == 35396
assert qc.cell_id.is_unique and summary.gene_order_same_as_BM.all()
summary.to_csv(OUT / 'sample_summary.tsv', sep='\t', index=False)
qc.to_csv(OUT / 'cell_qc_metrics.tsv.gz', sep='\t', index=False, compression='gzip')
markers.to_csv(OUT / 'marker_detection.tsv', sep='\t', index=False)
(OUT / 'file_integrity.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
labels = ['BM', 'Lung', 'Skin', 'Gut', 'Fat']
for ax, metric, label in zip(axes, ['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], ['UMIs per barcode', 'Detected genes per barcode', 'Mitochondrial UMIs (%)']):
    values = [frame[metric].dropna().to_numpy() for frame in qc_frames]
    ax.boxplot(values, tick_labels=labels, showfliers=False, widths=.55)
    ax.set_ylabel(label)
    if metric != 'pct_counts_mt':
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=.2)
fig.suptitle('GSE117567: unfiltered inspection (outliers omitted only from display)', fontsize=12)
fig.savefig(OUT / 'qc_overview.png', dpi=180)
plt.close(fig)

gene_list = [gene for group in MARKERS.values() for gene in group]
heat = markers.pivot(index='gene', columns='sample', values='detected_pct').reindex(index=gene_list, columns=summary['sample'])
fig, ax = plt.subplots(figsize=(7.8, 9.2), constrained_layout=True)
cmap = plt.colormaps['Blues'].copy()
cmap.set_bad('#D9D9D9')
im = ax.imshow(heat.to_numpy(), vmin=0, vmax=100, aspect='auto', cmap=cmap)
ax.set_xticks(range(5), labels)
ax.set_yticks(range(len(gene_list)), gene_list)
ax.set_title('Raw-count marker detection (% of all input barcodes)\nGray: gene absent from supplied annotation')
fig.colorbar(im, ax=ax, label='Detected barcodes (%)', shrink=.8)
fig.savefig(OUT / 'marker_detection.png', dpi=180)
plt.close(fig)

short_cols = ['sample', 'tissue_cn', 'n_genes_in_matrix', 'n_barcodes', 'median_UMI', 'median_genes', 'median_mt_pct']
def markdown_table(df):
    return '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---'] * len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False, name=None))
report = f'''# GSE117567 数据摸底

检查时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。
输入目录：`{SRC}`。5个组织、15个gz文件，总文件大小{sum(f.stat().st_size for f in SRC.glob('*.gz'))/1024**2:.2f} MiB。

## 文件与矩阵
每个样本为Matrix Market整数计数矩阵、两列genes（Ensembl ID/符号）和barcode列表。原矩阵为基因×细胞，进入AnnData需转置。
五个矩阵均完整读入、通过gzip解压校验，行列数与genes/barcodes匹配。计数为有限非负整数。全部样本基因顺序相同。
共35,396个barcode、每矩阵27,998个基因。Ensembl ID应保留，重复符号不可不经处理直接当唯一基因键。

## 初步质量分布
下面统计所有输入barcode，未做任何筛选。

{markdown_table(summary[short_cols].round(2))}

完整分位数、示例阈值的数量见sample_summary.tsv。示例阈值只描述数据分布，不是确定的QC规则，不代表已确认的成熟ILC2或最终训练数量。
检测基因<200的barcode共{int(summary.barcodes_lt200_genes.sum())}条；<500共{int(summary.barcodes_lt500_genes.sum())}条；线粒体比例>20%共{int(summary.barcodes_mt_gt20_pct.sum())}条。
仅用“基因>=200且线粒体<20%”描述性计数会保留{int(summary.illustrative_genes200_mt20_n.sum())}条；使用基因>=500会保留{int(summary.illustrative_genes500_mt20_n.sum())}条。本次没有真的筛掉细胞，也没有用这些数量替代最终QC/身份合格数。
皮肤mt比例p95约8.97%、p99约15.69%，高mt尾部比其他组织突出。各组织需分别看质量分布。

![QC分布](qc_overview.png)

## 基本生物学背景
依据GEO原始设计：8–14周龄，WT C57BL/6背景的报告鼠，稳态激活ILC2。肺/肠/皮肤/脂肪使用Lin-CD45+Red5+分选；骨髓使用Lin-CD45+Yarg+。
来源：https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117567
五个组织各1个公开文库。该目录没有动物ID、逐细胞亚群标签、分选门信息文件或原始FASTQ。文库不能自动当成独立动物重复。
无外源刺激不等于功能静息；报告鼠分选有选择偏差。骨髓需单独复核前体/成熟状态。

## marker摸底
marker_detection.tsv/PNG仅报告某基因raw count>0的比例；未完成注释。ILC2相关marker、前体/状态marker以及其他谱系相关marker用于下一阶段检查。
dropout、环境RNA和doublet都可能影响单个marker；不能凭Gata3阳性或阴性直接保留/删除细胞。
实测Gata3检测率为73.6%–95.9%；皮肤和肠道Il1rl1仅9.6%与10.5%，而皮肤Il18r1为61.6%。不能用所有细胞必须表达ST2/Il1rl1作为ILC2保留规则。
骨髓Ncr1、Tbx21、Nkg7的检测率分别约25.0%、10.0%、38.9%，需要聚类与多marker共表达核验成熟ILC2、前体及其他ILC谱系，不能自动全保留或全删除。
脂肪Epcam检测率约23.9%，肠道Pecam1约23.3%，需进一步检查表达强度和共表达，不能仅据检测到一个分子判定污染。
Trac未出现在这套提供的基因符号表中；这是参考注释缺失/命名问题，不能解释为T细胞标记不表达。29个所查marker中28个有对应gene row。

![marker检测率](marker_detection.png)

## 下一步Scanpy入口
按5个prefix分别读取，确认genes.tsv的旧版10x格式，转置为cells×genes，并用GSM:barcode作为全局唯一cell_id。
本文件组合为“旧版两列genes.tsv + gzip + 样本prefix”；建议显式读三个文件再构造AnnData，避免自动读取器误找features.tsv.gz。
先建sample/tissue/GEO/steady-state/reporter等可核实metadata；animal_id与biological replicate留空，勿从组织名推断。
保留完整Ensembl基因ID与未归一化计数，再逐文库确定QC、doublet和细胞身份。不要在GeneCompass输入准备前只保留HVG、进行scale或覆盖原始计数。
组织与文库完全对应，不宜把所有组织差异作为batch消除。聚类分析的处理矩阵和模型所需输入应分开保存。

## 本次范围
完成矩阵读取、完整性、计数属性、逐barcode基本QC指标及marker检测率。未运行Scanpy筛选、归一化、HVG、聚类、doublet检测、批次整合或GeneCompass训练；未修改输入文件。
'''
(OUT / 'GSE117567_数据摸底.md').write_text(report)
print('Inspection complete: ' + str(OUT), flush=True)
