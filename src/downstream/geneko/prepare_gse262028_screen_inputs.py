"""Prepare auditable guide-count inputs; no enrichment tests or model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from qc_utils import read_gene_annotation


def write_table(df, path):
    df.to_csv(path, sep='\t', index=False)


def main(cfg):
    raw, out, report = [ROOT / cfg[k] for k in ['source', 'output', 'report']]
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw / 'source_manifest.json').read_text())
    for entry in manifest:
        content = (raw / entry['filename']).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f"Source changed: {entry['filename']}")
    comparisons = pd.read_csv(raw / 'author_comparisons.tsv', sep='\t')
    geo = pd.read_csv(raw / 'GSE262028_samples.tsv', sep='\t').set_index('Sample_title')
    annotation = read_gene_annotation(ROOT / cfg['annotation']).reset_index()
    symbol_map = annotation.groupby('gene_symbol').gene_id.agg(lambda x: sorted(set(x)))
    guides = None
    counts, samples, unused = {}, [], []
    for endpoint, screens in cfg['endpoints'].items():
        for replicate, screen in enumerate(screens, 1):
            d = pd.read_csv(raw / f'GSE262028_{screen}_lib_data.csv.gz', dtype={'gene_id': str, 'gene_symbol': str})
            if not d.UID.is_unique:
                raise ValueError('Duplicate guide IDs')
            g = d[['UID', 'sgRNA.Target.Sequence', 'gene_id', 'gene_symbol']].rename(columns={
                'UID': 'sgRNA', 'sgRNA.Target.Sequence': 'sequence', 'gene_id': 'author_gene_id', 'gene_symbol': 'author_symbol'}).set_index('sgRNA')
            if guides is None:
                guides = g
            else:
                if set(g.index) != set(guides.index):
                    raise ValueError('Guide sets differ across screens')
                pd.testing.assert_frame_equal(guides, g.loc[guides.index])
            d = d.set_index('UID').loc[guides.index]
            pools = pd.read_csv(raw / f'GSE262028_{screen}_screen_list.csv.gz')
            use = pools[pools.Screen.eq(screen)]
            if len(use) != 2 or set(use.Bin) != {'P', 'N'} or not use.Library.eq('TF').all():
                raise ValueError(f'Unexpected author comparison: {screen}')
            for _, row in pools[~pools.Screen.eq(screen)].iterrows():
                unused.append(dict(screen=screen, pool=f'pool.{int(row.Pool)}', author_filename=row['lib.file.name'], reason='not selected by author Screen/Bin'))
            for bin_name in ['N', 'P']:
                row = use[use.Bin.eq(bin_name)].iloc[0]
                col = f'pool.{int(row.Pool)}'
                saved = comparisons[(comparisons.screen == screen) & (comparisons.bin == bin_name)]
                if len(saved) != 1 or saved.iloc[0].count_column != col or saved.iloc[0].author_library_filename != row['lib.file.name']:
                    raise ValueError('Saved comparison differs from original screen_list')
                name = f'{screen}_{bin_name}'
                v = d[col].to_numpy()
                if not np.isfinite(v).all() or (v < 0).any() or (v != np.floor(v)).any():
                    raise ValueError('Invalid raw guide counts')
                counts[name] = v.astype(np.int64)
                title = f"{screen}_{cfg['geo_title_suffix'][endpoint][bin_name]}"
                record = geo.loc[title]
                if isinstance(record, pd.DataFrame):
                    raise ValueError('Ambiguous GEO sample title')
                issues = ['GEO_library_Brie_vs_author_screen_list_TF']
                if endpoint == 'GATA3_lineage' and bin_name == 'P':
                    issues.append('GEO_positive_title_vs_negative_sorted_phenotype')
                if screen == 'S17' and bin_name == 'N':
                    issues.append('filename_linn_vs_GEO_TOM_negative; exact_gate_not_resolved')
                samples.append(dict(sample=name, endpoint=endpoint, screen=screen, screen_replicate=replicate,
                                    bin=bin_name, geo_accession=record.accession, geo_title=title,
                                    geo_source=record.Sample_source_name_ch1,
                                    geo_characteristics=record.Sample_characteristics_ch1,
                                    count_file=f'GSE262028_{screen}_lib_data.csv.gz', count_column=col,
                                    author_filename=row['lib.file.name'], author_library=row.Library,
                                    assignment_basis='author Screen/Bin + matching GEO title/source; conflicts retained',
                                    metadata_issues=';'.join(issues), interpretation=cfg['interpretation'][endpoint]))
    # A repeated sequence with different author target IDs is not independent gene evidence.
    target_count = guides.groupby('sequence').author_gene_id.nunique()
    guides['cross_target_sequence'] = guides.sequence.map(target_count).gt(1)
    guides['is_non_targeting'] = guides.author_symbol.eq('0')
    if not np.array_equal(guides.is_non_targeting, guides.index.str.startswith('NT')):
        raise ValueError('NT identifiers and author labels disagree')
    guides['Gene'] = guides.author_symbol.where(~guides.is_non_targeting, '__NT__')
    guides['ensembl_candidates'] = guides.author_symbol.map(lambda s: ';'.join(symbol_map.get(s, [])))
    guides['mapping_status'] = guides.author_symbol.map(lambda s: 'unique_exact_symbol' if len(symbol_map.get(s, [])) == 1 else ('ambiguous_symbol' if len(symbol_map.get(s, [])) > 1 else 'unmapped'))
    guides.loc[guides.is_non_targeting, 'mapping_status'] = 'non_targeting'
    guides['ensembl_gene_id'] = guides.ensembl_candidates.where(guides.mapping_status.eq('unique_exact_symbol'), '')
    guides['main_include'] = ~guides.cross_target_sequence if cfg['exclude_cross_target_sequences'] else True
    guides['exclude_reason'] = np.where(guides.main_include, '', 'identical_sequence_assigned_to_multiple_author_gene_ids')
    matrix = pd.DataFrame(counts, index=guides.index)
    table = guides[['Gene']].join(matrix).reset_index()
    write_table(table, out / 'counts_all.tsv')
    write_table(table[guides.main_include.to_numpy()], out / 'counts_main.tsv')
    write_table(guides.reset_index(), out / 'guide_annotation.tsv')
    write_table(guides[~guides.main_include].reset_index(), out / 'excluded_guides.tsv')
    write_table(pd.DataFrame(samples), out / 'samples.tsv')
    write_table(pd.DataFrame(unused), out / 'unused_pools.tsv')
    (out / 'nt_guides.txt').write_text('\n'.join(guides.index[guides.is_non_targeting & guides.main_include]) + '\n')
    eligibility, gene_rows, contrasts = [], [], []
    summary = []
    for endpoint, screens in cfg['endpoints'].items():
        dest = out / endpoint
        dest.mkdir(exist_ok=True)
        write_table(pd.DataFrame(samples).query('endpoint == @endpoint'), dest / 'samples.tsv')
        columns = [f'{s}_{b}' for s in screens for b in ['N', 'P']]
        write_table(table.loc[guides.main_include.to_numpy(), ['sgRNA', 'Gene'] + columns], dest / 'counts_main.tsv')
        for screen in screens:
            columns2 = [f'{screen}_N', f'{screen}_P']
            passed = matrix[columns2].ge(cfg['author_min_count_each_bin']).all(axis=1)
            eligible = guides.main_include & passed
            # Low counts are retained in main counts; this is a separate author-method input.
            write_table(table.loc[eligible.to_numpy(), ['sgRNA','Gene'] + columns2], dest / f'{screen}_counts_author50.tsv')
            (dest / f'{screen}_nt_author50.txt').write_text('\n'.join(guides.index[eligible & guides.is_non_targeting]) + '\n')
            eligibility.append(pd.DataFrame(dict(sgRNA=guides.index, endpoint=endpoint, screen=screen,
                                                count_N=matrix[columns2[0]].to_numpy(), count_P=matrix[columns2[1]].to_numpy(),
                                                pass_author_count=passed.to_numpy(), main_include=guides.main_include.to_numpy(),
                                                eligible_author50=eligible.to_numpy())))
            for (author_id, symbol), group in guides[~guides.is_non_targeting].groupby(['author_gene_id','author_symbol']):
                gene_rows.append(dict(endpoint=endpoint, screen=screen, author_gene_id=author_id, author_symbol=symbol,
                                      ensembl_gene_id=group.ensembl_gene_id.iloc[0], mapping_status=group.mapping_status.iloc[0],
                                      n_guides_original=len(group), n_guides_main=int(group.main_include.sum()),
                                      n_guides_author50=int(eligible.loc[group.index].sum())))
            summary.append(dict(endpoint=endpoint, screen=screen, guides_original=len(guides),
                                guides_main=int(guides.main_include.sum()), guides_author50=int(eligible.sum()),
                                targets_main=guides.loc[guides.main_include & ~guides.is_non_targeting, 'author_gene_id'].nunique(),
                                targets_author50=guides.loc[eligible & ~guides.is_non_targeting, 'author_gene_id'].nunique(),
                                nt_main=int((guides.main_include & guides.is_non_targeting).sum()),
                                nt_author50=int((eligible & guides.is_non_targeting).sum())))
            contrasts.append(dict(endpoint=endpoint, screen=screen, numerator=f'{screen}_P', denominator=f'{screen}_N',
                                  orientation='P_over_N', positive_enrichment='KO guide enriched in P; not equivalent to positive regulator',
                                  author_method='caRpools stat.wilcox; NT reference; per-screen; not run'))
    write_table(pd.concat(eligibility), out / 'guide_screen_eligibility.tsv')
    gene_table = pd.DataFrame(gene_rows)
    write_table(gene_table, out / 'gene_screen_coverage.tsv')
    write_table(pd.DataFrame(contrasts), out / 'contrasts.tsv')
    gene_map = guides.loc[~guides.is_non_targeting, ['author_gene_id','author_symbol','ensembl_gene_id','ensembl_candidates','mapping_status']].drop_duplicates()
    write_table(gene_map, out / 'gene_map.tsv')
    write_table(gene_map[gene_map.mapping_status.ne('unique_exact_symbol')], out / 'unresolved_gene_mapping.tsv')
    write_table(pd.DataFrame(summary), report / 'summary.tsv')
    # Conservation and source round-trip checks protect against altered rows/counts.
    main_read = pd.read_csv(out / 'counts_main.tsv', sep='\t').set_index('sgRNA')
    np.testing.assert_array_equal(main_read[list(matrix)], matrix.loc[main_read.index])
    if len(main_read) + int((~guides.main_include).sum()) != len(guides):
        raise ValueError('Guide records lost')
    if guides.loc[guides.main_include].sequence.duplicated().any():
        raise ValueError('Duplicate sequences remain in main input; review required')
    audit = dict(config=cfg, source_files=manifest,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 supplementary_metadata_sha256={name: hashlib.sha256((raw/name).read_bytes()).hexdigest()
                                                for name in ['author_comparisons.tsv','GSE262028_samples.tsv','GSE262028_family.soft.gz']},
                 gtf_sha256=hashlib.sha256((ROOT/cfg['annotation']).read_bytes()).hexdigest(),
                 counts_unchanged=True, no_guides_silently_lost=True, main_sequences_unique=True,
                 no_normalization=True, no_hit_labels_or_pvalues=True,
                 mapping_status=gene_map.mapping_status.value_counts().to_dict(),
                 excluded_guides=int((~guides.main_include).sum()),
                 known_limitations=['author_1131_vs_table_1133_unresolved','GEO_library_and_GATA3_phenotype_conflicts',
                                    'mapping_uses_project_GRCm38_98_exact_symbols_not_author_release',
                                    'no_sequence_alignment_or_off_target_validation'])
    (out / 'input_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (report / 'validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    readme = '''# GSE262028 筛选分析输入

处理后目录：data/processed/GSE262028_screen_inputs_v1/。两个终点分别为IL13、GATA3_lineage，不混合计算。

## 已核实的分组

| 终点 | Screen | N列 / 作者文件标签 | P列 / 作者文件标签 |
|---|---|---|---|
| IL13 | S17 | pool.4 / linn | pool.6 / tomp |
| IL13 | S25 | pool.5 / Neg | pool.7 / Tp |
| GATA3_lineage | S31 | pool.6 / NK | pool.4 / ILC2 |
| GATA3_lineage | S35 | pool.4 / TBET | pool.1 / GATA3 |

以上列分配直接由各screen_list中明确标注Screen、Bin和TF Library的行确定，并与此前author_comparisons及GEO标题核对。8个GEO记录分别对应8个分选群体；每个终点包含两个screen。其他历史pool列未作为当前比较输入，全部在unused_pools.tsv列出，源CSV保留。

GATA3相关终点包含ILC2/GATA3与NK/TBET相关群体差异，不应称为纯成熟ILC2内部GATA3高低比较。GEO部分GATA3标题为positive但sorted phenotype为negative，GEO文库标Brie但作者screen_list标TF；S17阴性文件名linn与GEO TOM-negative的精确门控关系也未完全解决。samples.tsv保留原文和冲突，采用作者P/N作为操作性分组，不能称所有实验门控细节均已确认。

## 主输入的处理

原4732条UID唯一，排除24条相同序列跨作者target ID的全部50条记录，主表剩4682条，其中200条NT对照。没有合并重复序列计数，也未归一化。所有选中计数均为非负整数，输出与源计数逐值一致。

原1133个目标符号中，Rhox2a/Rhox2e因全部guide存在跨基因序列歧义而失去主分析guide，剩1131个有guide支持的目标。数量与论文一致只是本次过滤结果，不能据此认定作者做过相同处理，原1133与论文1131的原因仍未独立证实。

用户确认主输入不提前删低计数guide。作者方法另要求每个screen的P和N计数都>=50，因此分别提供作者阈值资格和每screen过滤后的独立输入；这不是要求guide在两个screen都过阈值。每个基因是否有足够guide用于正式统计应结合gene_screen_coverage再锁定，当前未自行设定命中标签。

## 文件格式

| 文件 | 格式与用途 |
|---|---|
| counts_all.tsv | 4732行；sgRNA、Gene以及8个样本整数计数列；包含歧义guide |
| counts_main.tsv | 4682行；相同列结构；排除跨target重复序列 |
| IL13/counts_main.tsv | sgRNA、Gene、S17_N、S17_P、S25_N、S25_P |
| GATA3_lineage/counts_main.tsv | sgRNA、Gene、S31_N、S31_P、S35_N、S35_P |
| 各终点/Sxx_counts_author50.tsv | 对应screen两列计数；已排除歧义并通过该screen的作者阈值 |
| samples.tsv / 各终点/samples.tsv | 一个分选群体一行：GSM、终点、screen、P/N、源文件、原pool、原GEO描述和冲突 |
| contrasts.tsv | 每个screen一行，固定P为分子、N为分母及方向解释 |
| guide_annotation.tsv | 原UID、序列、原target ID/symbol、NT标记、重复序列标记、Ensembl候选、去留原因 |
| excluded_guides.tsv | 50条排除记录及理由 |
| guide_screen_eligibility.tsv | 每guide×screen一行，原P/N计数、作者阈值通过标记、最终资格 |
| gene_screen_coverage.tsv | 每gene×screen一行，原guide数、主表guide数、作者阈值后guide数 |
| gene_map.tsv / unresolved_gene_mapping.tsv | 作者基因ID/符号到项目GRCm38.98稳定Ensembl ID的精确符号映射 |
| nt_guides.txt | 主输入200条NT UID；Gene列统一记为__NT__，原gene_symbol=0保存在注释 |
| 各终点/Sxx_nt_author50.txt | 通过该screen阈值的NT UID，供screen内部参照 |

Gene列保留作者基因符号，不把未映射符号猜成其他基因，也不在筛选阶段因为FM词表缺席而删除。Ensembl映射使用项目GRCm38.98精确符号匹配，并非声称这是作者原始注释release；未做guide全基因组比对或off-target验证。后续模型比较需单独核对同义词、词表和token覆盖。

## 作者阈值后的规模

| 终点 | Screen | 主输入guide数 | 作者阈值后guide数 | 其中NT | 至少1条合格guide的目标数 |
|---|---|---:|---:|---:|---:|
'''
    for row in summary:
        readme += f"| {row['endpoint']} | {row['screen']} | {row['guides_main']} | {row['guides_author50']} | {row['nt_author50']} | {row['targets_author50']} |\n"
    readme += '\n基因映射统计：' + json.dumps(audit['mapping_status'], ensure_ascii=False) + '''。
未唯一映射的4个符号为D3Ertd254e、Dux、0610010B08Rik（未映射）和Zkscan7（两个Ensembl候选）。保留它们参与作者符号层面的筛选，不擅自用于模型ID比较。

## 下一步与边界

这一步只准备输入，不计算p值、FDR、富集分数或命中标签，不训练模型，也不做scRNA QC。
作者采用caRpools stat.wilcox，以NT guides作参照，每个screen独立分析；随后平均enrichment score并以Fisher法合并p值。后续如果使用MAGeCK，应标记为替代重分析而非原文精确复现。重复汇总与多重检验方案须在查看模型结果前固定。

固定P/N只是比较方向：P富集表示某KO guide更集中于P群体，不应直接命名为正调控因子；GATA3_lineage还涉及分化/谱系组成。未显著也不代表经实验证实的无功能。

复现命令：`python src/downstream/geneko/prepare_gse262028_screen_inputs.py`。
配置：configs/gse262028_screen_inputs_v1.json。
论文与方法依据：results/GSE262028_GSE242147_feasibility/paper_methods_extract.txt，https://pmc.ncbi.nlm.nih.gov/articles/PMC7616247/。
来源文件、SHA256、配置及验收记录见本报告目录validation.json。
'''
    (report / 'report.md').write_text(readme)
    (out / 'README.md').write_text(readme)
    print(pd.DataFrame(summary).to_string(index=False))
    print('Gene mappings:', audit['mapping_status'], 'excluded guide rows:', audit['excluded_guides'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gse262028_screen_inputs_v1.json')
    args = parser.parse_args()
    main(json.loads(args.config.read_text()))
