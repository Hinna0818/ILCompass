"""以「数据更新」样本表为底，补入「研究汇总」中缺少的研究，并按三类数据集分组输出CSV。"""
import pandas as pd

META = 'data/metadata/'
UPDATE = META + 'ILC2_new_datasets_metadata_2026-10-13.xlsx'
SUMMARY = META + 'ILC2_单细胞数据与metadata汇总_2026-10-12.xlsx'
OUT = META + 'ILC2_datasets_merged_2026-10-15.csv'

C1, C2, C3 = '1_稳态ILC2单细胞微调数据集', '2_小分子/药物扰动ILC2单细胞数据集', '3_基因敲除或扰动ILC2单细胞数据集'
GENETIC = {'E-MTAB-9795', 'GSE195973', 'GSE279338', 'GSE176357'}  # 数据更新中的遗传扰动研究
# 研究汇总中各研究的分类与依据；第2类按外源处理理解（细胞因子/神经肽/药物/抗生素/感染/过敏原等）
ADDED = {
    'GSE117567': (C1, '5组织稳态ILC2，无急性刺激'),
    'GSE153748': (C1, '5组织稳态ILC2；骨髓单独保留'),
    'GSE141330': (C1, '以稳态/发育成熟为主；感染、嵌合、培养子实验需拆出，不进稳态集'),
    'GSE150050': (C1, '人源组织ILC横断面，无直接刺激；仅辅助，不直接混训鼠模型'),
    'GSE102299': (C2, 'IL25/IL33/NMU/NMU+IL25及HDM体内给药，均WT'),
    'GSE136154': (C2, 'CGRP/IL33/IL33+CGRP体内给药'),
    'GSE131996': (C2, 'Nb感染时间过程（含稳态对照）'),
    'GSE124880': (C2, 'WT稳态与OVA过敏背景，需补barcode分组'),
    'GSE242569': (C2, 'IL33激活供者ILC2转移后组织适应；非完整给药对照'),
    'GSE254628': (C2, '抗生素±IL33处理'),
    'GSE268129': (C2, '母体polyIC处理，非ILC2直接刺激'),
    'GSE288743': (C2, 'Nb感染d5，无健康对照'),
    'GSE149857': (C2, '流感感染d5/d10，无健康对照'),
    'GSE150792': (C2, 'Nb感染下WT vs 嗜碱细胞清除；非ILC2内源基因扰动'),
    'GSE189285': (C2, 'PBS vs IL33'),
    'GSE232001': (C2, 'IL12+IL1β+IL18条件化培养及移植'),
    'GSE163367': (C2, '人过敏 vs 健康供者，疾病状态而非给药；仅辅助'),
    'GSE272724': (C3, 'WT vs Pac1/Dusp2 KO'),
}


def classify_update(r):
    if r['用途'].startswith('post-training'):
        return C1, '原表用途为稳态post-training'
    if r['研究'] in GENETIC or r['样本访问号'] == 'GSM8436017':
        return C3, f"基因型扰动：{r['基因型']}"
    return C2, f"外源处理：{r['处理/条件']}"


upd = pd.read_excel(UPDATE)
upd[['分类', '分类依据']] = upd.apply(classify_update, axis=1, result_type='expand')
upd['数据描述'] = ''
upd['记录来源'] = '数据更新（样本级）'

summ = pd.read_excel(SUMMARY, sheet_name='研究汇总')
samples = pd.read_excel(SUMMARY, sheet_name='样本metadata')
gse153748 = pd.read_csv(META + 'GSE153748_sample_metadata.tsv', sep='\t')
missing = summ[~summ['GEO'].isin(upd['研究'])]
assert set(missing['GEO']) == set(ADDED), set(missing['GEO']) ^ set(ADDED)


def text(v):
    return 'not_available' if pd.isna(v) else str(v).strip()


def gsm_range(geo):
    if geo == 'GSE153748':  # 研究汇总未列GSM，用本项目已核对的样本表
        g = gse153748['sample_id']
    else:
        g = samples.loc[samples['GEO'] == geo, 'GSM']
    if g.empty:
        return 'not_available（研究汇总未列GSM）'
    return f'{g.iloc[0]}…{g.iloc[-1]}({len(g)}个GSM)'


rows = []
for _, s in missing.iterrows():
    cat, reason = ADDED[s['GEO']]
    if pd.notna(s['论文名称']) and pd.notna(s['论文年份']):
        s['论文名称'] = f"{s['论文名称']}（{int(s['论文年份'])}）"
    keys = ['可用状态', '细胞/样本注释情况', '推荐模型用途', '表达矩阵入口已确认', '本次下载情况',
            'scBaseCount覆盖', 'scBase观测数（非QC后ILC2）', '论文名称', '论文链接', 'GEO链接']
    desc = '；'.join(f'{k}：{text(s[k])}' for k in keys if pd.notna(s[k]))
    human = s['物种'] == 'Human'
    rows.append({
        '用途': '辅助(人源)' if human else 'post-training(稳态)' if cat == C1 else '下游(扰动/KO/刺激)',
        '研究': s['GEO'], '样本访问号': gsm_range(s['GEO']), '样本名': '研究级汇总',
        '物种': text(s['物种']).lower(), '组织': text(s['组织']), '细胞群': text(s['细胞来源']),
        '基因型': '; '.join(samples.loc[samples['GEO'] == s['GEO'], '基因型'].dropna().unique()) or 'not_available',
        '处理/条件': text(s['实验条件']), '时间点': text(s['处理/取材时间']), '平台': text(s['测序类型']),
        '重复/合池': text(s['生物学重复及合池']), '数据文件': text(s['处理矩阵格式']), '备注': text(s['主要限制']),
        '分类': cat, '分类依据': reason, '数据描述': desc, '记录来源': '研究汇总补充（研究级）',
    })

out = pd.concat([upd, pd.DataFrame(rows)], ignore_index=True)
out['单细胞RNA'] = out['平台'].str.contains('bulk|scATAC', case=False).map({True: '否', False: '是'})
out = out[['分类'] + [c for c in out.columns if c != '分类']].sort_values('分类', kind='stable')
out.to_csv(OUT, index=False, encoding='utf-8-sig')
print(out.groupby(['分类', '记录来源']).agg(行数=('研究', 'size'), 研究数=('研究', 'nunique')))
print('输出', OUT, out.shape)
