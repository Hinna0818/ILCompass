"""统一宽松规则的关键边界：dropout、状态、混合群与双细胞分数不独立触发删除。"""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from ilc2_qc import PANELS, ACINAR, add_evidence, apply_standard
from qc_gse153748 import author_programs

ROOT = Path(__file__).resolve().parents[1]
policy = json.loads((ROOT / 'configs/1QC.json').read_text())['identity']
genes = list(dict.fromkeys([g for panel in PANELS.values() for g in panel] + ACINAR + ['Cd74']))
programs = [
    {'Gata3': 2, 'Icos': 1, 'Il17a': 8, 'Cd74': 10},
    {'Il7r': 2, 'Id2': 1, 'Rorc': 1},
    {'Gata3': 2},
    {'Icos': 2, 'Klrg1': 2},
    {'Gata3': 2, 'Icos': 1, 'Rorc': 3, 'Il23r': 3, 'Il22': 10},
    {'Gata3': 2, 'Icos': 1, 'Cd3d': 3, 'Cd3e': 3},
    {'Gata3': 2, 'Icos': 1, **{g: 50 for g in ACINAR[:3]}},
    {'Gata3': 2, 'Icos': 1, 'Cd79a': 3, 'Cd79b': 3},
    {'Gata3': 2, 'Icos': 1, 'Alas2': 5, 'Bpgm': 5, 'Gypa': 5},
    {'Gata3': 2, 'Icos': 1, 'Cd3d': 1, 'Cd3e': 1},
    {'Gata3': 2, 'Icos': 1, **{g: 1 for g in ACINAR[:3]}},
]
x = np.zeros((len(programs), len(genes)), dtype=np.int32)
for i, values in enumerate(programs):
    for gene, value in values.items():
        x[i, genes.index(gene)] = value
a = ad.AnnData(sparse.csr_matrix(x), var=pd.DataFrame({'gene_symbol': genes}, index=genes))
a.layers['counts'] = a.X.copy()
a.obs['total_counts'] = 1000
a.obs['n_genes_by_counts'] = 300
a.obs['library_cluster'] = 'mixed_group'
a.obs['doublet_score'] = .4
a.obs['state_stress_score'] = 100
add_evidence(a)
add_evidence(a)
assert a.obs.columns.is_unique
apply_standard(a, policy)
assert a.obs.selected_ILC2.tolist() == [True, True, False, False, False, False, False, False, False, True, True]
assert a.obs.mixed_cluster_review.all() and a.obs.doublet_review.all()
a.obs.loc[a.obs_names[0], 'total_counts'] = 499
apply_standard(a, policy)
assert not a.obs.selected_ILC2.iloc[0]
a.obs.loc[a.obs_names[0], 'total_counts'] = 500
apply_standard(a, policy)
assert a.obs.selected_ILC2.iloc[0]
assert set(author_programs()['effector']) == {'Il5', 'Il13', 'Tnfrsf9', 'Cxcr6', 'Dgat1'}
print('PASS: 低RNA/Gata3漏检/状态与混合群保留；单marker不足、强替代程序/腺泡/红系排除；500UMI边界及重复证据计算。')
