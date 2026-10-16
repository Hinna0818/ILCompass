"""在GSE117567既有基础QC细胞上对照新身份规则，不覆盖历史训练对象。

输入：results/GSE117567_ilc2_selection/*_annotated.h5ad 与旧去留表。
输出：results/ILC2_qc_standard/GSE117567_identity_comparison.tsv。
运行：python main/compare_ilc2_qc_standard.py
这不是原始输入/候选门槛重跑，不冒充完整新版本。
"""
from pathlib import Path
import json
import sys
import anndata as ad
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ilc2_qc import add_evidence, apply_standard


def main():
    policy = json.loads((ROOT / 'configs/1QC.json').read_text())['identity']
    previous = pd.read_csv(ROOT / 'results/GSE117567_ilc2_selection/cell_selection_decisions.tsv.gz', sep='\t', index_col=0)
    rows = []
    for path in sorted((ROOT / 'results/GSE117567_ilc2_selection').glob('GSM*_annotated.h5ad')):
        a = ad.read_h5ad(path)
        a.obs['library_cluster'] = a.obs.selection_cluster
        missing = add_evidence(a)
        apply_standard(a, policy)
        old = previous.selected_for_training.reindex(a.obs_names)
        assert not old.isna().any()
        rows.append({'sample_id':str(a.obs['sample'].iloc[0]), 'tissue':str(a.obs.tissue.iloc[0]),
                     'existing_basic_qc':a.n_obs, 'previous_strict_training':int(old.sum()),
                     'balanced_identity_candidates':int(a.obs.selected_ILC2.sum()),
                     'balanced_peripheral_candidates':int(a.obs.selected_ILC2.sum()) if a.obs.tissue.iloc[0]!='BM' else 0,
                     'retained_doublet_review':int((a.obs.selected_ILC2 & a.obs.doublet_review).sum()),
                     'newly_supported':int((a.obs.selected_ILC2 & ~old).sum()),
                     'previous_selected_now_excluded':int((~a.obs.selected_ILC2 & old).sum()),
                     'missing_markers':';'.join(missing), 'scope':'identity_only_on_existing_basic_QC'})
    out = ROOT / 'results/ILC2_qc_standard'
    out.mkdir(exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'GSE117567_identity_comparison.tsv',sep='\t',index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
