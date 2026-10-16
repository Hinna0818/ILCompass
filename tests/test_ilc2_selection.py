"""关键身份边界检查：保留 ST2 阴性 ILC2，不将单个 Rorc/Ccr6/Il17a 当污染。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from build_train_gse117567 import add_evidence, PANELS


def main() -> None:
    genes = list(dict.fromkeys(g for panel in PANELS.values() for g in panel))
    x = np.zeros((5, len(genes)), dtype=np.int32)
    def put(row: int, values: dict) -> None:
        for gene, value in values.items():
            x[row, genes.index(gene)] = value
    put(0, {"Gata3": 4, "Rora": 2, "Il7r": 2, "Icos": 3, "Ccr6": 5, "Il17a": 3})
    put(1, {"Gata3": 4, "Rora": 2, "Il7r": 2, "Icos": 3, "Rorc": 1})
    put(2, {"Rorc": 3, "Il23r": 3, "Il22": 10})
    put(3, {"Ncr1": 3, "Eomes": 3, "Prf1": 4, "Nkg7": 10})
    put(4, {"Gata3": 3, "Il7r": 2, "Thy1": 2, "Icos": 2, "Cd3d": 3, "Cd3e": 3})
    a = ad.AnnData(sparse.csr_matrix(x), var=pd.DataFrame({"gene_symbol": genes}, index=genes))
    a.layers["counts"] = a.X.copy()
    # 本测试仅测试身份程序，QC 指标视作已从另一阶段提供。
    a.obs["n_genes_by_counts"] = 600
    a.obs["total_counts"] = 2000
    add_evidence(a)
    assert a.obs.iloc[0].identity_positive and not a.obs.iloc[0].ilc3_program_review
    assert a.obs.iloc[1].identity_positive and not a.obs.iloc[1].ilc3_program_review
    assert a.obs.iloc[2].ilc3_program_review and not a.obs.iloc[2].identity_positive
    assert a.obs.iloc[3].lineage_NK_ILC1 and not a.obs.iloc[3].identity_positive
    assert a.obs.iloc[4].identity_positive and a.obs.iloc[4].lineage_T
    print("PASS: ST2阴性ILC2、单Rorc、皮肤Ccr6/Il17a、ILC3/NK与T混合程序边界。")


if __name__ == "__main__":
    main()
