"""基础 QC 的边界/数据完整性检查；不依赖真实矩阵或额外测试框架。"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from qc_utils import qc_sample, read_gene_annotation, select_reviewed_ilc2


def fixture(n_cells: int = 20) -> tuple:
    ids = [f"ENSMUSG{i:011d}" for i in range(205)]
    types = ["protein_coding"] * 9 + ["miRNA", "rRNA", "processed_pseudogene"] + ["antisense"] * 193
    annotation = pd.DataFrame({"gene_symbol": ids, "gene_biotype": types, "biotype_group": ["lncRNA" if t == "antisense" else t for t in types], "is_mt": [i == 8 for i in range(205)]}, index=ids)
    a = ad.AnnData(sparse.csr_matrix(np.ones((n_cells, 205), dtype=np.int32)), var=pd.DataFrame(index=ids))
    a.obs["sample"] = "synthetic"
    return a, annotation


def expect_error(action, text: str) -> None:
    try:
        action()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError("预期的输入校验没有触发：" + text)


def main() -> None:
    a, genes = fixture()
    original = a.X.copy()
    result = qc_sample(a, genes)
    assert result["full"].n_obs == 20 and result["core"].n_vars == 203
    assert (a.X != original).nnz == 0 and "qc_pass" not in a.obs
    assert not result["decisions"].qc_fail_high_genes.any()  # SD=0 不删细胞。
    assert np.array_equal(np.asarray(result["full"].X.sum(axis=1)).ravel(), result["full"].obs.total_counts)
    assert np.array_equal(result["full"].obs_names, result["core"].obs_names)
    reviewed = result["full"].copy()
    reviewed.obs["cell_type"] = "ILC2"
    reviewed.obs["identity_confidence"] = "high"
    reviewed.obs.loc[reviewed.obs_names[0], "cell_type"] = "NK"
    selected, identity_decisions = select_reviewed_ilc2(reviewed)
    assert selected.n_obs == 19 and len(identity_decisions) == 20
    expect_error(lambda: select_reviewed_ilc2(a), "已复核")

    # 15% 包含边界，>15% 才删除；基因过滤后不重算分母。
    x = a.X.toarray()
    x[0, 8] = 36  # 36/(204+36)=15%
    x[1, 8] = 37
    x[2, :4] = 0  # 201 检出基因仍达到 200，但 protein/miRNA 只剩 6。
    x[3] = 0
    a.X = sparse.csr_matrix(x)
    r = qc_sample(a, genes)
    assert r["decisions"].iloc[0].qc_pass
    assert r["decisions"].iloc[1].qc_fail_high_mt
    assert r["decisions"].iloc[2].qc_fail_low_coding_mirna
    assert r["decisions"].iloc[3].qc_fail_zero_counts
    assert np.isfinite(r["decisions"].pct_counts_mt).all()

    b, _ = fixture()
    x = b.X.toarray(); x[:-1, -4:] = 0
    b.X = sparse.csr_matrix(x)
    r = qc_sample(b, genes)
    assert r["decisions"].iloc[-1].qc_fail_high_genes
    assert r["full"].n_obs == 19

    c, _ = fixture(3)
    assert qc_sample(c, genes)["full"].n_obs == 0
    c, _ = fixture(4)
    x = c.X.toarray(); x[0] = 0; c.X = sparse.csr_matrix(x)
    r = qc_sample(c, genes)
    assert r["full"].n_obs == 0 and r["summary"]["n_fail_small_sample_after"] == 3

    c, _ = fixture()
    # 显式存储零不能被 getnnz 误计为表达；工具会 eliminate_zeros。
    c.X.data[0] = 0
    r = qc_sample(c, genes)
    assert r["decisions"].iloc[0].n_genes_by_counts == 204
    c.layers["counts"] = c.X.copy()
    c.X = c.X.astype(float); c.X.data[:] = np.log1p(c.X.data)
    assert qc_sample(c, genes)["full"].n_obs == 20  # counts 层优先。
    del c.layers["counts"]
    expect_error(lambda: qc_sample(c, genes), "非整数")
    c, _ = fixture()
    c.obs.loc[c.obs_names[0], "sample"] = "second_sample"
    expect_error(lambda: qc_sample(c, genes), "单样本")
    c, _ = fixture()
    expect_error(lambda: qc_sample(c, genes.iloc[1:]), "缺少匹配注释")
    c.X.data[0] = -1
    expect_error(lambda: qc_sample(c, genes), "非负")

    # 真实 GTF 属性解析及旧 lncRNA 类别映射。
    with tempfile.TemporaryDirectory() as directory:
        # 真实写盘边界：输入索引名可能来自 pandas 的整数列名 0。
        a, genes = fixture()
        a.obs.index.name = 0
        result = qc_sample(a, genes)
        target = Path(directory) / "qc.h5ad"
        result["full"].write_h5ad(target)
        reloaded = ad.read_h5ad(target)
        assert reloaded.shape == result["full"].shape and reloaded.obs.index.name == "cell_id"
        gtf = Path(directory) / "test.gtf"
        gtf.write_text('1\tEnsembl\tgene\t1\t2\t.\t+\t.\tgene_id "ENSMUSG00000000001"; gene_name "Test"; gene_biotype "antisense";\nMT\tEnsembl\tgene\t1\t2\t.\t+\t.\tgene_id "ENSMUSG00000000002"; gene_name "mt-Test"; gene_biotype "protein_coding";\n')
        parsed = read_gene_annotation(gtf)
        assert parsed.loc["ENSMUSG00000000001", "biotype_group"] == "lncRNA"
        assert parsed.loc["ENSMUSG00000000002", "is_mt"]
    print("PASS: QC 边界、SD=0/高基因数、少于4细胞、计数完整性、层选择、输入拒绝及GTF解析。")


if __name__ == "__main__":
    main()
