"""只读整理作者subcluster差异基因Excel；不修改AnnData或自动筛选。

运行：/opt/anaconda3/bin/python main/audit_gse153748_signatures.py
使用本机已有openpyxl，ilc2_model环境未安装此包；不为只读审计改动分析环境。
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/GSE153748_RAW/GSE153748_Tissue_ILC2_sub-cluster_signature_genes.xlsx"
OUT = ROOT / "results/GSE153748_qc/signature_reference"
TISSUES = {"BM ILC2": "BM", "LI ILC2": "LI", "Lung ILC2": "Lung", "Pancreas ILC2": "Pancreas", "SI ILC2": "SI"}


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    book = load_workbook(SOURCE, read_only=True, data_only=True)
    assert set(book.sheetnames) == set(TISSUES), book.sheetnames
    long, summaries, sheets = [], [], []
    for sheet in book:
        rows = list(sheet.iter_rows(values_only=True))
        lists = {}
        for column, header in enumerate(rows[3]):
            if header is None: continue
            match = re.fullmatch(r"(\d+)\s+VS\s+(\d+|all|others)", str(header).strip(), re.I)
            assert match, (sheet.title, header)
            cluster, comparison = match.groups()
            comparison = "all" if comparison.lower() in {"all", "others"} else comparison
            genes = [str(row[column]).strip() for row in rows[4:] if row[column] is not None and str(row[column]).strip().upper() not in {"NA", "N/A", "NAN", ""}]
            assert len(genes) == len(set(genes)), (sheet.title,header)
            # 原表含一个酶全名，保留原文并标记，不能静默当gene symbol映射。
            assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", gene) or gene == "Bile salt-activated lipase" for gene in genes)
            lists[(cluster, comparison)] = genes
            for rank, gene in enumerate(genes,1):
                long.append(dict(tissue=TISSUES[sheet.title], author_cluster=cluster,
                                 comparison=comparison, gene=gene, row_order=rank,
                                 sheet=sheet.title, excel_column=column+1))
        clusters = sorted({c for c,v in lists}, key=int)
        sheets.append(dict(sheet=sheet.title, tissue=TISSUES[sheet.title], n_author_clusters=len(clusters), n_comparison_columns=len(lists)))
        for cluster in clusters:
            pairwise = [set(genes) for (c,v),genes in lists.items() if c == cluster and v != "all"]
            intersection = set.intersection(*pairwise) if pairwise else set()
            counts = Counter(g for group in pairwise for g in group)
            one_all = lists[(cluster,"all")]
            summaries.append(dict(tissue=TISSUES[sheet.title], author_cluster=cluster,
                                  n_vs_all=len(one_all), vs_all_genes=";".join(one_all),
                                  n_union_pairwise=len(counts), n_intersection_pairwise=len(intersection),
                                  intersection_pairwise_genes=";".join(sorted(intersection)),
                                  recurring_pairwise_top=";".join(g for g,n in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:12])))
    book.close()
    write_table(OUT / "author_signature_long.tsv",long)
    write_table(OUT / "author_signature_summary.tsv",summaries)
    write_table(OUT / "workbook_sheets.tsv",sheets)
    assert len(summaries) == sum(s["n_author_clusters"] for s in sheets)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
    (OUT / "provenance.json").write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)),sha256=digest,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_unchanged=True,missing_gene_sentinel="NA",vs_others_normalized_to="all",
            no_cell_metadata_or_DE_statistics=True,n_signature_gene_entries=len(long),
            unresolved_gene_labels=["Bile salt-activated lipase"]),indent=2)+'\n')
    for row in summaries:
        genes=row["vs_all_genes"].split(";")
        print(row["tissue"], row["author_cluster"],row["n_vs_all"], 'top:',','.join(genes[:10]), 'intersection:',row["intersection_pairwise_genes"])


if __name__ == "__main__":
    main()
