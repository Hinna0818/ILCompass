"""把官方人类基因中位数经官方同源表映射为小鼠 ENSMUSG 中位数。

输入：external/GeneCompass 的 human_gene_median_after_filter.pickle、human_mouse_tokens.pickle、
homologous_hm_token.pickle（小鼠token->人token，官方 checkpoint 的 homologous_index 由它反向得到）。
另对官方表无同源的小鼠基因做保守 symbol 补充（见下 symbol_match），source 列区分 official/symbol_exact。
输出：data/references/mouse_gene_median.pickle（仅含有人类中位数的小鼠基因；缺失基因交给
官方 Normalized 的默认值 1）；results/GeneCompass_mouse_median/ 的逐基因表与 report.json。
这是人类中位数的同源代理，不是小鼠数据估计值。运行：python main/build_mouse_gene_median.py
"""
import collections
import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
gc = root / "external/GeneCompass"
out = root / "results/GeneCompass_mouse_median"
paths = {
    "median": gc / "scdata/dict/human_gene_median_after_filter.pickle",
    "tokens": gc / "scdata/dict/human_mouse_tokens.pickle",
    "homolog": gc / "prior_knowledge/homologous_hm_token.pickle",
    "human_names": gc / "scdata/Gene_id_name_dict.pickle",  # ENSG -> symbol
    "gtf": root / "data/references/Mus_musculus.GRCm38.84.gtf.gz",
}
load = lambda p: pickle.load(open(p, "rb"))
human_median, tokens, mouse2human = (load(paths[k]) for k in ("median", "tokens", "homolog"))
id_of = {v: k for k, v in tokens.items()}

rows = []
for gid, tok in tokens.items():
    if not gid.startswith("ENSMUSG"):
        continue
    htok = mouse2human.get(tok)
    hid = id_of[htok] if htok is not None else None
    med = human_median.get(hid) if hid else None
    status = "no_homolog" if hid is None else "homolog_no_median" if med is None else "mapped"
    rows.append((gid, tok, hid, htok, med, status, "official" if status == "mapped" else None))
df = pd.DataFrame(rows, columns=["mouse_id", "mouse_token", "human_id", "human_token", "median", "status", "source"])

# symbol_match：大小写不敏感的完全相同 symbol。要求 (1) 小鼠 symbol 在 GTF 中唯一；(2) 人类 symbol 在
# 全部 ENSG 中唯一，且该 ENSG 在词表并有中位数。该人类基因即使已被官方表分给某个 Gm/pseudogene
# （如 RPLP1->Gm5244），同名真基因仍共享其中位数；多对一在官方表中本就存在。
# 只补 no_homolog 基因；官方已有同源（含缺中位数的）不改。
mouse_sym = {}
for line in gzip.open(paths["gtf"], "rt"):
    f = line.split("\t")
    if line[0] != "#" and f[2] == "gene":
        a = dict(re.findall(r'(\w+) "([^"]*)"', f[8]))
        mouse_sym[a["gene_id"]] = a["gene_name"]
msym_n = collections.Counter(v.upper() for v in mouse_sym.values())
hsym = collections.defaultdict(list)
for k, v in load(paths["human_names"]).items():
    hsym[v.upper()].append(k)
def symbol_match(mid):
    sym = mouse_sym.get(mid, "").upper()
    cand = hsym.get(sym, [])
    if not sym or msym_n[sym] != 1 or len(cand) != 1:
        return None
    h = cand[0]
    return h if h in tokens and h in human_median else None
# 规则自检：在官方已映射的基因上，symbol 规则与官方同源一致的比例
pairs = df[df.status == "mapped"]
chk = [(r.human_id, sym_h) for r in pairs.itertuples()
       if len(c := hsym.get(mouse_sym.get(r.mouse_id, "").upper(), [])) == 1 and msym_n[mouse_sym[r.mouse_id].upper()] == 1
       for sym_h in c]
agree = float(np.mean([a == b for a, b in chk]))
for i in df.index[df.status == "no_homolog"]:
    h = symbol_match(df.at[i, "mouse_id"])
    if h:
        df.loc[i, ["human_id", "human_token", "median", "status", "source"]] = [h, tokens[h], human_median[h], "mapped", "symbol_exact"]
df["mouse_symbol"] = df.mouse_id.map(mouse_sym)
df["n_mouse_sharing_human"] = df.groupby("human_id")["mouse_id"].transform("size").where(df.human_id.notna())
df.to_csv(out / "mouse_gene_median.tsv", sep="\t", index=False)

mapped = df[df.status == "mapped"]
official_n = int((mapped.source == "official").sum())
mouse_median = dict(zip(mapped.mouse_id, mapped["median"].astype(float)))
dst = root / "data/references/mouse_gene_median.pickle"
pickle.dump(mouse_median, open(dst, "wb"))

# 覆盖：训练候选对象中的基因/核心基因，以及 ILC2 关键基因
adata = ad.read_h5ad(root / "data/train_data/GSE117567.h5ad")
var = adata.var
umi = np.asarray(adata.X.sum(0)).ravel()
in_model = var.index.isin(df.mouse_id)
has_med = var.index.isin(mapped.mouse_id)
core = var.is_core.astype(bool).values
key = ["Gata3", "Rora", "Il7r", "Thy1", "Id2", "Il1rl1", "Il17rb", "Icos", "Klrg1", "Il5", "Il13", "Areg", "Arg1", "Rorc", "Cd3e", "Ncr1"]
kv = var[var.gene_symbol.isin(key)].assign(has_median=lambda d: d.index.isin(mapped.mouse_id))
report = {
    "sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()} | {"output": hashlib.sha256(dst.read_bytes()).hexdigest()},
    "mouse_tokens": len(df),
    "official_mapped": official_n,
    "symbol_exact_added": len(mapped) - official_n,
    "symbol_rule_agreement_on_official_pairs": [len(chk), agree],
    "status_counts": df.status.value_counts().to_dict(),
    "human_genes_shared_by_multiple_mouse": int((mapped.groupby("human_id").size() > 1).sum()),
    "median_range_mapped": [float(mapped["median"].min()), float(mapped["median"].max())],
    "GSE117567_genes": len(var),
    "GSE117567_in_vocab": int(in_model.sum()),
    "GSE117567_with_median": int(has_med.sum()),
    "GSE117567_core": int(core.sum()),
    "GSE117567_umi_fraction_in_vocab": float(umi[in_model].sum() / umi.sum()),
    "GSE117567_umi_fraction_with_median": float(umi[has_med].sum() / umi.sum()),
    "GSE117567_core_with_median": int((core & has_med).sum()),
    "key_genes_without_median": kv[~kv.has_median].gene_symbol.tolist(),
    "key_genes_missing_from_object": sorted(set(key) - set(kv.gene_symbol)),
}
json.dump(report, open(out / "report.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
