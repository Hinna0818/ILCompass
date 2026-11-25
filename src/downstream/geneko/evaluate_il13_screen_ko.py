"""Frozen raw/LoRA Il13 gene-embedding response to the locked 229-target KO design.

Default: full evaluation with per-cell resume. --smoke: 3 genes x 2 cells,
separate engineering results. No training or resampling of controls.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "compare_pretrain/model_compare"))
import shared
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

DESIGN = ROOT / "compare_pretrain/model_compare/crispr_il13/design_v1"
OUT = ROOT / "results/GSE262028_IL13_insilico_v1"
SMOKE_GENES = ["Junb", "Bcl11b", "Tcf12"]


class Inputs:
    def __init__(self, smoke):
        self.design = json.loads((DESIGN / "design.json").read_text())
        cfg = self.design["config"]
        for name, digest in self.design["source_sha256"].items():
            if shared.file_hash(ROOT / name) != digest:
                raise ValueError(f"Frozen source changed: {name}")
        roster_path = DESIGN / "target_cell_controls.tsv.gz"
        if shared.file_hash(roster_path) != self.design["manifest_sha256"]:
            raise ValueError("Fixed target-cell/random-control manifest changed")
        self.targets = pd.read_csv(DESIGN / "targets.csv")
        coverage = pd.read_csv(ROOT / cfg["coverage_directory"] / "TF_primary_candidates.csv")
        pd.testing.assert_frame_equal(self.targets, coverage.sort_values("ensembl_gene_id").reset_index(drop=True))
        self.pairs = pd.read_csv(roster_path, sep="\t")
        if smoke:
            pieces = []
            for gene in SMOKE_GENES:
                # Exercise long inputs and targets at different ranks, without claiming biology.
                d = self.pairs[self.pairs.target_symbol.eq(gene)]
                pieces.append(d.sort_values(["input_length", "cell_row"], ascending=[False, True]).head(2))
            self.pairs = pd.concat(pieces).sort_values("pair_id").copy()
            if len(self.pairs) != 2 * len(SMOKE_GENES):
                raise ValueError("Smoke genes/cells unavailable")
        self.groups = dict(tuple(self.pairs.groupby("cell_row", sort=True)))
        ref = ROOT / cfg["reference_directory"]
        a = ad.read_h5ad(ref / "GSE126924_Lung_IL13positive_ILC2_1864.h5ad")
        self.cells = pd.read_csv(DESIGN / "cells.csv.gz")
        if not np.array_equal(a.obs_names, self.cells.cell_id):
            raise ValueError("Reference row alignment changed")
        lora = json.loads((ROOT / "compare_pretrain/genepass_lora/config.json").read_text())
        self.tokens = pickle.loads((ROOT / lora["token_dictionary"]).read_bytes())
        median = pickle.loads((ROOT / lora["median_reference"]).read_bytes())
        coverage_summary = json.loads((ROOT / cfg["coverage_directory"] / "summary.json").read_text())
        if shared.file_hash(ROOT / lora["median_reference"]) != coverage_summary["sha256"][lora["median_reference"]]:
            raise ValueError("Median reference changed")
        keep = a.var.is_core.to_numpy(bool) & a.var_names.isin(self.tokens)
        self.genes = a.var_names[keep].to_numpy()
        self.counts = a.layers["counts"][:, keep].tocsr()
        if (a.X != a.layers["counts"]).nnz or (self.counts.data < 0).any() or not np.equal(self.counts.data, np.floor(self.counts.data)).all():
            raise ValueError("Expected original integer counts")
        self.scale = np.array([median.get(g, 1.) for g in self.genes])
        if not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid median reference")
        self.column_by_token = {self.tokens[g]: j for j, g in enumerate(self.genes)}
        readout_genes = a.var_names[a.var.gene_symbol.eq(cfg["readout_symbol"])]
        if len(readout_genes) != 1:
            raise ValueError("Readout not unique")
        self.readout = self.tokens[readout_genes[0]]
        self.max_length = cfg["max_length"]
        self.tokenize = shared.official_tokenizer()
        saved = np.load(ref / "genecompass_inputs.npz")
        self.ids, self.values, self.lengths = [saved[k] for k in ["input_ids", "values", "lengths"]]
        self.random_columns = [f"random{j}_token_id" for j in range(cfg["random_deletions_per_target_cell"])]
        if len(self.random_columns) != 3 or cfg["target_expression_fraction"] != 0:
            raise ValueError("This runner implements the locked zero-expression + 3-control design")

    def sequence(self, normalized):
        ids, values = self.tokenize(np.log2(1 + normalized), self.genes, self.tokens)
        return np.asarray(ids[:self.max_length], dtype=np.int64), np.asarray(values[:self.max_length], dtype=np.float32)

    def cell_sequences(self, row, pairs):
        base = self.counts[row].toarray().ravel() / self.scale
        before = self.sequence(base)
        n = int(self.lengths[row])
        if not np.array_equal(before[0], self.ids[row, :n]) or not np.array_equal(before[1], self.values[row, :n]):
            raise ValueError("Regenerated baseline differs from frozen model inputs")
        seq = [before]
        perturbations = np.unique(pairs[["target_token_id", *self.random_columns]].to_numpy(dtype=np.int64))
        if self.readout in perturbations:
            raise ValueError("Il13 must not be directly perturbed")
        positions, refill = [], []
        for token in perturbations:
            col = self.column_by_token[int(token)]
            original = base[col]
            if original <= 0 or token not in before[0]:
                raise ValueError("Perturbed target absent from the baseline input")
            base[col] = 0.
            changed = self.sequence(base)
            base[col] = original
            if token in changed[0]:
                raise ValueError("Zero-expression KO retained its target")
            pos = np.flatnonzero(changed[0] == self.readout)
            positions.append(int(pos[0]) if len(pos) else -1)
            refill.append(len(set(changed[0]).difference(before[0])))
            seq.append(changed)
        return perturbations, seq, np.asarray(positions, dtype=np.int32), np.asarray(refill, dtype=np.int16)


@torch.inference_mode()
def readout_embeddings(model, seq, token, model_name):
    positions = [int(np.flatnonzero(ids == token)[0]) for ids, _ in seq]
    h = shared.encode(model, shared.batch(seq), model_name)
    if h.shape[1] != max(len(ids) for ids, _ in seq) + 1:
        raise ValueError("Expected leading CLS followed by gene embeddings")
    # The gene can move after re-ranking. CLS contributes exactly one output offset.
    z = h[torch.arange(len(seq), device=h.device), torch.tensor(positions, device=h.device) + 1].float().cpu().numpy()
    if not np.isfinite(z).all() or (np.linalg.norm(z, axis=1) == 0).any():
        raise ValueError("Invalid Il13 embeddings")
    return z


def cosine_distance(before, after):
    x, y = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    return 1 - np.clip(np.sum(x*y, axis=-1) / (np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1)), -1, 1)


def run_model(inputs, name, folder, batch_size):
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "cells"
    cache.mkdir(exist_ok=True)
    model_files = [ROOT / shared.MODELS["raw"] / "pytorch_model.bin", ROOT / shared.MODELS["raw"] / "config.json"]
    if name == "lora":
        model_files += [ROOT / shared.MODELS[name], ROOT / "compare_pretrain/genepass_lora/config.json",
                        ROOT / "compare_pretrain/genepass_lora/model.py"]
    signature = dict(model=name, precision="float32; TF32 disabled; float64 cosine",
        batch_size=batch_size, manifest_sha256=inputs.design["manifest_sha256"],
        pair_ids=inputs.pairs.pair_id.tolist(), source_sha256=inputs.design["source_sha256"],
        hashes={str(p.relative_to(ROOT)): shared.file_hash(p) for p in
                [*model_files, Path(__file__), ROOT / "compare_pretrain/model_compare/shared.py", ROOT / "src/genecompass_compat.py"]},
        torch_version=torch.__version__, cuda_version=torch.version.cuda)
    signature_path = folder / "run_signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("Run signature changed; do not mix cached results")
    else:
        shared.write_json(signature_path, signature)
    pending = [r for r in inputs.groups if not (cache / f"{r:06d}.npz").exists()]
    if not pending:
        print(f"{name}: all {len(inputs.groups)} cell caches complete", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; use --check-only for CPU input validation")
    model = shared.load(name)
    model.float().eval().requires_grad_(False)
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    start = time.monotonic()
    print(f"{name}: pending {len(pending)}/{len(inputs.groups)} cells", flush=True)
    for k, row in enumerate(pending):
        perturbed, seq, positions, refill = inputs.cell_sequences(row, inputs.groups[row])
        baseline = readout_embeddings(model, seq[:1], inputs.readout, name)[0]
        if k == 0:
            # Real model repeat/sham check at the native input length, not a synthetic mini-network.
            sham = readout_embeddings(model, [seq[0], seq[0]], inputs.readout, name)
            error = float(np.abs(sham - baseline).max())
            sham_distance = float(cosine_distance(baseline, sham).max())
            if error > 1e-4 or sham_distance > 1e-8:
                raise ValueError(f"Repeat/sham failed: max_abs={error}, cosine_distance={sham_distance}")
            shared.write_json(folder / "engineering_check.json", dict(cell_row=int(row),
                repeat_sham_max_abs=error, repeat_sham_max_cosine_distance=sham_distance,
                cls_offset=1, alignment="Il13 token ID separately located in every sequence"))
        distance = np.full(len(perturbed), np.nan, dtype=np.float64)
        valid = np.flatnonzero(positions >= 0)
        for left in range(0, len(valid), batch_size):
            idx = valid[left:left+batch_size]
            z = readout_embeddings(model, [seq[j+1] for j in idx], inputs.readout, name)
            distance[idx] = cosine_distance(baseline, z)
        final = cache / f"{row:06d}.npz"
        temp = cache / f"{row:06d}.tmp.npz"
        np.savez_compressed(temp, cell_row=row, perturbed_token=perturbed,
            Il13_distance=distance, Il13_rank_after=positions+1, new_tokens_after_KO=refill,
            Il13_rank_before=int(inputs.cells.Il13_rank.iloc[row]))
        temp.replace(final)
        if (k+1) % 20 == 0 or k == 0 or k+1 == len(pending):
            elapsed = time.monotonic()-start
            print(f"{name}: cells {len(inputs.groups)-len(pending)+k+1}/{len(inputs.groups)} "
                  f"new_elapsed={elapsed:.1f}s ETA={elapsed/(k+1)*(len(pending)-k-1)/60:.1f}min", flush=True)
    del model
    torch.cuda.empty_cache()


def screen_effects(targets):
    """Descriptive NT-centred log-ratio; intentionally NOT a caRpools p-value reproduction."""
    folder = ROOT / "data/processed/GSE262028_screen_inputs_v1/IL13"
    result = targets[["ensembl_gene_id", "author_symbol", "S17_n_guides_author50", "S25_n_guides_author50"]].copy()
    provenance = {}
    for screen in ["S17", "S25"]:
        path = folder / f"{screen}_counts_author50.tsv"
        d = pd.read_csv(path, sep="\t")
        nt = set((folder / f"{screen}_nt_author50.txt").read_text().splitlines())
        if (d[[f"{screen}_N", f"{screen}_P"]] <= 0).any().any() or nt != set(d.loc[d.Gene.eq("__NT__"), "sgRNA"]):
            raise ValueError("Invalid author-filtered screen inputs")
        ratio = np.log2(d[f"{screen}_P"] / d[f"{screen}_N"])
        centre = float(np.median(ratio[d.sgRNA.isin(nt)]))
        d["effect"] = ratio - centre
        grouped = d[~d.sgRNA.isin(nt)].groupby("Gene").effect.agg(["median", "size"])
        result[f"{screen}_effect"] = result.author_symbol.map(grouped["median"])
        if not np.array_equal(result.author_symbol.map(grouped["size"]), result[f"{screen}_n_guides_author50"]):
            raise ValueError("Screen guide counts changed since candidate selection")
        provenance[screen] = dict(sha256=shared.file_hash(path), nt_guides=len(nt), nt_log2_ratio_median=centre)
    result["mean_signed_effect"] = result[["S17_effect", "S25_effect"]].mean(axis=1)
    result["mean_abs_effect"] = result[["S17_effect", "S25_effect"]].abs().mean(axis=1)
    result["screen_sign_agreement"] = np.sign(result.S17_effect) == np.sign(result.S25_effect)
    return result, provenance


def summarize(inputs, folder, smoke):
    tables = {}
    for name in ["raw", "lora"]:
        signature = json.loads((folder / name / "run_signature.json").read_text())
        if (signature["model"] != name or signature["manifest_sha256"] != inputs.design["manifest_sha256"]
            or signature["pair_ids"] != inputs.pairs.pair_id.tolist()
            or signature["source_sha256"] != inputs.design["source_sha256"]):
            raise ValueError("Cached run does not match this fixed evaluation")
        for path, digest in signature["hashes"].items():
            if shared.file_hash(ROOT / path) != digest:
                raise ValueError(f"Cached model/code changed: {path}")
        pieces = []
        for row, pairs in inputs.groups.items():
            path = folder / name / "cells" / f"{row:06d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Evaluation incomplete: {path}")
            z = np.load(path)
            expected = np.unique(pairs[["target_token_id", *inputs.random_columns]].to_numpy(dtype=np.int64))
            if int(z["cell_row"]) != row or not np.array_equal(expected, z["perturbed_token"]):
                raise ValueError("Cached intervention identities differ from fixed roster")
            dist = dict(zip(z["perturbed_token"], z["Il13_distance"]))
            rank = dict(zip(z["perturbed_token"], z["Il13_rank_after"]))
            d = pairs.copy()
            d["target_distance"] = d.target_token_id.map(dist)
            for j in range(3):
                d[f"random{j}_distance"] = d[f"random{j}_token_id"].map(dist)
            d["target_Il13_rank_after"] = d.target_token_id.map(rank)
            d["valid"] = np.isfinite(d[["target_distance", *[f"random{j}_distance" for j in range(3)]]]).all(axis=1)
            # skipna=False: a missing random control must not silently change the control denominator.
            d["random_mean_distance"] = d[[f"random{j}_distance" for j in range(3)]].mean(axis=1, skipna=False)
            d["excess_distance"] = d.target_distance-d.random_mean_distance
            pieces.append(d)
        tables[name] = pd.concat(pieces).sort_values("pair_id").reset_index(drop=True)
    if not np.array_equal(tables["raw"].pair_id, tables["lora"].pair_id):
        raise ValueError("Models have different evaluation pairs")
    common = tables["raw"].valid & tables["lora"].valid
    output = []
    for name, table in tables.items():
        table["common_valid"] = common
        table.to_csv(folder / name / "cell_scores.tsv.gz", sep="\t", index=False)
        for gene, d in table.groupby("target_gene_id"):
            valid = d[d.common_valid]
            for subset, v in [("primary", valid), ("top_half", valid[valid.target_in_token_top_half])]:
                output.append(dict(model=name, gene_id=gene, gene_symbol=d.target_symbol.iloc[0], subset=subset,
                    planned_cells=len(d) if subset == "primary" else int(d.target_in_token_top_half.sum()),
                    valid_cells=len(v), mean_target_distance=v.target_distance.mean(),
                    mean_random_distance=v.random_mean_distance.mean(), mean_excess_distance=v.excess_distance.mean(),
                    median_excess_distance=v.excess_distance.median(),
                    fraction_excess_positive=(v.excess_distance>0).mean() if len(v) else np.nan,
                    mean_abs_Il13_rank_shift=(v.target_Il13_rank_after-v.Il13_rank).abs().mean(),
                    eligible_ge100=len(v)>=inputs.design["config"]["minimum_joint_cells"]))
    scores = pd.DataFrame(output)
    effects, screen_provenance = screen_effects(inputs.targets)
    effects.to_csv(folder / "screen_descriptive_effects.csv", index=False)
    shared.write_json(folder / "screen_method.json", dict(inputs=screen_provenance,
        method="Per-guide log2(P/N) minus median NT log2(P/N); per-gene median; no p-values/FDR; not caRpools reproduction"))
    scores = scores.merge(effects, left_on="gene_id", right_on="ensembl_gene_id", validate="many_to_one")
    scores.to_csv(folder / "gene_scores.csv", index=False)
    comparisons = scores.pivot(index=["gene_id", "gene_symbol", "subset"], columns="model",
        values=["valid_cells", "mean_target_distance", "mean_random_distance", "mean_excess_distance"])
    comparisons.columns = [f"{metric}_{name}" for metric, name in comparisons.columns]
    comparisons.to_csv(folder / "model_comparison.csv")
    metrics = []
    if not smoke:
        for (name, subset), d in scores[scores.eligible_ge100].groupby(["model", "subset"]):
            for endpoint in ["S17_effect", "S25_effect", "mean_abs_effect"]:
                truth = d[endpoint].abs()
                for predicted in ["mean_target_distance", "mean_excess_distance"]:
                    rho = float(spearmanr(d[predicted], truth).statistic) if len(d)>2 and d[predicted].nunique()>1 and truth.nunique()>1 else None
                    top = min(50, len(d))
                    a = set(d.nlargest(top, predicted).gene_id)
                    b = set(d.assign(truth=truth).nlargest(top, "truth").gene_id)
                    metrics.append(dict(model=name, subset=subset, screen=endpoint, score=predicted,
                        genes=len(d), spearman_vs_abs_screen_effect=rho,
                        top_n=top, top_n_overlap=len(a & b), top_n_overlap_fraction=len(a & b)/top))
        pd.DataFrame(metrics).to_csv(folder / "screen_concordance.csv", index=False)
    status = dict(engineering_smoke=smoke, genes=int(inputs.pairs.target_gene_id.nunique()),
        planned_pairs=len(common), common_valid_pairs=int(common.sum()), excluded_pairs=int((~common).sum()),
        completed_models=["raw", "lora"], actual_used_cells=len(inputs.groups))
    shared.write_json(folder / "complete.json", status)
    lines = ["# IL-13目标基因扰动评估", "", json.dumps(status, ensure_ascii=False), "",
        "两个模型均使用冻结229目标设计；smoke仅取3基因各2细胞，工程结果不能解释为生物学评测。",
        "主指标：Il13末层gene embedding的1-cos(after,before)，再减去同细胞3个随机KO的平均距离。",
        "正的额外变化表示对该目标更敏感，不表示促进/抑制，不是朝真实KO状态的相似度增益。",
        "两个模型只汇总共同有效组合；缺少Il13的任一变体会排除整个组合，详见cell_scores.tsv.gz。",
        "随机变体按cell+KO基因去重缓存；不同目标仍保留各自预先抽样的3个随机对照。",
        "sgRNA采用NT中心化log2(P/N)的基因内中位数，分别比较S17/S25以及平均绝对效应；这是描述性对照，不是原论文caRpools显著性复现。",
        "不同模型余弦空间尺度不同，不能仅因LoRA的距离更大就判定更好；应比较筛选效应一致性、排序与随机参照。",
        "每个基因是汇总单位；细胞不是独立生物重复。两个筛选与新生肺参考存在背景差异。",
        "", "输出：gene_scores.csv（主分析及前50%敏感性）；model_comparison.csv；screen_descriptive_effects.csv；正式全量另有screen_concordance.csv。"]
    (folder / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(status, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["raw", "lora", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or (args.check_only and args.summarize_only):
        p.error("Invalid batch size or incompatible stages")
    inputs = Inputs(args.smoke)
    folder = OUT / "smoke" if args.smoke else OUT
    folder.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        changed = missing = 0
        for row, pairs in inputs.groups.items():
            _, _, positions, refill = inputs.cell_sequences(row, pairs)
            changed += len(positions)
            missing += int((positions<0).sum())
        shared.write_json(folder / "input_check.json", dict(cells=len(inputs.groups), pairs=len(inputs.pairs),
            unique_interventions=changed, missing_Il13_variants=missing, original_baselines_match=True))
        print("Input check passed", flush=True)
        return
    if not args.summarize_only:
        for name in (["raw", "lora"] if args.model == "both" else [args.model]):
            run_model(inputs, name, folder / name, args.batch_size)
    if args.summarize_only or args.model == "both":
        summarize(inputs, folder, args.smoke)
    else:
        print("Single model complete; run the other model then --summarize-only", flush=True)


if __name__ == "__main__":
    main()
