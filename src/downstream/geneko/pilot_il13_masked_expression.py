"""Exploratory Tcf12/Maf KO: hide Il13 value, retain its ID, use native expression head.

Two frozen models, all fixed target reference cells, original 3 random KOs.
Report fixed-slot expression direction and reranking sensitivity separately.
"""
import argparse
import json
from pathlib import Path
import time

from evaluate_il13_screen_ko import Inputs, shared
import numpy as np
import pandas as pd
import torch

ROOT = shared.ROOT
BATCH = 2


@torch.inference_mode()
def predict(model, sequences, drop_tokens, readout, model_name):
    batch = shared.batch(sequences)
    positions = []
    for j, ((ids, _), drop) in enumerate(zip(sequences, drop_tokens)):
        pos = np.flatnonzero(ids == readout)
        if len(pos) != 1:
            raise ValueError("Il13 query missing or duplicated")
        positions.append(int(pos[0]))
        batch["values"][j, pos[0]] = -1.
        if drop is not None:
            where = np.flatnonzero(ids == drop)
            if len(where) != 1 or drop == readout:
                raise ValueError("Invalid fixed-slot context deletion")
            p = int(where[0])
            batch["input_ids"][j, p] = model.config.pad_token_id
            batch["values"][j, p] = 0.
            batch["attention_mask"][j, p] = 0
    br = torch.arange(len(sequences), device="cuda")
    pp = torch.tensor(positions, device="cuda")
    if not (batch["values"][br, pp] == -1).all() or not (batch["input_ids"][br, pp] == readout).all():
        raise ValueError("Il13 value masking / ID preservation failed")
    h = shared.encode(model, batch, model_name)
    if h.shape[1] != batch["input_ids"].shape[1] + 1:
        raise ValueError("Expected CLS offset")
    prediction = model.cls4value(h[br, pp+1]).flatten().float().cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite expression prediction")
    return prediction.astype(np.float64)


def logcp10k(pred, factor):
    # Same stable inverse transformation as functional-reconstruction evaluation.
    t = np.maximum(pred, 0.) * np.log(2)
    logcount = np.full_like(t, -np.inf)
    nz = t > 0
    logcount[nz] = t[nz] + np.log(-np.expm1(-t[nz]))
    return np.logaddexp(0, logcount + np.log(factor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["Tcf12", "Maf"], default="Tcf12")
    args = parser.parse_args()
    target = args.target
    OUT = ROOT / f"results/GSE262028_{target}_IL13_masked_expression_v1"
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(smoke=False)
    pairs = inputs.pairs[inputs.pairs.target_symbol.eq(target)].copy()
    expected = int(inputs.targets.loc[inputs.targets.author_symbol.eq(target), "joint_Il13_token_cells"].iloc[0])
    if len(pairs) != expected or pairs.cell_row.duplicated().any():
        raise ValueError("Expected complete fixed target cohort")
    effects = pd.read_csv(ROOT / "results/GSE262028_IL13_insilico_v1/screen_descriptive_effects.csv")
    screen = effects[effects.author_symbol.eq(target)].iloc[0]
    readout_col = inputs.column_by_token[inputs.readout]
    # Full original-library denominator, shared by control and every variant (no counterfactual renormalization).
    import anndata as ad
    a = ad.read_h5ad(ROOT / inputs.design["config"]["reference_directory"] / "GSE126924_Lung_IL13positive_ILC2_1864.h5ad")
    totals = np.asarray(a.X.sum(axis=1)).ravel()
    del a
    screen_record = {k: screen[k].item() if hasattr(screen[k], "item") else screen[k]
                     for k in ["author_symbol", "S17_effect", "S25_effect", "mean_signed_effect"]}
    provenance = dict(target=target, exploratory_selection="chosen after prior screen ranking/direction; not an independent unbiased benchmark",
        cells=len(pairs), mask="Il13 value=-1, ID retained; gene membership and original rank still reveal information",
        protocols=["rerank", "fixed_slot"], random_controls="same locked 3 genes per cell; no resampling",
        precision="FP32 inference; native log2(1+counts/proxy_median) head; TF32 off",
        fixed_slot="replace target slot with PAD/value0 and attention_mask0; other IDs/values/positions unchanged; no refill",
        median_reference="same frozen proxy reference as previous GC/LoRA KO task",
        screen=screen_record, manifest_sha256=inputs.design["manifest_sha256"],
        script_sha256=shared.file_hash(Path(__file__)))
    shared.write_json(OUT / "config.json", provenance)
    pairs.to_csv(OUT / "fixed_pairs.tsv.gz", sep="\t", index=False)
    all_records, checks = [], {}
    for name in ["raw", "lora"]:
        model = shared.load(name).float().eval().requires_grad_(False)
        records = []
        start = time.monotonic()
        for k, pair in enumerate(pairs.itertuples()):
            r = int(pair.cell_row)
            token_ids, seq, positions, _ = inputs.cell_sequences(r, pairs[pairs.cell_row.eq(r)])
            if (positions < 0).any():
                raise ValueError("Missing Il13 variant; cannot compare all four interventions")
            original = seq[0]
            pred0 = float(predict(model, [original], [None], inputs.readout, name)[0])
            if k == 0:
                sham = predict(model, [original, original], [None, None], inputs.readout, name)
                error = float(np.max(np.abs(sham-pred0)))
                if error > 1e-4:
                    raise ValueError(f"Masked expression repeat/sham mismatch: {error}")
                # Changing the hidden numeric value must not change the actual query inputs.
                altered = (original[0].copy(), original[1].copy())
                q = int(np.flatnonzero(original[0] == inputs.readout)[0])
                altered[1][q] = 999.
                leak = predict(model, [original, altered], [None, None], inputs.readout, name)
                leak_error = float(abs(leak[0]-leak[1]))
                if leak_error > 1e-6:
                    raise ValueError("Il13 numeric value leaked into query")
                checks[name] = dict(sham_native_max_abs=error, hidden_value_invariance_max_abs=leak_error)
            preds = {}
            for protocol in ["rerank", "fixed_slot"]:
                values = []
                for left in range(0, len(token_ids), BATCH):
                    tt = token_ids[left:left+BATCH]
                    if protocol == "rerank":
                        subset, drop = seq[left+1:left+1+len(tt)], [None]*len(tt)
                    else:
                        subset, drop = [original]*len(tt), list(tt)
                    values.extend(predict(model, subset, drop, inputs.readout, name))
                preds[protocol] = dict(zip(token_ids, values))
            factor = inputs.scale[readout_col] * 10000. / totals[r]
            truth = float(original[1][np.flatnonzero(original[0] == inputs.readout)[0]])
            for protocol, pred in preds.items():
                after = np.array([pred[pair.target_token_id], *[pred[getattr(pair, f"random{j}_token_id")] for j in range(3)]])
                delta = after-pred0
                delta_log = logcp10k(after, factor)-float(logcp10k(np.array([pred0]), factor)[0])
                record = dict(model=name, protocol=protocol, pair_id=pair.pair_id, cell_id=pair.cell_id,
                    cell_row=r, true_Il13_native=truth, pred_before_native=pred0,
                    pred_after_target_native=after[0], delta_target_native=delta[0],
                    delta_random_mean_native=float(delta[1:].mean()), excess_delta_native=float(delta[0]-delta[1:].mean()),
                    pred_before_logCP10K=float(logcp10k(np.array([pred0]), factor)[0]),
                    true_Il13_logCP10K=float(logcp10k(np.array([truth]), factor)[0]),
                    delta_target_logCP10K=delta_log[0], delta_random_mean_logCP10K=float(delta_log[1:].mean()),
                    excess_delta_logCP10K=float(delta_log[0]-delta_log[1:].mean()),
                    Il13_rank_before=pair.Il13_rank,
                    Il13_rank_after=int(positions[np.flatnonzero(token_ids==pair.target_token_id)[0]])+1 if protocol=="rerank" else pair.Il13_rank)
                for j in range(3):
                    record[f"delta_random{j}_native"] = delta[j+1]
                records.append(record)
            if (k+1) % 50 == 0 or k == 0 or k+1 == len(pairs):
                print(name, k+1, "/", len(pairs), f"elapsed={time.monotonic()-start:.1f}s", flush=True)
        pd.DataFrame(records).to_csv(OUT / f"{name}_cell_predictions.csv.gz", index=False)
        all_records.extend(records)
        provenance[name] = shared.provenance(name)
        del model
        torch.cuda.empty_cache()
    d = pd.DataFrame(all_records)
    summary = []
    for (name, protocol), group in d.groupby(["model", "protocol"]):
        # Numerical guard only, not a biological threshold or significance test.
        epsilon = max(1e-6, 10*checks[name]["sham_native_max_abs"])
        summary.append(dict(model=name, protocol=protocol, cells=len(group),
            mean_pred_before_native=group.pred_before_native.mean(), mean_true_native=group.true_Il13_native.mean(),
            baseline_native_MAE=(group.pred_before_native-group.true_Il13_native).abs().mean(),
            baseline_logCP10K_MAE=(group.pred_before_logCP10K-group.true_Il13_logCP10K).abs().mean(),
            mean_target_delta_native=group.delta_target_native.mean(),
            median_target_delta_native=group.delta_target_native.median(),
            mean_random_delta_native=group.delta_random_mean_native.mean(),
            mean_excess_delta_native=group.excess_delta_native.mean(),
            mean_target_delta_logCP10K=group.delta_target_logCP10K.mean(),
            mean_excess_delta_logCP10K=group.excess_delta_logCP10K.mean(),
            numerical_threshold=epsilon, fraction_up=(group.delta_target_native>epsilon).mean(),
            fraction_down=(group.delta_target_native < -epsilon).mean(),
            fraction_excess_positive=(group.excess_delta_native>epsilon).mean(),
            changed_Il13_rank_cells=int((group.Il13_rank_before!=group.Il13_rank_after).sum())))
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False)
    shared.write_json(OUT / "checks.json", checks)
    shared.write_json(OUT / "provenance.json", provenance)
    table = "\n".join(f"|{r.model}|{r.protocol}|{r.mean_target_delta_native:.6g}|{r.mean_random_delta_native:.6g}|{r.mean_excess_delta_native:.6g}|{r.fraction_up:.1%}|{r.fraction_down:.1%}|" for r in s.itertuples())
    report = f"""# {target} KO → Il13 masked-value reconstruction pilot

实际运行raw/LoRA两个冻结模型，共{len(pairs)}个固定参考细胞；原3个排名匹配随机KO复用。Il13基因ID保留，表达输入设为-1，不使用真实Il13数值预测。
调用原生cls4value表达头，无训练。这里只遮蔽value，与预训练的ID/value联合遮蔽不同；这是有条件查询的探索性测试。
gene身份、是否被检出和由原始表达确定的rank仍可见，不能称完全无表达信息泄漏。

实验参照{target}：S17={screen.S17_effect:.6f}、S25={screen.S25_effect:.6f}，为guide的NT中心化log2(P/N)基因内中位数；正值偏向IL13报告阳性群体，负值偏向阴性群体，不是Il13 RNA均值的logFC。
候选因前一轮screen效应/方向被事后选作试点；单基因不能用于无偏整体性能判断。

Δ=预测KO后Il13−预测未KO的Il13，两次均遮蔽Il13值。native单位为log2(1+counts/proxy_median)。
额外Δ=目标Δ−三次随机KO的平均Δ。另输出使用固定原始文库总量转换的log1p(CP10K)差；不把二者与screen效应直接数值相减。

|模型|处理|平均目标Δ|平均随机Δ|平均额外Δ|上升细胞比例|下降细胞比例|
|---|---|---:|---:|---:|---:|---:|
{table}

rerank沿用此前置零、完整重新排序和Top2048；fixed_slot在基线原槽位用PAD/0及attention_mask=0移除目标上下文，保留其他位置，不从2048以外补位。
fixed_slot因此同时控制位置与输入组成，不能称只改变一个位置因素的严格因果分解。
上升/下降按10倍实测sham误差（至少1e-6）划分，属于数值容差，不是显著性阈值；细胞不是独立动物重复。
没有真实{target} KO单细胞表达谱，不能计算KO预测MAE；baseline重构MAE仅针对未扰动Il13原始表达。

复现：`python -u src/downstream/geneko/pilot_il13_masked_expression.py --target {target}`。
结果：summary.csv、两模型cell_predictions.csv.gz、checks.json、provenance.json。未扩展到其余228目标。
"""
    (OUT / "report.md").write_text(report)
    shared.write_json(OUT / "complete.json", dict(models=["raw", "lora"], target=target, cells=len(pairs), protocols=["rerank", "fixed_slot"]))
    print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
