"""Select a CD8 cross-batch contrastive loss on frozen mouse/cytokine splits."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

if __package__:
    from . import cd8_retrieval, train
    from .cd8_pair_loss import DONOR, FIXED_EPOCH, frozen_rows
    from .gene_scores import pcc_delta
    from .mouse_donor import N_REPEATS
    from .mouse_panels import TOP_K, panel_indices
    from .mouse_scores import pds_pools
else:
    import cd8_retrieval, train
    from cd8_pair_loss import DONOR, FIXED_EPOCH, frozen_rows
    from gene_scores import pcc_delta
    from mouse_donor import N_REPEATS
    from mouse_panels import TOP_K, panel_indices
    from mouse_scores import pds_pools


WEIGHTS = (0., 0.01, 0.05)
FIXED_WEIGHT = 0.01


def select_candidate(curve):
    """Reference PCC is that of the original pure-MSE, PDS-selected checkpoint."""
    if not np.isfinite(curve[["inner_PDS", "inner_PCC_top20"]]).all().all():
        raise ValueError("Undefined CD8 inner validation metric")
    ranked = curve.sort_values(["inner_PDS", "weight", "epoch"], ascending=[False, True, True])
    baseline = ranked[ranked.weight.eq(0)].iloc[0]
    eligible = curve.inner_PCC_top20.ge(baseline.inner_PCC_top20)
    return ranked.loc[eligible].index[0], eligible


def save_matched_mse(ds, truth, device, dest, outer, test):
    """Isolate loss from duration when the inner selector changes the training epoch."""
    path = dest / "mse_matched_epoch_delta.npy"
    if path.exists():
        return
    curve = pd.read_csv(dest / "validation.tsv", sep="\t")
    chosen = curve[curve.selected].iloc[0]
    with TemporaryDirectory(dir=dest) as tmp:
        stage = Path(tmp) / path.name
        if chosen.weight == 0:
            copyfile(dest / "retrieval_delta.npy", stage)
        else:
            epoch = int(chosen.epoch)
            test_x = ds.inputs(test, {}, device)

            def save(e, model):
                np.save(stage, model(*test_x).cpu().numpy())

            cd8_retrieval.fit(ds, truth, outer, 0., epoch, device, [epoch], save)
        stage.replace(path)


def run_repeat(ds, truth, args, repeat, protocol, panels):
    inner, outer, val, test, meta = frozen_rows(ds, repeat, protocol)
    dest = args.out / f"repeat_{repeat}"
    if dest.exists():
        if json.loads((dest / "split.json").read_text()) != meta:
            raise ValueError(f"Saved rows differ from the frozen CD8 split in {dest}")
        save_matched_mse(ds, truth, args.device, dest, outer, test)
        return
    val_x = ds.inputs(val, {}, args.device)
    inner_x = ds.inputs(inner, {}, args.device)
    fit_y = torch.as_tensor(truth[inner.target_index], device=args.device)
    fit_objective = cd8_retrieval.CrossBatchRetrieval(inner, fit_y)
    panel_rows = [(i, k, panels[k]) for i, k in enumerate(val.target_index) if len(panels[k]) == TOP_K]
    records = []
    for weight in WEIGHTS:
        def validate(epoch, model):
            p = model(*val_x).cpu().numpy()
            pools = pds_pools(val, truth, {"FiLM": p}, repeat)
            correlations = [pcc_delta(p[i, panel], truth[k, panel]) for i, k, panel in panel_rows]
            inner_p = model(*inner_x)
            records.append(dict(weight=weight, epoch=epoch, inner_PDS=np.mean([s["PDS_cosine"] for s in pools]),
                                inner_PCC_top20=np.nanmean(correlations), n_PDS_pools=len(pools),
                                n_PCC_mice=int(np.isfinite(correlations).sum()),
                                fit_InfoNCE=fit_objective(inner_p).item(), fit_MSE=(inner_p - fit_y).square().mean().item(),
                                fit_batch_pairs=fit_objective.n_pairs, fit_covered_factors=fit_objective.n_ligands))

        cd8_retrieval.fit(ds, truth, inner, weight, max(train.EPOCHS), args.device, train.EPOCHS, validate)
    curve = pd.DataFrame(records)
    selected, eligible = select_candidate(curve)
    curve["eligible"] = eligible
    curve["selected"] = curve.index == selected
    chosen = curve.loc[selected]
    epoch, weight = int(chosen.epoch), float(chosen.weight)

    # Refit the selected setting and one prespecified fixed-positive sensitivity on outer fit rows.
    test_x = ds.inputs(test, {}, args.device)
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp) / dest.name
        stage.mkdir()
        for w in sorted({weight, FIXED_WEIGHT}):
            milestones = {FIXED_EPOCH}
            if w == weight:
                milestones.add(epoch)

            def save(e, model):
                p = model(*test_x).cpu().numpy()
                if w == weight and e == epoch:
                    np.save(stage / "retrieval_delta.npy", p)
                    torch.save(dict(state_dict=model.state_dict(), gene_id=ds.genes, settings=protocol,
                                    weight=weight, repeat=repeat, selected_epoch=epoch), stage / "retrieval.pt")
                if w == weight and e == FIXED_EPOCH:
                    np.save(stage / f"retrieval_epoch{e}_delta.npy", p)
                if w == FIXED_WEIGHT and e == FIXED_EPOCH:
                    np.save(stage / f"fixed001_epoch{e}_delta.npy", p)

            cd8_retrieval.fit(ds, truth, outer, w, max(milestones), args.device, milestones, save)
        curve.to_csv(stage / "validation.tsv", sep="\t", index=False)
        test.to_csv(stage / "predictions.tsv", sep="\t", index=False)
        (stage / "split.json").write_text(json.dumps(meta) + "\n")
        stage.rename(dest)
    save_matched_mse(ds, truth, args.device, dest, outer, test)
    print(f"Repeat {repeat}: lambda={weight}, epoch={epoch}, validation PDS={chosen.inner_PDS:.4f}, "
          f"Top-20 PCC={chosen.inner_PCC_top20:.4f}", flush=True)


def run(ds, args):
    protocol = dict(input_id=ds.input_id, target_id=train.target_id(ds), celltype="T_cell_CD8",
                    seed=train.SEED, epochs=list(train.EPOCHS), weights=list(WEIGHTS),
                    temperature=cd8_retrieval.TEMPERATURE, fixed_epoch=FIXED_EPOCH, fixed_weight=FIXED_WEIGHT,
                    targets="equal-mouse batch/canonical ligand means, centered over shared ligands on each side",
                    loss="full-gene mouse MSE + weight * bidirectional batch-pair mean InfoNCE",
                    selection="highest inner novel PDS with Top20 PCC >= PDS-selected pure-MSE PCC; "
                              "ties use lower weight then earlier epoch",
                    optimizer=train.ADAMW, batch_size="all fitting CD8 mice", shuffle_ligands=False,
                    device=args.device, threads=args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError(f"Existing protocol differs in {args.out}")
    if not path.exists():
        with TemporaryDirectory(dir=args.out) as tmp:
            p = Path(tmp) / path.name
            p.write_text(json.dumps(protocol, indent=2) + "\n")
            p.replace(path)
    truth = train.training_truth(ds)
    panels = panel_indices(ds, DONOR)
    with threadpool_limits(limits=1, user_api="blas"):
        for repeat in range(1, N_REPEATS + 1):
            run_repeat(ds, truth, args, repeat, protocol, panels)
        if __package__:
            from .cd8_retrieval_scores import evaluate
        else:
            from cd8_retrieval_scores import evaluate
        evaluate(ds, truth, args.out)
