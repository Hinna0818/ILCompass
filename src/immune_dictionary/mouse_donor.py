"""PULSAR-style donor/cytokine evaluation for a one-treatment-per-mouse experiment."""

import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
import torch

if __package__:
    from . import train
    from .gene_scores import pcc_delta
else:
    import train
    from gene_scores import pcc_delta


SEEN = "new_mouse_seen_cytokine"
NEW = "new_mouse_new_cytokine"
N_REPEATS = 5
MAJOR_FRACTION = 0.05


def reference_design(ds):
    pb = ad.read_h5ad(ds.folder / "rna/pseudobulk_counts.h5ad", backed="r")
    pbs = pb.obs[pb.obs.cyt_display == "PBS"].copy()
    pb.file.close()
    pbs["mouse_id"] = "PBS|" + pbs.rep.astype(str)
    ctx = pd.read_csv(ds.folder / "contexts.tsv", sep="\t").set_index("context_index")
    for c in ctx.itertuples():
        mice = pbs.loc[c.pbs_sample_ids.split(",")]
        if not (mice.processing_batch.eq(c.processing_batch) & mice.celltype.eq(c.celltype)).all():
            raise ValueError("PBS samples do not match the listed batch and cell type")
    rows = ds.examples.copy()
    # sample_id is a mouse/type pseudobulk, while rep01 can recur for every reagent.
    rows["mouse_id"] = rows.cyt_display.astype(str) + "|" + rows.rep.astype(str)
    rows["pbs_sample_ids"] = rows.context_index.map(ctx.pbs_sample_ids)
    if (not rows.processing_batch.eq(rows.context_index.map(ctx.processing_batch)).all()
            or not rows.celltype.eq(rows.context_index.map(ctx.celltype)).all()
            or rows.groupby("mouse_id").processing_batch.nunique().max() != 1
            or pbs.groupby("mouse_id").processing_batch.nunique().max() != 1):
        raise ValueError("A mouse or its PBS reference has inconsistent batch or cell-type metadata")
    return rows, pbs


def partition(rows, rng, n_batches, n_ligands):
    available = sorted(rows.processing_batch.unique().tolist())
    by_batch = {b: set(r.split_ligand) for b, r in rows.groupby("processing_batch")}

    def candidate_batches():
        yield sorted(rng.choice(available, n_batches, replace=False).tolist())
        candidates = list(combinations(available, n_batches))
        for i in rng.permutation(len(candidates)):
            yield list(candidates[i])

    # Reagent panels are partly nested in processing batches. Sample only designs that can measure both tasks.
    for batches in candidate_batches():
        in_test = set.union(*(by_batch[b] for b in batches))
        elsewhere = set.union(*(by_batch[b] for b in available if b not in batches))
        forced = in_test - elsewhere
        if len(forced) <= n_ligands < len(in_test):
            break
    else:
        raise ValueError("Cannot form the specified number of measurable held-out reagents")
    held = rows.processing_batch.isin(batches)
    others = rng.choice(sorted(in_test - forced), n_ligands - len(forced), replace=False)
    ligands = sorted(forced.union(others.tolist()))
    novel = rows.split_ligand.isin(ligands)
    role = pd.Series(np.select([held & novel, held, novel], [NEW, SEEN, "excluded"], default="fit"),
                     index=rows.index)
    return role, {"held_batches": batches, "held_split_ligands": ligands}


def make_repeat(rows, pbs, repeat):
    rng = np.random.default_rng(train.SEED + repeat - 1)
    roles = rows.copy()
    roles["outer_role"], outer = partition(rows, rng, n_batches=4, n_ligands=20)
    fit = roles[roles.outer_role == "fit"]
    roles["inner_role"] = "unused"
    inner_roles, inner = partition(fit, rng, n_batches=3, n_ligands=15)
    roles.loc[fit.index, "inner_role"] = inner_roles
    # The denominator includes all author-labelled immune PBS cells, including rare types below the fit cutoff.
    def pbs_fractions(batches):
        allowed = pbs[pbs.processing_batch.isin(batches)]
        cells = allowed.groupby("celltype", observed=True).n_cells.sum()
        return (cells / cells.sum()).to_dict()

    outer["pbs_cell_fraction"] = pbs_fractions(fit.processing_batch.unique())
    inner["pbs_cell_fraction"] = pbs_fractions(fit.loc[inner_roles == "fit", "processing_batch"].unique())
    return roles, {"repeat": repeat, "split_seed": train.SEED + repeat - 1, "outer": outer, "inner": inner}


def fit_repeat(ds, truth, rows, design, args):
    repeat = design["repeat"]
    dest = args.out / f"repeat_{repeat}"
    settings = dict(input_id=ds.input_id, target_id=train.target_id(ds), batch_size=args.batch_size,
                    epochs=list(train.EPOCHS), optimizer=train.ADAMW, weight_by="mouse_id",
                    selection="max mean of the two inner-regime mouse-mean all-gene PCCs; earliest tie",
                    major_fraction=MAJOR_FRACTION, seed=train.SEED, shuffle_ligands=False)
    if dest.exists():
        saved = torch.load(dest / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["design"] != design or saved["settings"] != settings:
            raise ValueError(f"Existing repeat used different inputs or settings: {dest}")
        return dest
    inner = rows[rows.inner_role == "fit"]
    def train_fn(fit, epochs, **kw):
        return train.fit_model(ds, fit, truth, epochs, args.device, args.batch_size, False,
                               weight_by=["mouse_id"], **kw)
    fractions = design["inner"]["pbs_cell_fraction"]
    val = rows[rows.inner_role.isin([SEEN, NEW]) & rows.celltype.map(fractions).ge(MAJOR_FRACTION)].copy()
    x_val = ds.inputs(val, {}, args.device)
    y_val = truth[val.target_index]
    curve = []

    def observe(epoch, model):
        pred = model(*x_val).cpu().numpy()
        val["pcc"] = [pcc_delta(p, y) for p, y in zip(pred, y_val)]
        scores = val.groupby(["inner_role", "mouse_id"]).pcc.mean().groupby(level=0).mean()
        if not np.isfinite(scores.reindex([SEEN, NEW])).all():
            raise ValueError("Both validation regimes need measured major cell types and finite PCCs")
        curve.append({"epoch": epoch, SEEN: scores[SEEN], NEW: scores[NEW], "mean_pcc": scores.mean()})

    train_fn(inner, max(train.EPOCHS), observe=observe, observe_epochs=train.EPOCHS)
    chosen = max(curve, key=lambda e: e["mean_pcc"])["epoch"]
    outer = rows[rows.outer_role == "fit"]
    model, _, _ = train_fn(outer, chosen)
    test = rows[rows.outer_role.isin([SEEN, NEW])].copy().reset_index(drop=True)
    test.insert(0, "prediction_index", np.arange(len(test)))
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp) / dest.name
        stage.mkdir()
        train.predict_oof(ds, test, model, {}, stage, args.device)
        rows[["target_index", "mouse_id", "sample_id", "celltype", "cyt_display", "split_ligand",
              "processing_batch", "context_index", "pbs_sample_ids", "outer_role", "inner_role"]].to_csv(
                  stage / "roles.tsv.gz", sep="\t", index=False)
        (stage / "split.json").write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n")
        pd.DataFrame(curve).to_csv(stage / "validation.tsv", sep="\t", index=False)
        torch.save(dict(state_dict=model.cpu().state_dict(), gene_id=ds.genes, settings=settings, design=design,
                        selected_epoch=chosen), stage / "checkpoint.pt")
        stage.rename(dest)
    print(f"Repeat {repeat}: selected epoch {chosen}, trained on {outer.mouse_id.nunique()} mice,"
          f" predicted {test.mouse_id.nunique()} held-out mice.", flush=True)
    return dest


def run(ds, args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, pbs = reference_design(ds)
    truth = train.training_truth(ds)
    runs = []
    for repeat in range(1, N_REPEATS + 1):
        roles, design = make_repeat(rows, pbs, repeat)
        dest = fit_repeat(ds, truth, roles, design, args)
        runs.append((dest, roles, design))
    # Measured gene panels are opened only by the evaluator, after all five predictions are fixed.
    if __package__:
        from .mouse_scores import evaluate
    else:
        from mouse_scores import evaluate
    evaluate(ds, truth, runs, args.out, args.edger)
