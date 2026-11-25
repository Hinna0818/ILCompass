"""CD8-only FiLM ablation on the frozen GSE202186 mouse-donor splits."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

if __package__:
    from . import train
    from .mouse_donor import NEW, SEEN, N_REPEATS
    from .mouse_scores import pds_pools
else:
    import train
    from mouse_donor import NEW, SEEN, N_REPEATS
    from mouse_scores import pds_pools


DONOR = train.ROOT / "results/GSE202186_PULSAR_mouse_donor"
PAIR_WEIGHT = 0.5
FIXED_EPOCH = 30
GROUP = ["processing_batch", "canonical_id"]


def condition_averager(rows):
    """Map mice to equal-mouse residual means for each matched batch and reagent."""
    code = rows.groupby(GROUP, sort=False).ngroup().to_numpy()
    counts = np.bincount(code)
    mean = np.zeros((len(counts), len(rows)), dtype=np.float32)
    mean[code, np.arange(len(rows))] = 1 / counts[code]
    return rows.drop_duplicates(GROUP).reset_index(drop=True), torch.from_numpy(mean)


def pair_weights(rows):
    """Equal weight per eligible batch and distinct reagent pair, split over member mice."""
    w = np.zeros((len(rows), len(rows)), dtype=np.float32)
    n_batches = 0
    for _, batch in rows.reset_index(drop=True).groupby("processing_batch"):
        labels, code, counts = np.unique(batch.canonical_id, return_inverse=True, return_counts=True)
        k = len(labels)
        if k >= 2:
            each = 1 / counts[code]
            block = (code[:, None] != code[None, :]) * np.outer(each, each) / (k * (k - 1))
            w[np.ix_(batch.index, batch.index)] = block
            n_batches += 1
    return torch.from_numpy(w / max(n_batches, 1))


def pair_loss(errors, weights):
    """Mean ||e_i - e_j||² over weighted pairs and genes without a pair-by-gene tensor."""
    # For a symmetric W, sum_ij W_ij ||e_i-e_j||² = 2 tr(E' (diag(W1) - W) E).
    weights = weights.to(errors)
    laplacian_errors = weights.sum(1, keepdim=True) * errors - weights @ errors
    return 2 * (errors * laplacian_errors).sum() / errors.shape[1]


def fit(ds, truth, rows, arm, epochs, device, observe_epochs, observe):
    regularizer = None
    if arm == "pair":
        conditions, mean = condition_averager(rows)
        w = pair_weights(conditions).to(device)
        mean = mean.to(device)

        def regularizer(residual, ids):
            # The full fold makes every within-batch contrast available on every update.
            errors = residual[torch.argsort(ids)]
            if len(conditions) != len(rows):
                errors = mean @ errors
            return PAIR_WEIGHT * pair_loss(errors, w)

    return train.fit_model(ds, rows, truth, epochs, device, len(rows), False, weight_by=["mouse_id"],
                           observe_epochs=observe_epochs, observe=observe, extra_loss=regularizer)


def frozen_rows(ds, repeat, protocol):
    source = DONOR / f"repeat_{repeat}"
    old = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if old["settings"]["input_id"] != ds.input_id or old["settings"]["target_id"] != protocol["target_id"]:
        raise ValueError("Data package differs from the archived donor split")
    roster = pd.read_csv(source / "roles.tsv.gz", sep="\t")
    cd8 = ds.examples[ds.examples.celltype == "T_cell_CD8"]
    rows = cd8.merge(roster[["target_index", "sample_id", "mouse_id", "outer_role", "inner_role"]],
                     on=["target_index", "sample_id"], validate="1:1").sort_values("target_index")
    if len(rows) != len(cd8) or not rows.mouse_id.is_unique:
        raise ValueError("Expected one frozen role per CD8 mouse")
    inner, outer = (rows[rows[c].eq("fit")] for c in ["inner_role", "outer_role"])
    val = rows[rows.inner_role.eq(NEW)].reset_index(drop=True).assign(outer_role=NEW, major=True)
    test = rows[rows.outer_role.isin([NEW, SEEN])].reset_index(drop=True)
    for fitting, held in [(inner, val), (outer, test)]:
        if (set(fitting.processing_batch) & set(held.processing_batch)
                or set(fitting.split_ligand) & set(held.loc[held.outer_role.eq(NEW), "split_ligand"])):
            raise ValueError("A validation or test batch or novel ligand appears in the fit rows")
    test.insert(0, "prediction_index", np.arange(len(test)))
    meta = dict(repeat=repeat, inner_fit=inner.target_index.tolist(), outer_fit=outer.target_index.tolist(),
                validation=val.target_index.tolist(), test=test.target_index.tolist())
    return inner, outer, val, test, meta


def run_repeat(ds, truth, args, repeat, protocol):
    inner, outer, val, test, meta = frozen_rows(ds, repeat, protocol)
    dest = args.out / f"repeat_{repeat}"
    if dest.exists():
        if json.loads((dest / "split.json").read_text()) != meta:
            raise ValueError(f"Incompatible saved mouse roles in {dest}")
        return
    val_x = ds.inputs(val, {}, args.device)
    test_x = ds.inputs(test, {}, args.device)
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp) / dest.name
        stage.mkdir()
        curve = []
        for arm in ("mse", "pair"):
            pds = {}

            def validate(epoch, model):
                pred = model(*val_x).cpu().numpy()
                pools = pds_pools(val, truth, {arm: pred}, repeat)
                pds[epoch] = np.mean([p["PDS_cosine"] for p in pools])

            fit(ds, truth, inner, arm, max(train.EPOCHS), args.device, train.EPOCHS, validate)
            if not np.isfinite(list(pds.values())).all():
                raise ValueError("Undefined inner validation PDS")
            chosen = max(pds, key=pds.get)
            curve.extend(dict(arm=arm, epoch=e, inner_new_CD8_PDS=p, selected=e == chosen) for e, p in pds.items())

            def save(epoch, model):
                pred = model(*test_x).cpu().numpy()
                if epoch == FIXED_EPOCH:
                    np.save(stage / f"{arm}_epoch{FIXED_EPOCH}_delta.npy", pred)
                if epoch == chosen:
                    np.save(stage / f"{arm}_delta.npy", pred)
                    torch.save(dict(state_dict=model.state_dict(), gene_id=ds.genes, settings=protocol,
                                    arm=arm, repeat=repeat, selected_epoch=chosen), stage / f"{arm}.pt")

            milestones = {chosen, FIXED_EPOCH}
            fit(ds, truth, outer, arm, max(milestones), args.device, milestones, save)
        pd.DataFrame(curve).to_csv(stage / "validation.tsv", sep="\t", index=False)
        test.to_csv(stage / "predictions.tsv", sep="\t", index=False)
        (stage / "split.json").write_text(json.dumps(meta) + "\n")
        stage.rename(dest)
    print(f"Repeat {repeat}: {len(outer)} CD8 training mice, {len(test)} held out. "
          f"Epochs {[(r['arm'], r['epoch']) for r in curve if r['selected']]}", flush=True)


def run(ds, args):
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(input_id=ds.input_id, target_id=train.target_id(ds), celltype="T_cell_CD8", seed=train.SEED,
                    epochs=list(train.EPOCHS), fixed_epoch=FIXED_EPOCH, pair_weight=PAIR_WEIGHT,
                    paired_unit="processing batch and canonical reagent, mice averaged within each condition",
                    selection="maximum inner novel-cytokine CD8 PDS, earliest tied epoch",
                    optimizer=train.ADAMW, batch_size="all fitting CD8 mice", shuffle_ligands=False,
                    device=args.device, threads=args.threads)
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError(f"Existing protocol differs in {args.out}")
    if not path.exists():
        with TemporaryDirectory(dir=args.out) as tmp:
            stage = Path(tmp) / path.name
            stage.write_text(json.dumps(protocol, indent=2) + "\n")
            stage.replace(path)
    truth = train.training_truth(ds)
    with threadpool_limits(limits=1, user_api="blas"):
        for repeat in range(1, N_REPEATS + 1):
            run_repeat(ds, truth, args, repeat, protocol)
        if __package__:
            from .cd8_loss_scores import evaluate
        else:
            from cd8_loss_scores import evaluate
        evaluate(ds, truth, args.out)
        score_fit_losses(ds, truth, args.out)


@torch.inference_mode()
def score_fit_losses(ds, truth, out):
    """Check how the selected heads fit the absolute and pair targets of their own training mice."""
    records = []
    indexed = ds.examples.set_index("target_index", drop=False)
    for repeat in range(1, N_REPEATS + 1):
        folder = out / f"repeat_{repeat}"
        split = json.loads((folder / "split.json").read_text())
        rows = indexed.loc[split["outer_fit"]]
        inputs = ds.inputs(rows, {}, "cpu")
        y = torch.from_numpy(truth[rows.target_index])
        conditions, mean = condition_averager(rows)
        weights = pair_weights(conditions)
        for arm in ("mse", "pair"):
            cp, model = train.load_checkpoint(folder / f"{arm}.pt", "cpu")
            residual = model(*inputs) - y
            records.append(dict(repeat=repeat, arm=arm, selected_epoch=cp["selected_epoch"],
                                mouse_MSE=residual.square().mean().item(),
                                within_batch_pair_MSE=pair_loss(mean @ residual, weights).item()))
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp) / "training_losses.tsv"
        pd.DataFrame(records).to_csv(stage, sep="\t", index=False)
        stage.replace(out / stage.name)
