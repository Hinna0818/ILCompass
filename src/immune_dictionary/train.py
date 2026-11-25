"""Fit a held-out FiLM head, regenerate OOF predictions, or predict from new embeddings."""

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

if __package__:
    from .film import CytokineFiLM
    from .splits import outer_fit, routed_rows
else:
    from film import CytokineFiLM
    from splits import outer_fit, routed_rows


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/processed/GSE202186_perturbation_v1"
RESULTS = ROOT / "results/GSE202186_perturbation_model/film_v1"
SEED = 202186
EPOCHS = (10, 30, 60)
ADAMW = {"lr": 0.001, "weight_decay": 0.0001}
CONDITION = ["celltype", "cyt_display"]
FOLD = ["batch", "hold_cell_group", "hold_ligand_group"]


class Dataset:
    """Only observable inputs and design metadata. Stimulated targets are opened by train only."""

    def __init__(self, folder, *, load_folds=True):
        self.folder = Path(folder)
        self.examples = pd.read_csv(self.folder / "examples.tsv", sep="\t", keep_default_na=False)
        rows = self.examples
        if "split_ligand" not in rows:
            rows["split_ligand"] = rows["cyt_display"]
        self.folds = pd.read_csv(self.folder / "folds.tsv", sep="\t") if load_folds else None
        self.genes = pd.read_csv(self.folder / "genes.tsv", sep="\t", usecols=["gene_id"],
                                 dtype=str, keep_default_na=False)["gene_id"].tolist()
        with np.load(self.folder / "contexts.npz") as contexts:
            self.cell = contexts["embeddings"]
            self.baseline = contexts["baseline"]
        with np.load(self.folder / "ligands/esm2.npz") as ligands:
            labels = ligands["cyt_display"].astype(str).tolist()
            self.ligand = ligands["embeddings"]
        self.ligand_index = {label: i for i, label in enumerate(labels)}
        if (self.cell.shape != (len(self.baseline), 768)
                or self.baseline.shape != (len(self.cell), len(self.genes))
                or self.ligand.shape != (len(labels), 1280)):
            raise ValueError("Unexpected PBS, ESM2 or gene dimensions")
        if (len(self.genes) != len(set(self.genes)) or "" in self.genes
                or len(labels) != len(self.ligand_index)
                or set(rows.cyt_display) - self.ligand_index.keys()):
            raise ValueError("Gene or ligand axis is incomplete or ambiguous")
        if (rows.target_index.duplicated().any() or rows.duplicated(["sample_id", "celltype"]).any()
                or not rows.context_index.between(0, len(self.cell) - 1).all()
                or not rows.target_index.ge(0).all()):
            raise ValueError("Invalid mouse, PBS context or target index")
        for key, group in (("sample_id", "processing_batch"), ("celltype", "cell_group"),
                           ("split_ligand", "ligand_group"), ("context_index", "processing_batch"),
                           ("context_index", "celltype")):
            if rows.groupby(key)[group].nunique().gt(1).any():
                raise ValueError(f"{key} crosses {group} boundaries")
        # A reloaded OOF head must use exactly its reference, embeddings and row axes.
        digest = hashlib.sha256((self.folder / "examples.tsv").read_bytes())
        digest.update("\n".join(self.genes + labels).encode())
        for array in (self.cell, self.baseline, self.ligand):
            digest.update(np.ascontiguousarray(array))
        self.input_id = digest.hexdigest()

    def inputs(self, rows, shuffle_map, device):
        ligands = rows.split_ligand.map(shuffle_map).fillna(rows.cyt_display)
        ligand_ids = ligands.map(self.ligand_index).to_numpy(dtype=int)
        return (torch.as_tensor(self.cell[rows.context_index], device=device, dtype=torch.float32),
                torch.as_tensor(self.ligand[ligand_ids], device=device, dtype=torch.float32))


def inner_split(outer, fold):
    held_ligands = [fold.val_ligand_1, fold.val_ligand_2]
    held_type = outer.celltype.eq(fold.val_celltype)
    held_ligand = outer.split_ligand.isin(held_ligands)
    held_batch = outer.processing_batch.eq(fold.val_batch)
    fit = outer[~(held_type | held_ligand | held_batch)]
    val_a = outer[held_batch & held_type & outer.split_ligand.isin(fit.split_ligand)]
    val_b = outer[held_batch & held_ligand & outer.celltype.isin(fit.celltype)]
    common_b = val_b.groupby("celltype").split_ligand.nunique().eq(2).any()
    if fit.empty or val_a.empty or not common_b:
        raise ValueError(f"Insufficient inner A/B coverage for fold {fold}")
    return fit, val_a, val_b


def permutation(fit, shuffle):
    if not shuffle:
        return {}
    # One fixed-seed cycle among observed training reagents keeps aliases together.
    label = fit.groupby("split_ligand").cyt_display.min()
    if len(label) < 2:
        raise ValueError("A protein mismatch control needs at least two training ligands")
    order = np.random.default_rng(SEED).permutation(label.index)
    return {str(a): str(label[b]) for a, b in zip(order, np.roll(order, -1))}


def weights(rows, group_by=CONDITION):
    sizes = rows.groupby(group_by).target_index.transform("size").to_numpy()
    n_groups = len(rows[group_by].drop_duplicates())
    return (len(rows) / (n_groups * sizes)).astype(np.float32)


def validation_set(dataset, rows, truth, shuffle_map, device):
    # One held batch gives each condition one common PBS and one reagent input.
    # Average mice first; every validation condition then contributes one MAE.
    representatives, targets = [], []
    for _, mice in rows.groupby(CONDITION, sort=False):
        if mice.context_index.nunique() != 1:
            raise ValueError("Validation mice in one condition do not share a PBS context")
        representatives.append(mice.index[0])
        targets.append(truth[mice.target_index].mean(axis=0))
    cell, ligand = dataset.inputs(rows.loc[representatives], shuffle_map, device)
    return cell, ligand, torch.as_tensor(np.stack(targets), device=device)


def fit_model(dataset, fit, truth, epochs, device, batch_size, shuffle, val=(), *,
              seed=SEED, observe_epochs=(), observe=None, weight_by=CONDITION, extra_loss=None, prediction_loss=None):
    torch.manual_seed(seed)
    model = CytokineFiLM(len(dataset.genes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
    rng = torch.Generator().manual_seed(seed)
    shuffle_map = permutation(fit, shuffle)
    x, e = dataset.inputs(fit, shuffle_map, device)
    y = torch.as_tensor(truth[fit.target_index], device=device)
    w = torch.as_tensor(weights(fit, weight_by), device=device)
    validation = [validation_set(dataset, rows, truth, shuffle_map, device) for rows in val]
    history = []
    # Use the same weight per mouse throughout an epoch, including its last batch.
    step_size = min(batch_size, len(fit))
    for epoch in range(1, epochs + 1):
        model.train()
        for ids in torch.randperm(len(fit), generator=rng).split(step_size):
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[ids], e[ids])
            residual = prediction - y[ids]
            per_mouse = residual.square().mean(dim=1)
            loss = (per_mouse * w[ids]).sum() / step_size
            if extra_loss is not None:
                loss = loss + extra_loss(residual, ids)
            if prediction_loss is not None:
                loss = loss + prediction_loss(prediction, ids)
            loss.backward()
            optimizer.step()
        if validation and epoch in EPOCHS:
            model.eval()
            with torch.inference_mode():
                a, b = [(model(z, s) - target).abs().mean().item() for z, s, target in validation]
            if not np.isfinite([a, b]).all():
                raise ValueError(f"Non-finite inner validation error at epoch {epoch}")
            history.append({"epoch": epoch, "mae_A": a, "mae_B": b, "mean_MAE": (a + b) / 2})
        if observe is not None and epoch in observe_epochs:
            model.eval()
            with torch.inference_mode():
                observe(epoch, model)
    return model.eval(), shuffle_map, history


def predict_oof(dataset, rows, model, shuffle_map, destination, device):
    cell, ligand = dataset.inputs(rows, shuffle_map, device)
    baseline = torch.as_tensor(dataset.baseline[rows.context_index], device=device)
    delta, perturbed = model.predict(cell, ligand, baseline)
    np.save(destination / "delta.npy", delta.cpu().numpy())
    np.save(destination / "perturbed.npy", perturbed.cpu().numpy())
    rows.to_csv(destination / "predictions.tsv", sep="\t", index=False)


def fold_id(fold):
    return tuple(int(getattr(fold, k)) for k in FOLD)


def fold_directory(out, fold):
    b, u, v = fold_id(fold)
    return out / f"b{b}_C{u}_P{v}"


def fold_layout(fold):
    return {k: getattr(fold, k) for k in FOLD + ["val_batch", "val_celltype", "val_ligand_1", "val_ligand_2"]}


def complete_folds(dataset):
    keys = set(dataset.folds[FOLD].itertuples(index=False, name=None))
    expected = set(product(dataset.examples.processing_batch.unique(), (1, 2, 3), (1, 2, 3)))
    if keys != expected or len(keys) != len(dataset.folds):
        raise ValueError("Incomplete or duplicated batch × cell-group × ligand-group head definitions")
    return dataset.folds.itertuples(index=False)


def target_id(dataset):
    manifest = dataset.folder / "dataset.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["target_values_sha256"]
    return str((dataset.folder / "targets.npy").stat().st_mtime_ns)


def training_truth(dataset):
    truth = np.load(dataset.folder / "targets.npy", mmap_mode="r")
    if (truth.ndim != 2 or truth.shape[1] != len(dataset.genes) or truth.dtype != np.float32
            or dataset.examples.target_index.max() >= len(truth)):
        raise ValueError("Target rows, dtype or gene axis do not match the dataset")
    return truth


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CytokineFiLM(len(checkpoint["gene_id"]))
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model.to(device).eval()


def run_head(dataset, fold, args, truth=None):
    directory = fold_directory(args.out, fold)
    outer = outer_fit(dataset.examples, fold)
    routed = routed_rows(dataset.examples, outer, fold)
    layout = fold_layout(fold)
    if args.command == "train":
        settings = {"batch_size": args.batch_size, "shuffle_ligands": args.shuffle_ligands,
                    "target_id": args.target_id}
    if directory.exists():
        checkpoint, model = load_checkpoint(directory / "checkpoint.pt", args.device)
        if checkpoint["input_id"] != dataset.input_id or checkpoint["fold"] != layout:
            raise ValueError(f"{directory} was fitted using different data or fold definitions")
        if args.command == "train":
            if checkpoint["settings"] != settings:
                raise ValueError(f"{directory} already contains a different training run")
            if all((directory / name).is_file() for name in ("delta.npy", "perturbed.npy", "predictions.tsv")):
                print(f"{directory.name} already fitted", flush=True)
                return
        with TemporaryDirectory(dir=directory) as tmp:
            stage = Path(tmp)
            predict_oof(dataset, routed, model, checkpoint["shuffle_map"], stage, args.device)
            for name in ("delta.npy", "perturbed.npy", "predictions.tsv"):
                os.replace(stage / name, directory / name)
        print(f"{directory.name} reloaded, {len(routed)} predictions", flush=True)
        return
    if args.command == "predict":
        raise FileNotFoundError(directory / "checkpoint.pt")
    inner, val_a, val_b = inner_split(outer, fold)
    _, _, history = fit_model(dataset, inner, truth, max(EPOCHS), args.device,
                              args.batch_size, args.shuffle_ligands, (val_a, val_b))
    chosen = min(history, key=lambda x: x["mean_MAE"])["epoch"]
    model, shuffle_map, _ = fit_model(dataset, outer, truth, chosen, args.device,
                                      args.batch_size, args.shuffle_ligands)
    checkpoint = {"state_dict": model.cpu().state_dict(), "gene_id": dataset.genes,
                  "fold": layout, "input_id": dataset.input_id, "settings": settings,
                  "seed": SEED, "selected_epoch": chosen, "shuffle_map": shuffle_map,
                  "outer_fit_target_indices": outer.target_index.tolist(),
                  "inner_fit_target_indices": inner.target_index.tolist(),
                  "val_A_target_indices": val_a.target_index.tolist(),
                  "val_B_target_indices": val_b.target_index.tolist()}
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp) / "head"
        stage.mkdir()
        torch.save(checkpoint, stage / "checkpoint.pt")
        pd.DataFrame(history).to_csv(stage / "selection.tsv", sep="\t", index=False)
        predict_oof(dataset, routed, model.to(args.device), shuffle_map, stage, args.device)
        stage.rename(directory)
    print(f"{directory.name} selected epoch {chosen} using {len(inner)} inner rows;"
          f" refit on {len(outer)}, wrote {len(routed)} predictions", flush=True)


def refit(dataset, args):
    """Choose training duration from inner validation only, then fit one deployment head."""
    if args.out.resolve().is_relative_to(args.predictions.resolve()):
        raise ValueError("The final model must have its own output directory outside the OOF run")
    settings, sources = None, []
    truth_id = target_id(dataset)
    for fold in complete_folds(dataset):
        directory = fold_directory(args.predictions, fold)
        cp = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        if (cp["input_id"] != dataset.input_id or cp["gene_id"] != dataset.genes or cp["seed"] != SEED
                or cp["fold"] != fold_layout(fold)
                or cp["settings"]["shuffle_ligands"] or cp["settings"]["target_id"] != truth_id):
            raise ValueError(f"{directory}: expected an unshuffled head from this dataset and fold")
        if settings is not None and settings != cp["settings"]:
            raise ValueError("OOF heads used different fitting configurations")
        settings = cp["settings"]
        validation = pd.read_csv(directory / "selection.tsv", sep="\t")
        if (tuple(validation.epoch) != EPOCHS or not np.isfinite(validation[["mae_A", "mae_B"]]).all().all()
                or not np.allclose(validation.mean_MAE, (validation.mae_A + validation.mae_B) / 2)):
            raise ValueError(f"{directory}: invalid inner-validation selection table")
        chosen = int(validation.loc[validation.mean_MAE.idxmin(), "epoch"])
        if chosen != cp["selected_epoch"]:
            raise ValueError(f"{directory}: checkpoint epoch disagrees with inner validation")
        sources.append({**dict(zip(FOLD, fold_id(fold))), "head": directory.name, "selected_epoch": chosen})
    chosen = int(np.median([s["selected_epoch"] for s in sources]))
    info = dict(n_rows=len(dataset.examples), n_conditions=len(dataset.examples[CONDITION].drop_duplicates()),
                n_source_heads=len(sources), selected_epoch=chosen, seed=SEED,
                epoch_rule="Median of standard OOF heads' inner-validation selected epochs",
                input_id=dataset.input_id, settings=settings, optimizer={"name": "AdamW", **ADAMW},
                source_root=str(args.predictions.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "checkpoint.pt"
    if path.exists():
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if any(cp.get(k) != v for k, v in info.items()) or cp.get("source_heads") != sources:
            raise ValueError(f"{path} already contains a different model")
        print(f"{path} already fitted", flush=True)
        return
    model, _, _ = fit_model(dataset, dataset.examples, training_truth(dataset), chosen, args.device,
                            settings["batch_size"], False)
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp)
        pd.DataFrame(sources).to_csv(stage / "source_epochs.tsv", sep="\t", index=False)
        (stage / "refit.json").write_text(json.dumps(info, indent=2) + "\n")
        torch.save({**info, "source_heads": sources, "gene_id": dataset.genes,
                    "state_dict": model.cpu().state_dict()}, stage / "checkpoint.pt")
        for name in ("source_epochs.tsv", "refit.json", "checkpoint.pt"):
            os.replace(stage / name, args.out / name)
    print(f"Refit on {len(dataset.examples)} rows for {chosen} epochs, wrote {path}", flush=True)


def infer(args):
    checkpoint, model = load_checkpoint(args.checkpoint, args.device)
    if checkpoint["settings"]["shuffle_ligands"]:
        raise ValueError("Use the named-ligand 'predict' command for the mismatch control")
    with np.load(args.inputs) as data:
        if data["gene_id"].astype(str).tolist() != checkpoint["gene_id"]:
            raise ValueError("PBS baseline gene_id must exactly match checkpoint gene order")
        cell, ligand, baseline = [torch.as_tensor(data[k], device=args.device, dtype=torch.float32)
                                  for k in ("embeddings", "ligand_embeddings", "baseline")]
        if cell.ndim != 2 or cell.shape != (len(baseline), 768) or ligand.shape != (len(baseline), 1280):
            raise ValueError("Expected paired [N,768] GeneCompass and [N,1280] ESM2 embeddings")
        delta, perturbed = model.predict(cell, ligand, baseline)
        ids = {"sample_id": data["sample_id"]} if "sample_id" in data else {}
        if ids and ids["sample_id"].shape != (len(cell),):
            raise ValueError("sample_id must identify each input row in order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        path = Path(tmp) / "predictions.npz"
        np.savez(path, delta=delta.cpu().numpy(), perturbed=perturbed.cpu().numpy(),
                 gene_id=np.asarray(checkpoint["gene_id"]), **ids)
        os.replace(path, args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for mode in ("train", "predict", "infer", "refit", "mouse-donor", "cd8-pair-loss", "cd8-retrieval"):
        p = commands.add_parser(mode)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=4, help="CPU threads used by PyTorch")
        if mode == "infer":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--inputs", type=Path, required=True,
                           help="NPZ containing embeddings, ligand_embeddings, baseline, gene_id")
            p.add_argument("--output", type=Path, required=True)
            continue
        p.add_argument("--dataset", type=Path, default=DATASET)
        if mode == "cd8-retrieval":
            p.description = "Compare CD8 FiLM loss functions with cross-batch cytokine retrieval"
            p.set_defaults(threads=3)
            p.add_argument("--out", type=Path, default=ROOT / "results/GSE202186_CD8_retrieval")
            continue
        if mode == "cd8-pair-loss":
            p.description = "Compare CD8-only FiLM with and without a within-batch cytokine contrast loss"
            p.add_argument("--out", type=Path, default=ROOT / "results/GSE202186_CD8_pair_loss")
            continue
        if mode == "mouse-donor":
            p.description = "Five mouse- and PBS-batch-disjoint splits with measured Top-20 DE scoring"
            p.add_argument("--out", type=Path, default=ROOT / "results/GSE202186_PULSAR_mouse_donor")
            p.add_argument("--batch-size", type=int, default=128)
            p.add_argument("--edger", type=Path, help="Optional three-mouse edgeR reference for novel cytokines")
            continue
        if mode == "refit":
            p.add_argument("--predictions", type=Path, default=RESULTS, help="Standard OOF head directories")
            p.add_argument("--out", type=Path, default=RESULTS.parent / "final_model")
            continue
        p.add_argument("--out", type=Path, default=RESULTS)
        subset = p.add_mutually_exclusive_group(required=True)
        subset.add_argument("--fold", type=int, nargs=3, metavar=("B", "C", "P"))
        subset.add_argument("--all", action="store_true", help="All available heads in folds.tsv")
        if mode == "train":
            p.add_argument("--batch-size", type=int, default=128)
            p.add_argument("--shuffle-ligands", action="store_true", help="Training-ligand mismatch control")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    if args.command == "infer":
        infer(args)
        return
    dataset = Dataset(args.dataset, load_folds=args.command not in ("mouse-donor", "cd8-pair-loss", "cd8-retrieval"))
    if args.command == "cd8-retrieval":
        if __package__:
            from .cd8_retrieval_experiment import run
        else:
            from cd8_retrieval_experiment import run
        run(dataset, args)
        return
    if args.command == "cd8-pair-loss":
        if __package__:
            from .cd8_pair_loss import run
        else:
            from cd8_pair_loss import run
        run(dataset, args)
        return
    if args.command == "mouse-donor":
        if __package__:
            from .mouse_donor import run
        else:
            from mouse_donor import run
        run(dataset, args)
        return
    if args.command == "refit":
        refit(dataset, args)
        return
    folds = dataset.folds
    if folds.duplicated(FOLD).any():
        raise ValueError("folds.tsv contains duplicate head definitions")
    if args.fold is not None:
        folds = folds[(folds[FOLD] == args.fold).all(axis=1)]
        if folds.empty:
            raise ValueError(f"Unknown fold {args.fold}")
    truth = None
    if args.command == "train":
        if args.batch_size < 1:
            raise ValueError("batch_size must be positive")
        truth = training_truth(dataset)
        args.target_id = target_id(dataset)
        args.out.mkdir(parents=True, exist_ok=True)
    for fold in folds.itertuples(index=False):
        if pd.isna(fold.val_batch):
            if args.fold is not None:
                raise ValueError(f"Insufficient inner coverage for fold {fold_id(fold)}")
            print(f"Skipping fold {fold_id(fold)} marked without inner coverage", flush=True)
            continue
        run_head(dataset, fold, args, truth)


if __name__ == "__main__":
    main()
