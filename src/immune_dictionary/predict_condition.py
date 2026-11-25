"""Predict a labeled 31,053-gene profile from a dictionary PBS context and cytokine."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

if __package__:
    from .train import DATASET, RESULTS, Dataset, load_checkpoint
else:
    from train import DATASET, RESULTS, Dataset, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint", type=Path, default=RESULTS.parent / "final_model/checkpoint.pt")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--cytokine", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output .tsv or .tsv.gz")
    args = parser.parse_args()
    ds = Dataset(args.dataset)
    checkpoint, model = load_checkpoint(args.checkpoint, "cpu")
    if checkpoint["input_id"] != ds.input_id or checkpoint["gene_id"] != ds.genes:
        raise ValueError("Checkpoint and dictionary axes or encoders differ.")
    if checkpoint["settings"]["shuffle_ligands"]:
        raise ValueError("Use a standard sequence-matched checkpoint for a biological prediction.")
    contexts = pd.read_csv(args.dataset / "contexts.tsv", sep="\t").set_index(["processing_batch", "celltype"])
    if (args.batch, args.celltype) not in contexts.index:
        raise ValueError(f"No PBS mouse has sufficient {args.celltype} cells in batch {args.batch}.")
    i = contexts.at[(args.batch, args.celltype), "context_index"]
    if args.cytokine == "PBS":
        ligand = np.zeros((1, ds.ligand.shape[1]), dtype=np.float32)
    else:
        if args.cytokine not in ds.ligand_index:
            raise ValueError(f"Unknown stimulus {args.cytokine!r}; use a cyt_display from ligands/ligand_map.tsv.")
        ligand = ds.ligand[[ds.ligand_index[args.cytokine]]]
    base = ds.baseline[[i]]
    delta, treated = model.predict(torch.from_numpy(ds.cell[[i]]), torch.from_numpy(ligand), torch.from_numpy(base))
    table = pd.read_csv(args.dataset / "genes.tsv", sep="\t", usecols=["gene_id", "gene_symbol"], keep_default_na=False)
    table["processing_batch"], table["celltype"], table["cyt_display"] = args.batch, args.celltype, args.cytokine
    table["pbs_log2_CPM1"], table["predicted_delta"], table["predicted_perturbed"] = base[0], delta[0].numpy(), treated[0].numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        path = Path(tmp) / args.out.name
        table.to_csv(path, sep="\t", index=False)
        path.replace(args.out)
    print(f"{args.cytokine} / {args.celltype}, batch {args.batch}: {len(table)} genes -> {args.out}")


if __name__ == "__main__":
    main()
