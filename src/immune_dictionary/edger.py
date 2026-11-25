"""Score-only edgeR ground truth for eligible three-mouse conditions from original UMI counts."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd

if __package__:
    from .assemble import DEFAULT_DATA, ROOT, CONDITION
else:
    from assemble import DEFAULT_DATA, ROOT, CONDITION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "results/GSE202186_perturbation_model/edger")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    ex = pd.read_csv(args.dataset / "examples.tsv", sep="\t", keep_default_na=False)
    ex = ex[ex.complete_three]
    ctx = pd.read_csv(args.dataset / "contexts.tsv", sep="\t")
    pb = ad.read_h5ad(args.dataset / "rna/pseudobulk_counts.h5ad")
    gene_ids = pd.read_csv(args.dataset / "genes.tsv", sep="\t", usecols=["gene_id"])
    if gene_ids.gene_id.tolist() != pb.var_names.tolist() or not np.issubdtype(pb.X.dtype, np.integer):
        raise ValueError("The DEG gene axis or integer count source disagrees with the training dataset.")
    # Write each source mouse/type once. A PBS mouse spanning multiple GEO channels is already aggregated.
    ids = set(ex.sample_id)
    pbs = ctx.set_index("context_index").pbs_sample_ids.str.split(",")
    for i in ex.context_index.unique():
        ids.update(pbs.at[i])
    all_ids = sorted(ids)
    remap = {s: i + 1 for i, s in enumerate(all_ids)}  # R indices
    indices = pb.obs_names.get_indexer(all_ids)
    if (indices < 0).any():
        raise ValueError("A supervised treatment or matched PBS sample is missing its raw counts.")
    conditions, samples = [], []
    for k, ((ct, ligand), obs) in enumerate(ex.groupby(CONDITION, sort=True)):
        treated = obs.sample_id.tolist()
        controls = sorted({s for i in obs.context_index for s in pbs.at[i]})
        if len(set(treated)) != 3:
            raise ValueError(f"Expected three independent treated mice for {ct}, {ligand}.")
        for label, mice in [("pbs", controls), ("treated", treated)]:
            for s in mice:
                samples.append((k, remap[s], pb.obs.at[s, "processing_batch"], label))
        conditions.append((k, ct, ligand, len(controls)))
    conditions = pd.DataFrame(conditions, columns=["condition_index", *CONDITION, "n_pbs_mice"])
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out, prefix=".edger-") as tmp:
        work = Path(tmp)
        pd.DataFrame(samples, columns=["condition_index", "column", "batch", "condition"]).to_csv(
            work / "samples.tsv", sep="\t", index=False)
        with (work / "counts.bin").open("wb") as dest:
            for start in range(0, len(indices), 64):
                pb.X[indices[start:start + 64]].toarray().astype("<i4", copy=False).tofile(dest)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run([args.rscript, str(Path(__file__).with_suffix(".R")), str(work), str(pb.n_vars),
                        str(len(all_ids)), str(args.workers)], env=env, check=True)
        codes = np.fromfile(work / "codes.bin", dtype=np.uint8).reshape(len(conditions), pb.n_vars)
        stats = pd.read_csv(work / "design.tsv", sep="\t")
        conditions = conditions.merge(stats, on="condition_index", validate="1:1").sort_values("condition_index")
        conditions["n_tested"] = (codes > 0).sum(axis=1)
        conditions["n_deg"] = (codes >= 2).sum(axis=1)
        conditions.to_csv(work / "conditions.tsv", sep="\t", index=False)
        effects = pd.read_csv(work / "effects.tsv", sep="\t")
        effects = effects.merge(conditions[["condition_index", *CONDITION]], on="condition_index", validate="m:1")
        gene_index = effects.gene_index.to_numpy(dtype=int)
        effects["gene_id"] = pb.var_names.to_numpy()[gene_index]
        effects["gene_symbol"] = pb.var.gene_symbol.to_numpy()[gene_index]
        effects.to_csv(work / "significant_genes.tsv.gz", sep="\t", index=False)
        np.save(work / "gene_codes.npy", codes)
        shutil.copyfile(args.dataset / "genes.tsv", work / "genes.tsv")
        for name in ["gene_codes.npy", "genes.tsv", "session.txt", "significant_genes.tsv.gz", "conditions.tsv"]:
            (work / name).replace(args.out / name)
    print(f"edgeR: {len(conditions)} conditions, {int((conditions.n_deg > 0).sum())} with FDR <= 0.05.")


if __name__ == "__main__":
    main()
