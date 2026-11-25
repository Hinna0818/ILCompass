"""Assemble the mouse cytokine dictionary and freeze its nested out-of-batch splits."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

if __package__:
    from .splits import outer_fit, routed_rows
else:
    from splits import outer_fit, routed_rows


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/processed/GSE202186_perturbation_v1"
MIN_CELLS = 20
CELL_GROUPS = (
    ("T_cell_CD8", "Treg", "NK_cell", "cDC1", "Lymphocyte_proliferating", "Neutrophil", "Plasma_cell"),
    ("MigDC", "B_cell", "Macrophage", "Langerhans", "ILC", "eTAC", "Basophil"),
    ("T_cell_CD4", "T_cell_gd", "cDC2", "pDC", "Monocyte", "Mast_cell", "B_cell_GC"),
)
CELL_TO_GROUP = {c: i for i, group in enumerate(CELL_GROUPS, 1) for c in group}
MATCH = ["processing_batch", "celltype"]
CONDITION = ["celltype", "cyt_display"]


def ligand_groups(mapping):
    """Place aliases and identical model inputs in the earliest original label's group."""
    labels = sorted(mapping.cyt_display)
    parent = {p: p for p in labels}

    def find(p):
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    resolved = mapping[mapping.status != "unresolved"]
    for key in ["canonical_id", "input_key"]:
        if resolved[key].eq("").any():
            raise ValueError(f"A resolved ligand is missing {key}.")
        for _, rows in resolved.groupby(key):
            for name in rows.cyt_display:
                a, b = find(rows.cyt_display.iloc[0]), find(name)
                parent[max(a, b)] = min(a, b)
    original = {p: i % 3 + 1 for i, p in enumerate(labels)}
    out = mapping.copy()
    out["split_ligand"] = out.cyt_display.map(find)
    out["ligand_group"] = out.split_ligand.map(original)
    return out


def choose_inner(allowed, outer_batch, batches):
    for offset in range(1, len(batches)):
        vb = batches[(batches.index(outer_batch) + offset) % len(batches)]
        val = {(c, p) for b, c, p in allowed if b == vb}
        others = {(c, p) for b, c, p in allowed if b != vb}
        for vc in sorted({c for c, p in val}):
            rest = {(c, p) for c, p in others if c != vc}
            # For B, both held-out ligands must be assayed in a common known type.
            for l1, l2 in combinations(sorted({p for c, p in val}), 2):
                train = {(c, p) for c, p in rest if p not in (l1, l2)}
                seen_p = {p for c, p in train}
                if not any(c == vc and p in seen_p for c, p in val):
                    continue
                if any((c, l1) in val and (c, l2) in val for c in {c for c, p in train}):
                    return vb, vc, l1, l2
    raise ValueError(f"No eligible two-task inner validation split for batch {outer_batch}.")


def freeze_splits(examples, batches):
    batches = sorted(batches)
    heads = []
    routes = {"A": set(), "B": set()}
    for b in batches:
        for u in (1, 2, 3):
            for v in (1, 2, 3):
                fold = SimpleNamespace(batch=b, hold_cell_group=u, hold_ligand_group=v)
                allowed = outer_fit(examples, fold)
                keys = ["processing_batch", "celltype", "split_ligand"]
                vb, vc, p1, p2 = choose_inner(set(allowed[keys].itertuples(index=False, name=None)), b, batches)
                heads.append((b, u, v, vb, vc, p1, p2))
                for task, part in routed_rows(examples, allowed, fold).groupby("task"):
                    ids = set(part.target_index)
                    if ids & routes[task]:
                        raise ValueError(f"A mouse/type was routed to two {task} heads.")
                    routes[task].update(ids)
    for task, ids in routes.items():
        examples[f"eligible_{task}"] = examples.target_index.isin(ids)
    return pd.DataFrame(heads, columns=["batch", "hold_cell_group", "hold_ligand_group",
                                        "val_batch", "val_celltype", "val_ligand_1", "val_ligand_2"])


def logcpm(counts):
    sizes = np.asarray(counts.sum(axis=1)).ravel()
    if (sizes <= 0).any():
        raise ValueError("Empty pseudobulk among the selected samples.")
    return np.log2(1 + counts.toarray() * (1e6 / sizes)[:, None]).astype(np.float32)


def write_dataset(data, output):
    pb = ad.read_h5ad(data / "rna/pseudobulk_counts.h5ad")
    obs = pb.obs.rename_axis("sample_id").reset_index()
    for key in ["sample_id", "cyt_display", "rep", "celltype"]:
        obs[key] = obs[key].astype(str)
    if not pb.var_names.is_unique or not pb.obs_names.is_unique or not np.issubdtype(pb.X.dtype, np.integer):
        raise ValueError("Expected unique gene and mouse/type IDs and integer UMI counts.")
    if set(obs.celltype) != set(CELL_TO_GROUP):
        raise ValueError("Author cell labels differ from the prespecified C1–C3 groups.")
    if obs.groupby(["cyt_display", "rep"]).processing_batch.nunique().max() != 1:
        raise ValueError("A mouse is assigned to multiple processing batches.")
    obs["count_index"] = np.arange(len(obs))
    protein = pd.read_csv(data / "ligands/ligand_map.tsv", sep="\t", keep_default_na=False)
    if protein.cyt_display.duplicated().any() or set(protein.cyt_display) != set(obs.cyt_display) - {"PBS"}:
        raise ValueError("The ligand inventory must include every original treatment exactly once.")
    if not set(protein.status) <= {"reagent_verified", "reference_proxy", "unresolved"}:
        raise ValueError("Unknown ligand evidence status.")
    protein = ligand_groups(protein)
    used = protein[protein.status != "unresolved"]
    with np.load(data / "ligands/esm2.npz") as cache:
        names, emb = cache["cyt_display"].astype(str), cache["embeddings"]
        if set(names) != set(used.cyt_display) or emb.shape != (len(used), 1280) or not np.isfinite(emb).all():
            raise ValueError("Resolved ligand inventory and ESM2 cache disagree.")

    controls = obs[(obs.cyt_display == "PBS") & (obs.n_cells >= MIN_CELLS)].copy()
    with np.load(data / "rna/pbs_genecompass.npz") as cache:
        znames, zs = cache["obs_names"].astype(str), cache["embeddings"]
    if len(np.unique(znames)) != len(znames) or zs.shape != (len(znames), 768) or not np.isfinite(zs).all():
        raise ValueError("Invalid GeneCompass PBS feature matrix.")
    zindices = pd.Index(znames).get_indexer(controls.sample_id)
    if (zindices < 0).any():
        raise ValueError("A qualifying PBS mouse/type is missing its original GeneCompass feature.")
    controls["embedding_index"] = zindices
    contexts, baseline, cell_vectors = [], [], []
    for context_id, ((b, c), rows) in enumerate(controls.groupby(MATCH, sort=True)):
        baseline.append(logcpm(pb.X[rows.count_index.to_numpy()]).mean(axis=0))
        cell_vectors.append(zs[rows.embedding_index.to_numpy()].mean(axis=0))
        contexts.append((context_id, b, c, len(rows), rows.n_cells.sum(), ",".join(rows.sample_id)))
    ctx = pd.DataFrame(contexts, columns=["context_index", *MATCH, "n_pbs_mice", "n_pbs_cells", "pbs_sample_ids"])
    base = np.stack(baseline)
    np.savez(output / "contexts.npz", baseline=base, embeddings=np.stack(cell_vectors).astype(np.float32))
    ctx.to_csv(output / "contexts.tsv", sep="\t", index=False)

    stim = obs[obs.cyt_display != "PBS"]
    good = stim[stim.n_cells >= MIN_CELLS].merge(ctx[MATCH + ["context_index"]], on=MATCH, validate="m:1")
    good = good.merge(protein[["cyt_display", "canonical_id", "split_ligand", "ligand_group", "status"]],
                      on="cyt_display", validate="m:1")
    examples = good[good.status != "unresolved"].sort_values(["processing_batch", *CONDITION, "rep"]).copy()
    examples["cell_group"] = examples.celltype.map(CELL_TO_GROUP)
    examples["n_eligible_mice"] = examples.groupby(CONDITION).rep.transform("size")
    examples["complete_three"] = examples.n_eligible_mice == 3
    examples["target_index"] = np.arange(len(examples))
    folds = freeze_splits(examples, obs.processing_batch.unique().tolist())
    target = np.lib.format.open_memmap(output / "targets.npy", mode="w+", dtype=np.float32,
                                       shape=(len(examples), pb.n_vars))
    target_digest = hashlib.sha256()
    for start in range(0, len(target), 64):
        chunk = examples.iloc[start:start + 64]
        delta = logcpm(pb.X[chunk.count_index.to_numpy()]) - base[chunk.context_index]
        target[start:start + len(chunk)] = delta
        target_digest.update(delta)
    target.flush()
    del target
    examples.to_csv(output / "examples.tsv", sep="\t", index=False)
    folds.to_csv(output / "folds.tsv", sep="\t", index=False)
    protein.to_csv(output / "ligand_groups.tsv", sep="\t", index=False)
    genes = pb.var.copy()
    if "gene_id" in genes:
        if not (genes.gene_id.astype(str).to_numpy() == pb.var_names.to_numpy()).all():
            raise ValueError("Conflicting gene identifiers.")
        genes = genes.drop(columns="gene_id")
    genes.rename_axis("gene_id").to_csv(output / "genes.tsv", sep="\t")

    grid = pd.MultiIndex.from_product([sorted(CELL_TO_GROUP), sorted(protein.cyt_display)], names=CONDITION)
    coverage = pd.DataFrame(index=grid)
    for name, frame in [("nonempty_mice", stim), ("mice_with_20_cells", stim[stim.n_cells >= MIN_CELLS]),
                        ("mice_with_matching_pbs", good), ("model_mice", examples),
                        ("task_A_mice", examples[examples.eligible_A]), ("task_B_mice", examples[examples.eligible_B])]:
        coverage[name] = frame.groupby(CONDITION).size().reindex(grid, fill_value=0)
    coverage = coverage.reset_index().merge(protein[["cyt_display", "status"]], on="cyt_display")
    coverage["coverage_reason"] = np.select(
        [coverage.status == "unresolved", coverage.mice_with_20_cells == 0,
         coverage.mice_with_matching_pbs == 0, coverage.model_mice < 3],
        ["unresolved_protein", "too_few_stimulated_cells", "no_qualifying_matched_PBS", "one_or_two_eligible_mice"],
        default="three_eligible_mice")
    coverage.to_csv(output / "condition_coverage.tsv", sep="\t", index=False)
    used_ctx = ctx[ctx.context_index.isin(examples.context_index.unique())]
    summary = {
        "accession": "GSE202186", "author_annotations": "SCP2554", "mouse_id_columns": ["cyt_display", "rep"],
        "n_genes": pb.n_vars, "output_unit": "log2(CPM + 1), full-gene library per mouse and cell type",
        "target": "treated mouse log expression minus equal-mouse same-batch same-type qualifying PBS mean",
        "target_values_sha256": target_digest.hexdigest(),
        "min_cells_per_mouse_type": MIN_CELLS, "all_immune_cells": int(obs.n_cells.sum()), "n_nonempty_mouse_types": len(obs),
        "n_mice": len(obs[["cyt_display", "rep"]].drop_duplicates()), "n_batches": int(obs.processing_batch.nunique()),
        "author_types": len(CELL_TO_GROUP), "original_ligands": len(protein), "encoded_ligands": len(used),
        "ligand_evidence": {k: int(v) for k, v in protein.status.value_counts().items()},
        "all_possible_conditions": len(grid), "observed_before_protein_filter": int((coverage.mice_with_matching_pbs > 0).sum()),
        "eligible_mouse_types_before_protein_filter": len(good), "model_records": len(examples),
        "stimulated_mice_in_model": len(examples[["cyt_display", "rep"]].drop_duplicates()),
        "stimulated_cells_in_model": int(examples.n_cells.sum()), "pbs_cells_in_model": int(used_ctx.n_pbs_cells.sum()),
        "contexts_in_model": len(used_ctx), "pbs_contexts_in_cache": len(ctx),
        "model_conditions": int((coverage.model_mice > 0).sum()), "three_mouse_conditions": int((coverage.model_mice == 3).sum()),
        "types_in_model": int(examples.celltype.nunique()),
        "complete_types": int(coverage.loc[coverage.model_mice == 3, "celltype"].nunique()),
        "coverage_by_eligible_mice": {str(k): int(v) for k, v in coverage.model_mice.value_counts().sort_index().items()},
        "n_outer_heads": len(folds), "task_A_eligible_rows": int(examples.eligible_A.sum()),
        "task_B_eligible_rows": int(examples.eligible_B.sum()),
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    data = parser.parse_args().dataset
    with TemporaryDirectory(dir=data, prefix=".assembling-") as temp:
        stage = Path(temp)
        summary = write_dataset(data, stage)
        # Publish the manifest last so partially constructed targets cannot be mistaken for a dataset.
        for path in sorted(stage.iterdir(), key=lambda p: p.name == "dataset.json"):
            path.replace(data / path.name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
