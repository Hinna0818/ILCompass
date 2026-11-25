"""Scientific boundaries for the dictionary FiLM, using small synthetic gene targets."""

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from immune_dictionary import train
from immune_dictionary.film import CytokineFiLM


def make_data(folder):
    rng = np.random.default_rng(9)
    celltypes = [f"c{i}{j}" for i in (1, 2, 3) for j in ("a", "b")]
    ligands = [f"p{i}{j}" for i in (1, 2, 3) for j in ("a", "b")] + ["alias2a"]
    rows, contexts = [], []
    for batch in (1, 2, 3):
        for cell in celltypes:
            context = len(contexts)
            contexts.append(rng.normal(size=768))
            for ligand in ligands:
                key = "p2a" if ligand == "alias2a" else ligand
                rows.append(dict(sample_id=f"{batch}_{ligand}", rep=f"rep0{batch}", celltype=cell,
                                 cyt_display=ligand, canonical_id=key, split_ligand=key, n_cells=25,
                                 processing_batch=batch, cell_group=int(cell[1]), ligand_group=int(key[1]),
                                 target_index=len(rows), context_index=context, complete_three=True))
    n_genes = 7
    proteins = rng.normal(size=(len(ligands), 1280)).astype("float32")
    proteins[-1] = proteins[ligands.index("p2a")]
    (folder / "ligands").mkdir()
    np.savez(folder / "ligands/esm2.npz", cyt_display=ligands, embeddings=proteins)
    np.savez(folder / "contexts.npz", embeddings=np.asarray(contexts, dtype="float32"),
             baseline=rng.uniform(size=(len(contexts), n_genes)).astype("float32"))
    pd.DataFrame({"gene_id": [f"ENSMUSG{i}" for i in range(n_genes)]}).to_csv(
        folder / "genes.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(folder / "examples.tsv", sep="\t", index=False)
    np.save(folder / "targets.npy", rng.normal(size=(len(rows), n_genes)).astype("float32"))
    pd.DataFrame([[1, 1, 1, 2, "c2a", "p2a", "p2b"]], columns=train.FOLD + [
        "val_batch", "val_celltype", "val_ligand_1", "val_ligand_2"]).to_csv(
            folder / "folds.tsv", sep="\t", index=False)


class DictionaryFiLMTests(unittest.TestCase):
    def test_routing_and_mouse_weights(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_data(folder)
            ds = train.Dataset(folder)
            fold = next(ds.folds.itertuples(index=False))
            outer = train.outer_fit(ds.examples, fold)
            inner, a, b = train.inner_split(outer, fold)
            self.assertFalse((outer.processing_batch.eq(1) | outer.cell_group.eq(1)
                              | outer.ligand_group.eq(1)).any())
            self.assertTrue(set(inner.sample_id).isdisjoint(set(a.sample_id) | set(b.sample_id)))
            self.assertTrue(set(inner.context_index).isdisjoint(set(a.context_index) | set(b.context_index)))
            self.assertFalse((inner.processing_batch.eq(2) | inner.celltype.eq("c2a")
                              | inner.cyt_display.isin(["p2a", "alias2a", "p2b"])).any())
            self.assertIn("alias2a", set(b.cyt_display))
            mapping = train.permutation(outer, True)
            self.assertEqual(set(mapping), set(outer.split_ligand))
            self.assertTrue(set(mapping.values()) <= set(outer.cyt_display))
            self.assertNotIn("alias2a", train.permutation(inner, True))
            predictions = []
            for u in (1, 2, 3):
                for v in (1, 2, 3):
                    head = fold._replace(hold_cell_group=u, hold_ligand_group=v)
                    fit = train.outer_fit(ds.examples, head)
                    pred = train.routed_rows(ds.examples, fit, head)
                    self.assertTrue(set(pred.context_index).isdisjoint(fit.context_index))
                    self.assertTrue(set(pred.sample_id).isdisjoint(fit.sample_id))
                    predictions.append(pred)
            coverage = pd.concat(predictions).groupby(["task", "target_index"]).size()
            self.assertEqual(len(coverage), 2 * ds.examples.processing_batch.eq(1).sum())
            self.assertTrue(coverage.eq(1).all())
            uneven = outer.iloc[3:].copy()
            uneven["weight"] = train.weights(uneven)
            by_condition = uneven.groupby(train.CONDITION).weight
            np.testing.assert_allclose(by_condition.sum(), len(uneven) / by_condition.ngroups)
            self.assertTrue(by_condition.nunique().eq(1).all())

    def test_predict_after_truth_removed_and_gene_order(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            out = folder / "out"
            make_data(folder)
            args = ["--dataset", str(folder), "--out", str(out), "--fold", "1", "1", "1", "--threads", "1"]
            train.main(["train", *args])
            head = out / "b1_C1_P1"
            self.assertEqual(pd.read_csv(head / "selection.tsv", sep="\t").epoch.tolist(), [10, 30, 60])
            predictions = pd.read_csv(head / "predictions.tsv", sep="\t")
            with np.load(folder / "contexts.npz") as contexts:
                features = contexts["embeddings"][predictions.context_index]
                baseline = contexts["baseline"][predictions.context_index]
            with np.load(folder / "ligands/esm2.npz") as ligands:
                lookup = dict(zip(ligands["cyt_display"], ligands["embeddings"]))
            protein = np.stack([lookup[label] for label in predictions.cyt_display])
            checkpoint, _ = train.load_checkpoint(head / "checkpoint.pt", "cpu")
            expected_delta = np.load(head / "delta.npy")
            self.assertTrue(np.isfinite(expected_delta).all())
            np.testing.assert_equal(np.load(head / "perturbed.npy"), baseline + expected_delta)
            self.assertFalse(set(checkpoint["outer_fit_target_indices"]) & set(predictions.target_index))
            (folder / "targets.npy").rename(folder / "truth_hidden.npy")
            (head / "delta.npy").unlink()
            train.main(["predict", *args])
            np.testing.assert_equal(np.load(head / "delta.npy"), expected_delta)
            data = dict(embeddings=features, ligand_embeddings=protein, baseline=baseline,
                        gene_id=checkpoint["gene_id"])
            np.savez(folder / "new.npz", **data)
            inference = ["infer", "--checkpoint", str(head / "checkpoint.pt"), "--inputs",
                         str(folder / "new.npz"), "--output", str(folder / "result.npz"), "--threads", "1"]
            train.main(inference)
            with np.load(folder / "result.npz") as new:
                np.testing.assert_equal(new["delta"], expected_delta)
            data["gene_id"] = list(reversed(data["gene_id"]))
            np.savez(folder / "new.npz", **data)
            with self.assertRaisesRegex(ValueError, "gene order"):
                train.main(inference)

    def test_zero_cytokine_returns_pbs_exactly(self):
        model = CytokineFiLM(7)
        baseline = torch.rand(3, 7)
        delta, final = model.predict(torch.randn(3, 768), torch.zeros(3, 1280), baseline)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(final, baseline))


if __name__ == "__main__":
    unittest.main()
