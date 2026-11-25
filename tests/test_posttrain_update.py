"""只测试会影响数据角色、loss 权重及验证可比性的边界。"""
from pathlib import Path
import sys
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posttrain_data import SPLITS, assign_splits, encode
from posttrain_training import aggregate, batches, step_optimizer, training_loss, save_training_state, restore_training_state


class PosttrainUpdateTests(unittest.TestCase):
    def test_resume_restores_adam_schedule_rng_and_exact_next_update(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: 1 / (u + 1))
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        def update():
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(4, 3)).square().sum().backward()
            optimizer.step()
            scheduler.step()

        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        update()
        progress = {"update": 1, "next_micro": 4}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            save_training_state(path, model, optimizer, scheduler, scaler, progress)
            expected_random = (random.random(), np.random.random())
            update()
            expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
            expected_rate = scheduler.get_last_lr()
            state = torch.load(path, weights_only=False)
            self.assertEqual(restore_training_state(state, model, optimizer, scheduler, scaler), progress)
            self.assertEqual((random.random(), np.random.random()), expected_random)
            update()
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, expected_weights[k]))
            self.assertEqual(scheduler.get_last_lr(), expected_rate)

    def test_resume_batch_position_keeps_shuffle_and_masks(self):
        ds = Dataset.from_dict({"input_ids": [[i] * 10 for i in range(10)],
                               "values": [[1.] * 10] * 10, "species": [[1]] * 10})
        def collate(ex):
            return {"input_ids": torch.tensor([e["input_ids"] for e in ex]),
                    "labels": torch.where(torch.rand(len(ex), 10) > .5, 1, -100)}
        kwargs = dict(ds=ds, idx=np.arange(10), collator=collate, batch_size=3,
                      seed=19, device=torch.device("cpu"), shuffle=True)
        full = list(batches(**kwargs))
        resumed = list(batches(**kwargs, start_batch=2))
        self.assertEqual(len(resumed), len(full) - 2)
        for actual, expected in zip(resumed, full[2:]):
            for key in actual:
                self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_amp_overflow_skips_then_recovers_without_advancing_schedule(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([p], lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536.)
        initial_scheduler_step = scheduler.last_epoch
        # FP32 参数经过 half 运算反传：scale=65536 使 half 梯度溢出，但前向 loss 有限。
        scaler.scale(p.to(torch.float16).float()).backward()
        step = step_optimizer(optimizer, scaler, [p], 1.)
        if step["stepped"]:
            scheduler.step()
        self.assertFalse(step["stepped"])
        self.assertEqual(p.item(), 1.)
        self.assertEqual(scaler.get_scale(), 32768.)
        self.assertEqual(scheduler.last_epoch, initial_scheduler_step)
        self.assertIsNone(p.grad)
        scaler.scale(p.to(torch.float16).float()).backward()
        step = step_optimizer(optimizer, scaler, [p], 1.)
        if step["stepped"]:
            scheduler.step()
        self.assertTrue(step["stepped"])
        self.assertAlmostEqual(p.item(), .9, places=6)
        self.assertEqual(scheduler.last_epoch, initial_scheduler_step + 1)

    def test_non_amp_nonfinite_gradient_stops_without_parameter_update(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([p], lr=.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        p.grad = torch.tensor(float("inf"))
        with self.assertRaises(FloatingPointError):
            step_optimizer(optimizer, scaler, [p], 1.)
        self.assertEqual(p.item(), 1.)

    def test_split_small_library_and_input_order(self):
        obs = pd.DataFrame({"split_stratum": ["small"] * 136 + ["other"] * 1000},
                           index=[f"cell{i:04d}" for i in range(1136)])
        split = assign_splits(obs, [.9, .05, .05], 0)
        self.assertEqual(split.iloc[:136].value_counts().to_dict(), {SPLITS[0]: 122, SPLITS[1]: 7, SPLITS[2]: 7})
        reordered = obs.sample(frac=1, random_state=3)
        self.assertTrue(split.eq(assign_splits(reordered, [.9, .05, .05], 0).reindex(obs.index)).all())
        with self.assertRaises(ValueError):
            assign_splits(obs.iloc[:1], [.9, .05, .05], 0)

    def test_encoding_raw_counts_unchanged_and_padding(self):
        x = sparse.csr_matrix([[0, 4, 8, 100]])
        before = x.copy()
        ids, vals, length = encode(x, np.array([1, 2, 3, -1]), np.array([1., 2., 8., 1.]), 4)
        np.testing.assert_array_equal(ids, [[2, 3, 0, 0]])
        np.testing.assert_allclose(vals, [[np.log2(3), 1., 0., 0.]])
        self.assertEqual(length.tolist(), [2])
        self.assertEqual((x != before).nnz, 0)

    def test_fixed_normalization_preserves_gradient(self):
        out = SimpleNamespace(id_loss=torch.tensor(10., requires_grad=True), value_loss=torch.tensor(.01, requires_grad=True))
        cfg = {"loss_mode": "normalized", "loss_weights": {"id": .5, "expression": .5}}
        loss = training_loss(out, cfg, {"id": 10., "expression": .01})
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        self.assertAlmostEqual(out.id_loss.grad.item(), .05, places=6)
        self.assertAlmostEqual(out.value_loss.grad.item(), 50., places=4)

    def test_metric_weight_uses_masked_positions(self):
        rows = [{"id": 10., "expression": 1., "top1": 0., "masked_positions": 1, "cells": 1},
                {"id": 2., "expression": .1, "top1": 1., "masked_positions": 3, "cells": 1}]
        result = aggregate(rows, {"id": 10., "expression": 1.})
        self.assertAlmostEqual(result["id"], 4.)
        self.assertAlmostEqual(result["top1"], .75)

    def test_mask_rng_does_not_reset_training_rng(self):
        ds = Dataset.from_dict({"input_ids": [[1] * 10], "values": [[1.] * 10], "species": [[1]]})
        def collate(ex):
            return {"labels": torch.where(torch.rand(1, 10) > .5, 1, -100)}
        torch.manual_seed(99)
        state = torch.random.get_rng_state().clone()
        first = next(batches(ds, [0], collate, 1, 12345, torch.device("cpu")))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = next(batches(ds, [0], collate, 1, 12345, torch.device("cpu")))
        self.assertTrue(torch.equal(first["labels"], second["labels"]))


if __name__ == "__main__":
    unittest.main()
