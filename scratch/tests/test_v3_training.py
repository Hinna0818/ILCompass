"""Pre-LN architecture, unchanged loss, PAD handling and v3 checkpoint replay."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model_v3 import V3Config, V3Transformer, PreNormBlock
from model import MODEL_INPUTS, mlm_objective
from train import make_optimizer, make_scheduler, save_checkpoint, load_training_checkpoint, train
from test_model_training import loader, INPUT


def small():
    return V3Transformer(V3Config(n_genes=20, max_length=64, hidden_size=24, num_heads=3,
                                 intermediate_size=48, num_layers=2, dropout=.1))


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pre_norm_zero_residual_is_identity(self):
        cfg = V3Config(n_genes=20, hidden_size=24, num_heads=3, intermediate_size=48, dropout=0)
        block = PreNormBlock(cfg)
        with torch.no_grad():
            for layer in (block.attn_out, block.ffn[-1]):
                layer.weight.zero_(); layer.bias.zero_()
        x = torch.randn(2, 5, 24)
        valid = torch.tensor([[True] * 5, [True, True, False, False, False]])
        torch.testing.assert_close(block(x, valid), x.masked_fill(~valid[..., None], 0), rtol=0, atol=0)

    def test_parameter_count_and_no_auxiliary_heads(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/training_v3.json').read_text())
        with torch.device('meta'):
            model = V3Transformer(V3Config(n_genes=26725, **cfg['model']))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 28077926)
        self.assertEqual((cfg['id_loss_weight'], cfg['value_loss_weight']), (.2, .8))
        self.assertFalse(any('tissue' in key or 'cell_head' in key or 'batch_head' in key for key in model.state_dict()))

    def test_padding_invariance_and_original_joint_loss(self):
        model = small().eval()
        batch = next(iter(loader()))
        inputs = {key: batch[key] for key in MODEL_INPUTS}
        reference = model(**inputs)
        padded = {key: torch.nn.functional.pad(value, (0, 9), value=False if key == 'attention_mask' else 0)
                  for key, value in inputs.items()}
        actual = model(**padded)
        torch.testing.assert_close(reference['cell_embedding'], actual['cell_embedding'], atol=1e-6, rtol=1e-5)
        self.assertTrue((actual['gene_embeddings'][~padded['attention_mask']] == 0).all())
        result = mlm_objective(model, batch)
        loss = (.2 * result['id_sum'] + .8 * result['value_sum']) / result['valid_cells']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_replays_next_stochastic_update(self):
        cfg = dict(learning_rate=2e-4, min_learning_rate=1e-6, weight_decay=.01,
                   adam_betas=[.9, .999], adam_eps=1e-8, warmup_fraction=.05)
        model = small()
        optimizer = make_optimizer(model, cfg); scheduler = make_scheduler(optimizer, cfg, 20)
        batches = list(loader())
        def step(batch):
            optimizer.zero_grad(set_to_none=True)
            result = mlm_objective(model, batch)
            ((.2 * result['id_sum'] + .8 * result['value_sum']) / result['valid_cells']).backward()
            optimizer.step(); scheduler.step()
        step(batches[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            save_checkpoint(path, model, optimizer, scheduler, dict(global_step=1), cfg, INPUT, {'v3': 'fixed'}, {})
            step(batches[1]); expected = copy.deepcopy(model.state_dict())
            load_training_checkpoint(path, model, optimizer, scheduler, cfg, INPUT, {'v3': 'fixed'})
            step(batches[1])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_shared_loop_trains_v3_and_keeps_tail_window(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/training_v3.json').read_text())
        vocab = pd.DataFrame(dict(gene_id=range(20), gene_symbol=range(20),
                                 training_positive_count=1, is_functional67=False))
        with tempfile.TemporaryDirectory() as folder:
            cfg.update(output=folder, epochs=1, accumulation_steps=3, precision='fp32')
            model = small()
            train(model, loader(), loader(), vocab, cfg, INPUT, torch.device('cpu'), {'v3': 'tiny'}, {})
            saved = torch.load(Path(folder) / 'last.pt', weights_only=False)
            self.assertEqual(saved['model_config']['norm_type'], 'pre_ln')
            self.assertEqual(saved['state']['global_step'], 2)
            self.assertTrue(saved['state']['finished'])
            self.assertTrue((Path(folder) / 'best.pt').exists())


if __name__ == '__main__':
    unittest.main()
