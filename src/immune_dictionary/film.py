"""Conditional mean expression from frozen PBS GeneCompass and pooled ESM2 features."""

import torch
from torch import nn


class CytokineFiLM(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(768, 32), nn.GELU())
        self.ligand = nn.Sequential(nn.Linear(1280, 32, bias=False), nn.Tanh())
        self.gamma = nn.Linear(32, 32, bias=False)
        self.beta = nn.Linear(32, 32, bias=False)
        self.readout = nn.Linear(32, n_genes, bias=False)

    def forward(self, cell: torch.Tensor, ligand: torch.Tensor) -> torch.Tensor:
        """Signed change from PBS in log2(CPM+1) units, on the full gene axis."""
        a, s = self.cell(cell), self.ligand(ligand)
        # The change in a FiLM transform relative to its identity (PBS) transform.
        return self.readout(self.gamma(s) * a + self.beta(s))

    @torch.inference_mode()
    def predict(self, cell: torch.Tensor, ligand: torch.Tensor, baseline: torch.Tensor):
        delta = self(cell, ligand)
        if baseline.shape != delta.shape:
            raise ValueError("One PBS baseline on the model's gene axis is required per input row")
        return delta, baseline + delta
