"""Mean-response prototype; signed log1p residuals are not a count generator."""
import torch
from torch import nn


class ResidualPredictor(nn.Module):
    def __init__(self, source_h, mean_effects, hidden_size=128):
        super().__init__()
        self.register_buffer("h_mean", source_h.mean(0))
        self.register_buffer("h_std", source_h.std(0, unbiased=False).clamp_min(1e-4))
        self.register_buffer("mean_effects", mean_effects)
        self.condition = nn.Linear(2, 32, bias=False)
        self.head = nn.Sequential(nn.Linear(source_h.shape[1] + 32, hidden_size),
                                  nn.GELU(), nn.Linear(hidden_size, mean_effects.shape[1]))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, source_h, source_y, condition):
        if condition.shape != (len(source_h), 2) or not ((condition == 0) | (condition == 1)).all():
            raise ValueError("Expected binary [IL33,CGRP] condition.")
        if condition.bool().all(-1).any():
            raise ValueError("This prototype excludes combination prediction.")
        x = torch.cat([(source_h - self.h_mean) / self.h_std, self.condition(condition)], -1)
        residual = self.head(x) * condition.sum(-1, keepdim=True)
        return source_y + condition @ self.mean_effects + residual
