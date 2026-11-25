"""Frozen-FM FiLM head for two named stimuli and their combination.

The caller supplies cached/no-grad FM embeddings and fixed-panel log1p(CP10K).
Fit embedding_mean/std on Train PBS only, after fixing Train/Validation splits.
Condition columns follow stimuli (default IL33,CGRP); (0,0) is PBS identity.
No observed treatment expression or empirical treatment mean is an input.
"""
import torch
from torch import nn

STIMULI = ('IL33', 'CGRP')


class FiLMPerturbationModel(nn.Module):
    """Trainable condition vectors, FiLM adapter, and shared gene-shift decoder.

    Defaults: FM dimension -> 256 -> 128; condition 32 -> 128 -> (gamma,beta);
    concatenate basal/modulated states, then 256 -> 128 -> G.
    Only the two single-stimulus vectors are parameters; combination is their sum.
    """
    def __init__(self, gene_ids, embedding_mean, embedding_std,
                 condition_dim=32, hidden_dim=128, projection_dim=256, stimuli=STIMULI):
        super().__init__()
        self.stimuli = tuple(stimuli)
        if len(self.stimuli) != 2 or len(set(self.stimuli)) != 2 or any(not isinstance(s, str) or not s for s in self.stimuli):
            raise ValueError('Expected two distinct named stimuli')
        self.gene_ids = tuple(gene_ids)
        if (not self.gene_ids or len(set(self.gene_ids)) != len(self.gene_ids)
                or any(not isinstance(g, str) or not g for g in self.gene_ids)):
            raise ValueError('gene_ids must be an ordered, unique, nonempty gene panel')
        mean = torch.as_tensor(embedding_mean, dtype=torch.float32, device='cpu').detach().clone()
        std = torch.as_tensor(embedding_std, dtype=torch.float32, device='cpu').detach().clone()
        if (mean.ndim != 1 or not mean.numel() or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or (std < 0).any()):
            raise ValueError('Expected finite 1D Train-only embedding mean/std with nonnegative std')
        self.register_buffer('h_mean', mean)
        self.register_buffer('h_std', std.clamp_min(1e-4))
        self.basal_projection = nn.Sequential(
            nn.Linear(mean.numel(), projection_dim), nn.LayerNorm(projection_dim),
            nn.GELU(), nn.Linear(projection_dim, hidden_dim))
        self.stimulus_embedding = nn.Embedding(len(self.stimuli), condition_dim)
        nn.init.normal_(self.stimulus_embedding.weight, std=.02)
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        # Start close to identity modulation without blocking condition gradients.
        nn.init.normal_(self.film_generator[-1].weight, std=1e-3)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(self.gene_ids)))
        # Initial predictions equal control-copy for every stimulus, including the combination.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_condition(self, condition):
        """Binary [B,2] multi-hot -> sum of learned single-stimulus vectors."""
        if (condition.ndim != 2 or condition.shape[1] != len(self.stimuli)
                or not ((condition == 0) | (condition == 1)).all()):
            raise ValueError(f'Expected binary [batch,2] condition columns {self.stimuli}')
        return condition.to(self.stimulus_embedding.weight) @ self.stimulus_embedding.weight

    def encode_cell(self, cell_embedding: torch.Tensor) -> torch.Tensor:
        """Project frozen FM features using the training WT normalization."""
        h = (cell_embedding.detach() - self.h_mean) / self.h_std
        return self.basal_projection(h)

    def forward(self, cell_embedding, control_expression, condition):
        """Return raw/effective shifts and nonnegative predicted log expression.

        cell_embedding: [B,d], detached cached or no-grad frozen FM output.
        control_expression: [B,G], nonnegative log1p(CP10K), in gene_ids order.
        condition: [B,2], binary multi-hot in STIMULI order, never a dose vector.
        """
        if (cell_embedding.ndim != 2 or control_expression.ndim != 2
                or cell_embedding.shape[1] != self.h_mean.numel()
                or control_expression.shape != (cell_embedding.shape[0], len(self.gene_ids))
                or condition.ndim != 2 or condition.shape[0] != cell_embedding.shape[0]):
            raise ValueError('Expected aligned [B,FM_dim], [B,G], [B,2] inputs')
        if (not torch.isfinite(cell_embedding).all() or not torch.isfinite(control_expression).all()
                or (control_expression < 0).any()):
            raise ValueError('Expected finite FM embeddings and nonnegative log1p(CP10K) expression')
        e = self.encode_condition(condition)
        # The external FM must also run in eval/no_grad; detach protects this boundary.
        z = self.encode_cell(cell_embedding)
        gamma, beta = self.film_generator(e).chunk(2, dim=-1)
        modulated = (1 + gamma) * z + beta
        raw_delta = self.decoder(torch.cat([z, modulated], dim=-1))
        # Gate by presence, not number of stimuli: a combination must not double the shift.
        raw_delta = raw_delta * condition.any(dim=-1, keepdim=True).to(raw_delta)
        raw_expression = control_expression + raw_delta
        prediction = raw_expression.clamp_min(0)
        return dict(raw_delta=raw_delta, raw_expression=raw_expression,
                    prediction=prediction, effective_delta=prediction - control_expression,
                    basal_embedding=z)

    def get_extra_state(self):
        """Keep gene/condition semantics alongside the numerical checkpoint weights."""
        return dict(gene_ids=self.gene_ids, stimuli=self.stimuli)

    def set_extra_state(self, state):
        if tuple(state['gene_ids']) != self.gene_ids or tuple(state['stimuli']) != self.stimuli:
            raise ValueError('Checkpoint gene order or stimulus columns do not match this model')
