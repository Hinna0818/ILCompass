"""Frozen posttrained ILC2 encoder + DeSCOPE-inspired conditional-prior VAE.

Inputs are encoded with posttrain_data.encode, not normalized expression targets.
This module does not open any downstream train/validation/test expression files.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external/GeneCompass"))
from genecompass_compat import load_genecompass

OFFICIAL_CHECKPOINT = ROOT / "model/posttrained/base_last1_20261022_official_0.8_0.2"


class FrozenILC2Encoder(nn.Module):
    def __init__(self, checkpoint=OFFICIAL_CHECKPOINT):
        super().__init__()
        model, report = load_genecompass(checkpoint, device="cpu")
        if report["missing"] or report["unexpected"]:
            raise ValueError(f"Incomplete backbone load: {report}")
        self.bert = model.bert  # MLM prediction heads are not used downstream.
        self.hidden_size = model.config.hidden_size
        self.checkpoint = str(checkpoint)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent VAE.train() must not reactivate backbone dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids, values, attention_mask):
        if input_ids.shape != values.shape or input_ids.shape != attention_mask.shape:
            raise ValueError("IDs, encoded values and gene attention mask must have equal shapes.")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("Attention mask must be binary.")
        if (attention_mask.sum(1) == 0).any():
            raise ValueError("Empty gene sequence.")
        species = torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
        hidden = self.bert(input_ids=input_ids, values=values, attention_mask=attention_mask,
                           species=species, emb_warmup_alpha=1.0, return_dict=True).last_hidden_state
        if self.bert.use_cls_token:
            hidden = hidden[:, 1:]  # Internal mouse CLS is not a gene.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1)


class GaussianHead(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(),
                                 nn.Linear(256, latent_dim * 2))

    def forward(self, x):
        return self.net(x).chunk(2, dim=-1)  # mean, log variance


def sample_gaussian(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


def posterior_prior_kl(q_mean, q_logvar, p_mean, p_logvar):
    """KL(q||p): sum latent dimensions, then mean cells; no free bits or rescaling."""
    return 0.5 * (p_logvar - q_logvar + torch.exp(q_logvar - p_logvar)
                  + (q_mean - p_mean).square() * torch.exp(-p_logvar) - 1).sum(-1).mean()


class ILC2PerturbationVAE(nn.Module):
    """E0 model: condition=[IL33, CGRP]; no batch/protocol embedding for one protocol.

    forward receives independently sampled, background-matched control/treated
    embeddings. MSE reconstructs the treated cell from its own posterior.
    predict receives controls only. Output gene order must be locked on train.
    """
    def __init__(self, output_gene_ids, checkpoint=OFFICIAL_CHECKPOINT, beta=0.01):
        super().__init__()
        if not output_gene_ids or len(set(output_gene_ids)) != len(output_gene_ids):
            raise ValueError("Provide a nonempty, unique ordered output gene panel.")
        if beta < 0:
            raise ValueError("beta must be nonnegative.")
        self.output_gene_ids = tuple(output_gene_ids)
        self.beta = beta
        self.encoder = FrozenILC2Encoder(checkpoint)
        # Bias-free linear sum: PBS=0, IL33=eA, CGRP=eB, combination=eA+eB.
        self.perturbation = nn.Linear(2, 32, bias=False)
        self.prior = GaussianHead(self.encoder.hidden_size + 32, 64)
        self.posterior = GaussianHead(self.encoder.hidden_size + 32, 64)
        self.decoder = nn.Sequential(nn.Linear(64 + 32, 256), nn.GELU(),
                                     nn.Linear(256, len(output_gene_ids)), nn.Softplus())

    def condition_embedding(self, condition):
        if condition.ndim != 2 or condition.shape[1] != 2:
            raise ValueError("Condition must be [batch, 2] in IL33,CGRP order.")
        if not ((condition == 0) | (condition == 1)).all():
            raise ValueError("E0 uses binary perturbation indicators.")
        return self.perturbation(condition.to(self.perturbation.weight.dtype))

    def forward(self, control_h, treated_h, condition, target_y):
        e = self.condition_embedding(condition)
        if condition.bool().all(-1).any():
            raise ValueError("Combination responses cannot enter the training forward path.")
        if target_y.shape != (len(control_h), len(self.output_gene_ids)):
            raise ValueError("Target expression does not match batch/output panel.")
        pm, pl = self.prior(torch.cat([control_h, e], -1))
        qm, ql = self.posterior(torch.cat([treated_h, e], -1))
        prediction = self.decoder(torch.cat([sample_gaussian(qm, ql), e], -1))
        mse = F.mse_loss(prediction, target_y)
        kl = posterior_prior_kl(qm, ql, pm, pl)
        return {"loss": mse + self.beta * kl, "mse": mse, "kl": kl,
                "weighted_kl": self.beta * kl, "posterior_reconstruction": prediction}

    @torch.no_grad()
    def predict(self, control_h, condition):
        """Prior sampling only; no treated input or true target required."""
        e = self.condition_embedding(condition)
        mean, logvar = self.prior(torch.cat([control_h, e], -1))
        return self.decoder(torch.cat([sample_gaussian(mean, logvar), e], -1))

    def get_extra_state(self):
        return {"output_gene_ids": self.output_gene_ids, "beta": self.beta,
                "backbone_checkpoint": self.encoder.checkpoint}

    def set_extra_state(self, state):
        if tuple(state["output_gene_ids"]) != self.output_gene_ids:
            raise ValueError("Saved output gene order differs from this model.")
        if state["backbone_checkpoint"] != self.encoder.checkpoint:
            raise ValueError("Saved backbone identity differs from this model.")
        self.beta = state["beta"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build official-backbone VAE; no training or dataset access")
    parser.add_argument("--gene-panel", type=Path, required=True,
                        help="Ordered train-only output Ensembl IDs, one per line, no header")
    args = parser.parse_args()
    model = ILC2PerturbationVAE(args.gene_panel.read_text().splitlines())
    print(model)
    print("Frozen parameters:", sum(p.numel() for p in model.parameters() if not p.requires_grad))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
