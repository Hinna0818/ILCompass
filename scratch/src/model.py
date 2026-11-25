"""E1b: small, randomly initialized GeneCompass-style joint MLM encoder."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    n_genes: int
    max_length: int = 8192
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 6
    intermediate_size: int = 768
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.attn_out = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.ffn = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.intermediate_size), nn.GELU(),
                                 nn.Linear(cfg.intermediate_size, cfg.hidden_size))
        self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, valid):
        b, length, width = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.heads, width // self.heads).unbind(2)
        # SDPA True means an allowed key; PAD queries are cleared after the block.
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                                  dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(b, length, width)
        x = self.attn_norm(x + self.dropout(self.attn_out(attended)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x.masked_fill(~valid.unsqueeze(-1), 0)


class ILC2Transformer(nn.Module):
    """Special IDs are frozen by the data protocol: PAD=0, MASK=1, genes start at 2.

    Only the four forward arguments are encoder inputs. Labels and source metadata
    must stay outside forward. IDF pooling is for unmasked representation extraction.
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_heads or config.n_genes <= 0:
            raise ValueError("Invalid hidden/head dimensions or vocabulary")
        self.config = config
        d = config.hidden_size
        self.gene_embedding = nn.Embedding(config.n_genes + 2, d, padding_idx=0)
        self.value_projection = nn.Sequential(nn.Linear(d + 1, d), nn.GELU())
        self.rank_embedding = nn.Embedding(config.max_length, d)
        self.input_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])
        self.id_transform = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                           nn.LayerNorm(d, eps=config.layer_norm_eps))
        self.id_bias = nn.Parameter(torch.zeros(config.n_genes))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.apply(self._initialize)
        with torch.no_grad():
            self.gene_embedding.weight[0].zero_()

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, values, attention_mask, position_ids):
        if input_ids.shape != values.shape or input_ids.shape != attention_mask.shape or input_ids.shape != position_ids.shape:
            raise ValueError("Encoder input shapes must agree")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds audited position capacity; no automatic truncation")
        gene = self.gene_embedding(input_ids)
        x = self.value_projection(torch.cat([gene, values.unsqueeze(-1).to(gene.dtype)], dim=-1))
        x = self.dropout(self.input_norm(x + self.rank_embedding(position_ids)))
        x = x.masked_fill(~attention_mask.unsqueeze(-1), 0)
        for layer in self.layers:
            x = layer(x, attention_mask)
        pooled = x.sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
        return {"gene_embeddings": x, "cell_embedding": pooled}

    def predict_targets(self, hidden, target_mask):
        h = hidden[target_mask]
        # Weight tying excludes PAD/MASK; never form [B,L,vocabulary] logits.
        logits = F.linear(self.id_transform(h), self.gene_embedding.weight[2:], self.id_bias)
        return logits, self.value_head(h).squeeze(-1)

    def pool_idf(self, hidden, input_ids, attention_mask, gene_idf):
        if ((input_ids < 2) & attention_mask).any() or len(gene_idf) != self.config.n_genes:
            raise ValueError("IDF pooling needs unmasked real genes and frozen vocabulary-aligned IDF")
        weights = gene_idf[(input_ids - 2).clamp_min(0)] * attention_mask
        return (hidden * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-12)


MODEL_INPUTS = ("input_ids", "values", "attention_mask", "position_ids")


def mlm_objective(model, batch):
    """Return SUMS of cell-mean losses; trainer supplies the accumulation denominator.

    FP32 CE/MSE keep the continuous value loss stable under BF16 autocast.
    Empty target cells contribute neither numerator nor denominator.
    """
    hidden = model(**{key: batch[key] for key in MODEL_INPUTS})["gene_embeddings"]
    mask = batch["target_mask"] & batch["attention_mask"]
    logits, predicted_values = model.predict_targets(hidden, mask)
    labels = batch["labels"][mask]
    id_losses = F.cross_entropy(logits.float(), labels, reduction="none")
    value_losses = (predicted_values.float() - batch["value_labels"][mask].float()).square()
    cells = mask.nonzero(as_tuple=True)[0]
    counts = mask.sum(1)
    id_per_cell = hidden.new_zeros(len(counts), dtype=torch.float32).index_add(0, cells, id_losses)
    value_per_cell = hidden.new_zeros(len(counts), dtype=torch.float32).index_add(0, cells, value_losses)
    id_per_cell = id_per_cell / counts.clamp_min(1)
    value_per_cell = value_per_cell / counts.clamp_min(1)
    return dict(id_sum=id_per_cell.sum(), value_sum=value_per_cell.sum(), valid_cells=(counts > 0).sum(),
                id_per_cell=id_per_cell, value_per_cell=value_per_cell, target_counts=counts,
                id_losses=id_losses.detach(), value_losses=value_losses.detach(), labels=labels.detach(),
                correct=(logits.argmax(-1) == labels).detach(), target_cells=cells,
                branches=batch["corruption_type"][mask])
