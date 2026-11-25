"""Eight-layer Pre-LN encoder; unchanged v1 joint ID/value MLM objective."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model import EncoderBlock, ILC2Transformer, ModelConfig


@dataclass
class V3Config(ModelConfig):
    norm_type: str = 'pre_ln'


class PreNormBlock(EncoderBlock):
    def forward(self, x, valid):
        b, length, width = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, length, 3, self.heads, width // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                                  dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(b, length, width)
        x = x + self.dropout(self.attn_out(attended))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x.masked_fill(~valid.unsqueeze(-1), 0)


class V3Transformer(ILC2Transformer):
    def __init__(self, config):
        if config.norm_type != 'pre_ln':
            raise ValueError('v3 uses Pre-LN blocks')
        super().__init__(config)
        self.layers = nn.ModuleList(PreNormBlock(config) for _ in range(config.num_layers))
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers.apply(self._initialize)
        self._initialize(self.encoder_norm)

    def forward(self, input_ids, values, attention_mask, position_ids):
        encoded = super().forward(input_ids, values, attention_mask, position_ids)
        hidden = self.encoder_norm(encoded['gene_embeddings']).masked_fill(~attention_mask.unsqueeze(-1), 0)
        return dict(gene_embeddings=hidden,
                    cell_embedding=hidden.sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1))
