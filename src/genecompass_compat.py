"""加载官方 GeneCompass checkpoint 的兼容层。

官方 checkpoint（Small，scidb 下载）的 config 带 use_value_emb=true：value 先经
ContinuousValueEncoder 变成 hidden 维向量，cat_fc 输入为 hidden*6。固定提交
59e5e48 及 main 分支的 KnowledgeBertEmbeddings 只实现了标量 value 拼接
（hidden*5+1），无法直接加载。这里仅替换 embedding 层，encoder 与 MLM 头沿用官方代码。
"""
from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig

from genecompass import BertForMaskedLM
from genecompass.knowledge_embeddings import KnowledgeBertEmbeddings

PRIORS = ("promoter", "co_exp", "gene_family", "peca_grn")


class ValueEncoder(nn.Module):
    """与官方 ContinuousValueEncoder 参数名一致，去掉写死的 .cuda()。"""

    def __init__(self, d_model, dropout=0.1, max_value=255):
        super().__init__()
        self.max_value = max_value
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.clamp(x.unsqueeze(-1).float(), max=self.max_value)
        return self.dropout(self.norm(self.linear2(self.activation(self.linear1(x)))))


def _patch_embeddings(emb, config, value_position):
    """value_position: 'first'、'after_word'（与官方标量拼接位置一致）或 'last'。"""
    h = config.hidden_size
    emb.values_embeddings = ValueEncoder(h, dropout=config.hidden_dropout_prob)
    emb.concat_embeddings.cat_fc = nn.Linear(h * (2 + len(PRIORS)), h)
    emb.value_position = value_position
    original_forward = emb.forward

    def forward(input_ids=None, values=None, species=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, past_key_values_length=0, emb_warmup_alpha=1.0, **_):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = emb.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        value_embeds = emb.values_embeddings(values)
        parts = [value_embeds] if value_position == "first" else []
        parts.append(emb.word_embeddings(input_ids))
        if value_position == "after_word":
            parts.append(value_embeds)
        shifted = input_ids.detach().clone()
        if species is not None:
            mouse = species.squeeze(1) == 1
            shifted[mouse] = emb.homologous_index[shifted[mouse]]
        for name in PRIORS:
            prior = getattr(emb, f"{name}_knowledge")[shifted]
            parts.append(emb_warmup_alpha * getattr(emb, f"{name}_embeddings")(prior))
        if value_position == "last":
            parts.append(value_embeds)
        x = emb.concat_embeddings(torch.cat(parts, dim=2))
        x = x + emb.token_type_embeddings(token_type_ids) + emb.position_embeddings(position_ids)
        return emb.dropout(emb.LayerNorm(x))

    emb.forward = forward
    emb._official_forward = original_forward
    return emb


def load_genecompass(model_dir, value_position="after_word", device="cpu"):
    """从官方 checkpoint 目录构建模型；先验 buffer 与同源映射直接取自 checkpoint。

    返回 (model, load_report)。load_report 中 missing/unexpected 必须为空才可用于训练。
    """
    model_dir = Path(model_dir)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    config = BertConfig.from_pretrained(model_dir)
    homolog = state["bert.embeddings.homologous_index"]
    knowledges = {name: state[f"bert.embeddings.{name}_knowledge"] for name in PRIORS}
    knowledges["homologous_gene_human2mouse"] = {
        i: int(j) for i, j in enumerate(homolog.tolist()) if i != j
    }
    model = BertForMaskedLM(config, knowledges)
    if config.to_dict().get("use_value_emb", False):
        _patch_embeddings(model.bert.embeddings, config, value_position)
    report = model.load_state_dict(state, strict=False)
    model.tie_weights()
    return model.to(device), {"missing": report.missing_keys, "unexpected": report.unexpected_keys}
