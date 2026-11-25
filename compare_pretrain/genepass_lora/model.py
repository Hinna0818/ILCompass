"""Q/V LoRA only; frozen native heads evaluated on selected MLM positions."""
from common import *
import math
import torch
from torch import nn
from torch.nn import functional as F
from genecompass_compat import load_genecompass

class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha,dropout):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.A=nn.Parameter(torch.empty(rank,base.in_features));self.B=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5));self.dropout=nn.Dropout(dropout)
    def forward(self,x):return self.base(x)+F.linear(F.linear(self.dropout(x),self.A),self.B)*self.scale

def build(cfg,adapter=None):
    model,report=load_genecompass(ROOT/cfg['checkpoint'])
    if report['missing'] or report['unexpected'] or not model.bert.use_cls_token:raise ValueError(report)
    model.requires_grad_(False)
    for layer in model.bert.encoder.layer:
        for name in ('query','value'):
            base=getattr(layer.attention.self,name)
            setattr(layer.attention.self,name,LoRALinear(base,cfg['rank'],cfg['alpha'],cfg['lora_dropout']))
    # Older HF checkpointing is reentrant and needs a grad-bearing input even with frozen embeddings.
    model.bert.encoder.gradient_checkpointing=cfg['gradient_checkpointing']
    def require_input_grad(module,args,output):
        if module.training and torch.is_grad_enabled() and cfg['gradient_checkpointing']:output.requires_grad_(True)
        return output
    model.bert.embeddings.register_forward_hook(require_input_grad)
    if adapter is not None:
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if set(adapter)!=expected:raise ValueError('Adapter keys mismatch')
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in adapter:p.copy_(adapter[n])
    return model

def masked_outputs(model,batch):
    h=model.bert(input_ids=batch['input_ids'],values=batch['values'],attention_mask=batch['attention_mask'],species=batch['species'],emb_warmup_alpha=1.,return_dict=True).last_hidden_state[:,1:]
    mask=batch['labels'].ne(-100)
    if not mask.any():raise ValueError('No masked targets')
    # Both native heads are token-wise: selecting positions first preserves native losses.
    z=h[mask];logits=model.cls(z).float();pred=model.cls4value(z).squeeze(-1).float()
    targets=batch['labels'][mask];truth=batch['labels_values'][mask].float()
    ce=F.cross_entropy(logits,targets);mse=F.mse_loss(pred,truth)
    return dict(loss=.2*ce+.8*mse,id=ce,expression=mse,targets=targets,pred=pred,truth=truth,logits=logits)
