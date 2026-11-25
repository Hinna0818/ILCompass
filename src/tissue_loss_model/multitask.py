"""Tissue auxiliary task; labels/weights come only from the locked training split."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix


def tissue_info(meta):
    if meta.tissue.eq('').any() or meta.tissue.isna().any():
        raise ValueError('Missing tissue annotation; do not infer labels')
    counts = meta.tissue.value_counts().sort_index()
    if len(counts) != 6:
        raise ValueError('This experiment requires the six existing tissues')
    return dict(labels=counts.index.tolist(), train_counts=counts.astype(int).tolist(),
                class_weights=(len(meta) / (len(counts) * counts)).tolist())


def add_labels(ds, meta, info):
    mapping = {name: i for i, name in enumerate(info['labels'])}
    labels = meta.tissue.map(mapping)
    if labels.isna().any():
        raise ValueError('Unknown tissue outside the training label vocabulary')
    return ds.add_column('tissue_labels', labels.astype(int).tolist())


class TissueCollator:
    def __init__(self, mlm_collator):
        self.mlm_collator = mlm_collator

    def __call__(self, examples):
        # Labels are targets only: never pass them to the tokenizer or encoder.
        labels = torch.tensor([x['tissue_labels'] for x in examples], dtype=torch.long)
        batch = self.mlm_collator([{k: v for k, v in x.items() if k != 'tissue_labels'} for x in examples])
        batch['tissue_labels'] = labels
        return batch


def gene_mean(hidden, attention_mask, use_cls):
    if use_cls:
        hidden = hidden[:, 1:]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    if hidden.shape[:2] != attention_mask.shape or (mask.sum(1) == 0).any():
        raise ValueError('Invalid gene pooling mask')
    return (hidden * mask).sum(1) / mask.sum(1)


class TissueGeneCompass(nn.Module):
    def __init__(self, backbone, info):
        super().__init__()
        self.backbone = backbone
        self.info = info
        self.tissue_head = nn.Linear(backbone.config.hidden_size, len(info['labels']))
        nn.init.normal_(self.tissue_head.weight, std=backbone.config.initializer_range)
        nn.init.zeros_(self.tissue_head.bias)
        self.register_buffer('class_weights', torch.tensor(info['class_weights'], dtype=torch.float32))

    @property
    def bert(self):
        return self.backbone.bert

    @property
    def config(self):
        return self.backbone.config

    def forward(self, tissue_labels, **batch):
        # One corrupted view serves all three training tasks; no second train forward.
        out = self.backbone(**batch, output_hidden_states=True, return_dict=True)
        h = gene_mean(out.hidden_states[-1], batch['attention_mask'], self.bert.use_cls_token)
        out['hidden_states'] = None
        logits = self.tissue_head(h)
        per_cell = F.cross_entropy(logits.float(), tissue_labels, reduction='none')
        out['tissue_logits'] = logits
        # Global train-frequency weights, fixed denominator N_batch: unbiased macro objective.
        out['tissue_loss'] = (per_cell * self.class_weights[tissue_labels]).mean()
        return out

    def classify(self, batch):
        hidden = self.bert(**batch, emb_warmup_alpha=1.0, return_dict=True).last_hidden_state
        return self.tissue_head(gene_mean(hidden, batch['attention_mask'], self.bert.use_cls_token))

    def save_auxiliary(self, directory):
        partial = directory / 'tissue_head.partial.pt'
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in self.tissue_head.state_dict().items()},
                        tissue_info=self.info), partial)
        partial.replace(directory / 'tissue_head.pt')


def attach_tissue_head(backbone, directory, info, device):
    # The pretrained prior warmup must already be complete, as in the existing runs.
    if float(backbone.emb_warmup.alpha) < .999:
        raise ValueError('Pretrained prior embedding warmup is incomplete')
    model = TissueGeneCompass(backbone, info).to(device)
    path = Path(directory) / 'tissue_head.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['tissue_info'] != info:
            raise ValueError('Saved tissue head label order/class weights changed')
        model.tissue_head.load_state_dict(saved['state_dict'], strict=True)
    return model


def classification_metrics(target, prediction, ce, labels):
    target, prediction, ce = np.asarray(target), np.asarray(prediction), np.asarray(ce)
    cm = confusion_matrix(target, prediction, labels=np.arange(len(labels)))
    support = cm.sum(1)
    present = support > 0
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(labels)), where=present)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(len(labels)), where=cm.sum(0)>0)
    f1 = np.divide(2*np.diag(cm), support+cm.sum(0), out=np.zeros(len(labels)), where=support+cm.sum(0)>0)
    by_tissue = {name: dict(cells=int(support[i]), ce=float(ce[target==i].mean()),
                            recall=float(recall[i]), precision=float(precision[i]), F1=float(f1[i]))
                 for i, name in enumerate(labels) if present[i]}
    return dict(cells=len(target), accuracy=float(np.mean(target==prediction)),
                balanced_accuracy=float(recall[present].mean()), macro_F1=float(f1[present].mean()),
                ce=float(ce.mean()), macro_ce=float(np.mean([v['ce'] for v in by_tissue.values()])),
                labels=labels, confusion=cm.tolist(), by_tissue=by_tissue)


@torch.no_grad()
def evaluate_unmasked(model, ds, meta, cfg, device):
    from posttrain_training import autocast
    targets, predictions, errors = [], [], []
    for left in range(0, len(meta), cfg['eval_batch_size']):
        rows = ds[meta.index[left:left+cfg['eval_batch_size']].tolist()]
        length = torch.tensor(rows['length'], device=device)
        n = len(rows['input_ids'][0])
        batch = dict(input_ids=torch.tensor(rows['input_ids'], device=device, dtype=torch.long),
                     values=torch.tensor(rows['values'], device=device, dtype=torch.float32),
                     species=torch.tensor(rows['species'], device=device, dtype=torch.long),
                     attention_mask=torch.arange(n, device=device)[None, :] < length[:, None])
        y = torch.tensor(rows['tissue_labels'], device=device, dtype=torch.long)
        with autocast(device, cfg):
            logits = model.classify(batch)
        targets.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
        errors.extend(F.cross_entropy(logits.float(), y, reduction='none').cpu().tolist())
    return classification_metrics(targets, predictions, errors, model.info['labels'])
