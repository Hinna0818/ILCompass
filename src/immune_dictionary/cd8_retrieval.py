"""Match predicted CD8 cytokine responses to independent training batches."""

from itertools import combinations

import torch
from torch.nn import functional as F

if __package__:
    from . import train
    from .cd8_pair_loss import condition_averager
else:
    import train
    from cd8_pair_loss import condition_averager


TEMPERATURE = 0.1


def centered_directions(values):
    return F.normalize(values - values.mean(0, keepdim=True), dim=1, eps=1e-8)


class CrossBatchRetrieval:
    """Equal-weight bidirectional InfoNCE across batch pairs sharing at least two reagents."""

    def __init__(self, rows, targets):
        if targets.ndim != 2 or len(targets) != len(rows):
            raise ValueError("Expect a row-aligned gene expression delta for each fitting mouse")
        conditions, mean = condition_averager(rows)
        self.mean = mean.to(targets).to_sparse()
        observed = torch.sparse.mm(self.mean, targets.detach())
        conditions["condition_index"] = conditions.index
        batches = [batch.set_index("canonical_id").condition_index
                   for _, batch in conditions.groupby("processing_batch", sort=False)]
        eligible = set()
        query_indices = []
        self.references = []
        self.sizes = []
        for left, right in combinations(batches, 2):
            common = left.index.intersection(right.index, sort=False)
            if len(common) < 2:
                continue
            eligible.update(common)
            labels = torch.arange(len(common), device=targets.device)
            for source, reference in ((left, right), (right, left)):
                query_indices.extend(source.loc[common].tolist())
                ref_ids = reference.loc[common].tolist()
                self.references.append((centered_directions(observed[ref_ids]), labels))
                self.sizes.append(len(common))
        self.query_indices = torch.tensor(query_indices, dtype=torch.long, device=targets.device)
        self.n_pairs = len(self.references) // 2
        self.n_ligands = len(eligible)

    def __call__(self, prediction):
        if not self.n_pairs:
            return prediction[:0].sum()
        means = torch.sparse.mm(self.mean, prediction)
        queries = means.index_select(0, self.query_indices).split(self.sizes)
        losses = []
        for query, (reference, labels) in zip(queries, self.references):
            logits = centered_directions(query) @ reference.T / TEMPERATURE
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()


def fit(ds, truth, rows, weight, epochs, device, observe_epochs, observe):
    regularizer = None
    if weight:
        targets = torch.as_tensor(truth[rows.target_index], device=device)
        retrieval = CrossBatchRetrieval(rows, targets)

        def regularizer(prediction, ids):
            # The training loop shuffles the complete fit set on every full-batch update.
            return weight * retrieval(prediction[torch.argsort(ids)])

    return train.fit_model(ds, rows, truth, epochs, device, len(rows), False, weight_by=["mouse_id"],
                           observe_epochs=observe_epochs, observe=observe, prediction_loss=regularizer)
