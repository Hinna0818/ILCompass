"""Unmasked mmap tokens -> epoch-aware batches -> dynamic PAD and joint MLM.

Only input_ids/values/attention_mask/position_ids are model inputs. All labels,
corruption metadata and source identities are supervision/diagnostics only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from token_data import read_shard

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scratch/configs/input_pipeline_v03.json"


class TokenShardDataset(Dataset):
    def __init__(self, tokenized, split):
        self.root = Path(tokenized)
        self.frame = pd.read_csv(self.root / "cell_manifest.tsv.gz", sep="\t")
        self.frame = self.frame[self.frame.split.eq(split)].reset_index(drop=True)
        if self.frame.empty or not self.frame.cell_id.is_unique:
            raise ValueError(f"Missing or duplicated split cells: {split}")
        self.split = split
        self.lengths = self.frame.length.to_numpy(dtype=np.int64)
        self._shards = {}

    def __len__(self):
        return len(self.frame)

    def __getstate__(self):
        # Spawn workers reopen read-only mappings rather than pickle array data.
        return {**self.__dict__, "_shards": {}}

    def __getitem__(self, key):
        # Epoch travels in each worker task, including persistent/prefetched work.
        index, epoch = key if isinstance(key, tuple) else (int(key), 0)
        row = self.frame.iloc[index]
        name = row.shard
        if name not in self._shards:
            self._shards[name] = read_shard(self.root / name)
        shard = self._shards[name]
        j = int(row.shard_row)
        a, b = map(int, shard["offsets"][j:j+2])
        if b-a != row.length or int(shard["source_rows"][j]) != row.source_row:
            raise ValueError("Shard no longer matches cell manifest")
        return dict(gene_ids=shard["gene_ids"][a:b], values=shard["values"][a:b],
                    cell_id=row.cell_id, source_row=int(row.source_row), sample_index=int(index),
                    epoch=int(epoch), split=self.split)


class RandomLengthBatchSampler(Sampler):
    """Global random pools, length-sort within pools, then globally shuffle batches.

Every cell is visited once, including the final incomplete batch. State is based
on batches consumed by the trainer, never on indices dispatched by prefetch.
"""
    def __init__(self, lengths, batch_size, bucket_size, seed=0, shuffle=True):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if batch_size <= 0 or bucket_size < batch_size or bucket_size % batch_size:
            raise ValueError("bucket_size must be a positive multiple of batch_size")
        if len(self.lengths) == 0 or (self.lengths <= 0).any():
            raise ValueError("Empty or invalid sequence lengths")
        self.batch_size, self.bucket_size = int(batch_size), int(bucket_size)
        self.seed, self.shuffle = int(seed), bool(shuffle)
        self.epoch, self.start_batch = 0, 0

    @property
    def total_batches(self):
        return (len(self.lengths)+self.batch_size-1)//self.batch_size

    def set_epoch(self, epoch, start_batch=0):
        if epoch < 0 or not 0 <= start_batch <= self.total_batches:
            raise ValueError("Invalid epoch or consumed-batch offset")
        self.epoch = int(epoch) if self.shuffle else 0
        self.start_batch = int(start_batch)

    def batches(self):
        if not self.shuffle:
            order = np.argsort(self.lengths, kind="stable")
            return [order[i:i+self.batch_size] for i in range(0,len(order),self.batch_size)]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        order = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0,len(order),self.bucket_size):
            pool = order[start:start+self.bucket_size]
            pool = pool[np.argsort(self.lengths[pool],kind="stable")]
            batches += [pool[j:j+self.batch_size] for j in range(0,len(pool),self.batch_size)]
        return [batches[i] for i in rng.permutation(len(batches))]

    def __iter__(self):
        for batch in self.batches()[self.start_batch:]:
            yield [(int(i),self.epoch) for i in batch]

    def __len__(self):
        return self.total_batches-self.start_batch

    def state_dict(self, consumed_batches):
        """consumed_batches is relative to the current iterator's start offset."""
        start = self.start_batch+int(consumed_batches)
        if not self.start_batch <= start <= self.total_batches:
            raise ValueError("Invalid number of consumed batches")
        return dict(epoch=self.epoch,start_batch=start,seed=self.seed,batch_size=self.batch_size,
                    bucket_size=self.bucket_size,num_samples=len(self.lengths),shuffle=self.shuffle)

    def load_state_dict(self, state):
        current=self.state_dict(0)
        for key in ("seed","batch_size","bucket_size","num_samples","shuffle"):
            if state[key] != current[key]:
                raise ValueError(f"Sampler resume configuration mismatch: {key}")
        self.set_epoch(state["epoch"],state["start_batch"])


class JointMLMCollator:
    def __init__(self, normalization, config, n_genes, training):
        self.p = float(normalization["mask_probability"])
        self.pad_id = int(normalization["pad_token_id"])
        self.mask_id = int(normalization["mask_token_id"])
        self.first = int(normalization["first_gene_token_id"])
        self.n_genes = int(n_genes)
        self.seed = int(config["mask_seed"])
        self.replace = float(config["mask_replacement_fraction"])
        self.random = float(config["random_replacement_fraction"])
        self.training = bool(training)
        if not 0 <= self.p <= 1 or not 0 <= self.replace <= 1 or not 0 <= self.random <= 1-self.replace:
            raise ValueError("Invalid masking probabilities")

    def __call__(self, samples):
        lengths = np.array([len(s["gene_ids"]) for s in samples],dtype=np.int64)
        if not len(samples) or (lengths <= 0).any():
            raise ValueError("Empty samples")
        b,l = len(samples),int(lengths.max())
        ids = np.full((b,l),self.pad_id,dtype=np.int64)
        values = np.zeros((b,l),dtype=np.float32)
        target = np.zeros((b,l),dtype=bool)
        kind = np.zeros((b,l),dtype=np.int8)
        labels = np.full((b,l),-100,dtype=np.int64)
        target_gene_ids = np.full((b,l),-100,dtype=np.int64)
        value_labels = np.zeros((b,l),dtype=np.float32)
        valid = np.arange(l)[None,:] < lengths[:,None]
        for i,sample in enumerate(samples):
            n = lengths[i]
            ids[i,:n] = sample["gene_ids"]
            values[i,:n] = sample["values"]
            epoch = sample["epoch"] if self.training else 0
            domain = {"train":11,"validation":12,"test":13}[sample["split"]]
            rng = np.random.default_rng(np.random.SeedSequence([
                self.seed,int(epoch),int(sample["source_row"]),domain,int(self.training)]))
            chosen = rng.random(n) < self.p
            target[i,:n] = chosen
            positions = np.flatnonzero(chosen)
            target_gene_ids[i,positions] = ids[i,positions]
            # ID head predicts only real genes. CE classes are 0..n_genes-1.
            labels[i,positions] = ids[i,positions]-self.first
            value_labels[i,positions] = values[i,positions]
            branch = rng.random(len(positions))
            mask_positions = positions[branch < self.replace]
            random_positions = positions[(branch >= self.replace)&(branch < self.replace+self.random)]
            keep_positions = positions[branch >= self.replace+self.random]
            ids[i,mask_positions] = self.mask_id
            values[i,mask_positions] = -1
            ids[i,random_positions] = rng.integers(self.first,self.first+self.n_genes,size=len(random_positions))
            values[i,random_positions] = -1
            kind[i,mask_positions],kind[i,random_positions],kind[i,keep_positions] = 1,2,3
        positions = np.broadcast_to(np.arange(l,dtype=np.int64),(b,l)).copy()
        positions[~valid]=0
        arrays = dict(input_ids=ids,values=values,attention_mask=valid,position_ids=positions,
            labels=labels,target_gene_ids=target_gene_ids,value_labels=value_labels,
            target_mask=target,corruption_type=kind,lengths=lengths,target_counts=target.sum(axis=1),
            source_rows=np.array([s["source_row"] for s in samples],dtype=np.int64),
            sample_indices=np.array([s["sample_index"] for s in samples],dtype=np.int64),
            epochs=np.array([s["epoch"] if self.training else 0 for s in samples],dtype=np.int64))
        batch = {k:torch.from_numpy(v) for k,v in arrays.items()}
        batch["loss_cell_mask"] = batch["target_counts"] > 0
        batch["cell_ids"] = [s["cell_id"] for s in samples]
        return batch


def build_loader(split, config=None):
    cfg = json.loads(DEFAULT_CONFIG.read_text()) if config is None else config
    tokenized = ROOT / cfg["tokenized"]
    prepared = tokenized.parent
    norm = json.loads((prepared / "config.json").read_text())
    detail = json.loads((tokenized / "config.json").read_text())
    vocab = pd.read_csv(prepared / "gene_vocab.tsv",sep="\t",usecols=["token_id"])
    dataset = TokenShardDataset(tokenized,split)
    if dataset.lengths.max() > detail["capacity"]:
        raise ValueError("Input exceeds audited position capacity")
    sampler = RandomLengthBatchSampler(dataset.lengths,cfg["batch_size"],cfg["bucket_size"],
        cfg["sampler_seed"],shuffle=split=="train")
    collator = JointMLMCollator(norm,cfg,len(vocab),training=split=="train")
    return DataLoader(dataset,batch_sampler=sampler,collate_fn=collator,num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"]>0,pin_memory=cfg["pin_memory"],
        generator=torch.Generator().manual_seed(cfg["sampler_seed"]))
