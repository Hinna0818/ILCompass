from common import *
import numpy as np
import pandas as pd
import torch
from posttrain_training import make_collator

class Inputs:
    def __init__(self,cfg):
        folder=ROOT/cfg['data_directory'];self.report=json.loads((folder/'complete.json').read_text())
        if self.report['signature']!=signature(cfg):raise ValueError('Data provenance mismatch; rerun prepare into a new directory')
        self.meta=pd.read_csv(folder/'cells.tsv.gz',sep='\t')
        self.ids=np.load(folder/'input_ids.npy',mmap_mode='r');self.values=np.load(folder/'values.npy',mmap_mode='r');self.lengths=np.load(folder/'lengths.npy')
        if len(self.meta)!=len(self.ids):raise ValueError('Input row mismatch')
        self.collator=make_collator(cfg);self.cfg=cfg
        panel=pd.read_csv(folder/'functional_panel.csv');self.functional=panel.loc[panel.in_input_gene_space,'token_id'].astype(int).to_numpy()
    def rows(self,split):return np.flatnonzero(self.meta.split.eq(split))
    def batch(self,rows,seed,validation=False):
        # Validation mask depends only on corpus row, not packing or batch size.
        width=int(self.lengths[rows].max())
        def call(sub,w,s):
            examples=[dict(input_ids=self.ids[r,:w].tolist(),values=self.values[r,:w].tolist(),species=[1]) for r in sub]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(s));return self.collator(examples)
        if not validation:b=call(rows,width,seed)
        else:
            pieces=[call([r],int(self.lengths[r]),seed+int(r)) for r in rows];b={}
            for key in pieces[0]:
                b[key]=torch.cat([torch.nn.functional.pad(p[key],(0,width-p[key].shape[1]),value=-100 if key=='labels' else 0) if key!='species' else p[key] for p in pieces])
        if (b['labels'][b['attention_mask']==0]!=-100).any():raise ValueError('PAD selected as target')
        return b
