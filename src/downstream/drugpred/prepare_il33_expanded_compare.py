"""Explicit PBS/IL33 merge and matched frozen features for three encoders.

External IL33 expression is not opened here. Original counts remain in h5ad;
log1p(CP10K) targets and median/top2048 encoder inputs are separate artifacts.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'
import argparse
import json
import pickle
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from models import ROOT
from posttrain_data import encode, file_hash
from genecompass_coverage import map_genes
from qc_utils import _raw_counts

CFG = json.loads((ROOT/'configs/drugpred_il33_expanded_deg.json').read_text())
DATA = ROOT/CFG['data_directory']
OUT = ROOT/CFG['result_directory']
SOURCE = ROOT/CFG['source_pool']
PATHS = dict(train=SOURCE/'train/train.h5ad',
    development=SOURCE/'development_validation/development_validation.h5ad',
    external=SOURCE/'external_test/input/external_input.h5ad')


def normalized(a, genes):
    x = a.X.astype('float32')
    x = x.multiply((1e4/np.asarray(x.sum(1)).ravel())[:,None]).tocsr()
    x.data = np.log1p(x.data)
    loc = a.var_names.get_indexer(genes)
    y = np.full((a.n_obs, len(genes)), np.nan, dtype='float32')
    valid = loc >= 0
    y[:,valid] = x[:,loc[valid]].toarray()
    return y


def prepare():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    prep = json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    signature = dict(config=CFG, sources={k:file_hash(p) for k,p in PATHS.items()},
        checkpoints={k:file_hash(ROOT/p/'pytorch_model.bin') for k,p in CFG['models'].items()},
        tokens=file_hash(ROOT/prep['token_dictionary']), medians=file_hash(ROOT/prep['gene_median_reference']),
        gene_panel=file_hash(ROOT/CFG['gene_panel']), encoding=prep['encoding'],
        max_gene_tokens=prep['max_gene_tokens'], precision='float32', pooling='masked_gene_mean')
    if (DATA/'signature.json').exists() and json.loads((DATA/'signature.json').read_text()) != signature:
        raise ValueError('Preparation changed; choose a new version')
    (DATA/'signature.json').write_text(json.dumps(signature, indent=2))
    (OUT/'config.json').write_text(json.dumps(CFG, indent=2))
    if (DATA/'prepared.json').exists():
        print('PREPARE completed cache reused', flush=True); return
    tokens = pickle.loads((ROOT/prep['token_dictionary']).read_bytes())
    medians = pickle.loads((ROOT/prep['gene_median_reference']).read_bytes())
    genes = pd.read_csv(ROOT/CFG['gene_panel'], sep='\t')
    genes.to_csv(DATA/'genes.tsv', sep='\t', index=False)
    inputs, pbs_meta, counts, seen = [], [], [], set()
    for role,path in PATHS.items():
        a = ad.read_h5ad(path)
        a = a[a.obs.condition.isin(['PBS','IL33'])].copy()
        a.X = _raw_counts(a.X)
        if ((a.X != a.layers['counts']).nnz or not a.obs.qc_version.eq('ilc2_balanced_v1').all()
                or not a.obs_names.is_unique or not a.var_names.is_unique
                or seen.intersection(a.obs_names) or not a.obs.selected_ILC2.all()):
            raise ValueError('Counts, QC, identity or source disjointness failed')
        seen.update(a.obs_names)
        if role == 'train':
            if set(a.obs.response_group) != set(CFG['group_weights']):
                raise ValueError('Unexpected training backgrounds')
        elif role == 'development':
            if not a.obs.response_group.eq('GSE136154_D').all():
                raise ValueError('Unexpected development background')
        elif not (a.obs.study.eq('GSE189285') & a.obs.condition.eq('PBS') & a.obs.tissue.eq('Lung')).all():
            raise ValueError('External preparation accepts lung PBS only')
        a.obs['evaluation_role'] = role
        a.obs['allowed_for_fit'] = role == 'train'
        np.save(DATA/f'{role}_y.npy', normalized(a, genes.gene_id))
        a.obs.to_csv(DATA/f'{role}_metadata.tsv.gz', sep='\t', index_label='cell_id')
        pbs = a[a.obs.condition.eq('PBS')].copy()
        pbs_meta.append(pbs.obs)
        col,med,_ = map_genes(pbs.var_names, tokens, medians)
        inputs.append(encode(pbs.X, col, med, prep['max_gene_tokens'], prep['encoding']))
        for (study,condition),n in a.obs.groupby(['study','condition'], observed=True).size().items():
            counts.append(dict(role=role, study=study, condition=condition, cells=int(n)))
        if role != 'external':
            a.uns.clear()
            a.uns['qc_version'] = 'ilc2_balanced_v1'
            a.uns['role'] = role
            a.write_h5ad(DATA/f'{role}.h5ad', compression='gzip')
        else:
            genes['external_measured'] = genes.gene_id.isin(a.var_names)
            genes.to_csv(DATA/'genes.tsv', sep='\t', index=False)
        print('PREPARED', role, a.shape, 'PBS', pbs.n_obs, flush=True)
    np.savez_compressed(DATA/'PBS_inputs.npz', ids=np.concatenate([p[0] for p in inputs]),
        vals=np.concatenate([p[1] for p in inputs]), length=np.concatenate([p[2] for p in inputs]))
    pd.concat(pbs_meta).to_csv(DATA/'PBS_metadata.tsv.gz', sep='\t', index_label='cell_id')
    pd.DataFrame(counts).to_csv(OUT/'cell_counts.tsv', sep='\t', index=False)
    boundary = dict(train_cells=24649, development_cells=7553, external_PBS=2853,
        output_genes=len(genes), external_measured_genes=int(genes.external_measured.sum()),
        external_target=str((SOURCE/'external_test/sealed/external_target_sealed.h5ad').relative_to(ROOT)),
        external_target_opened=False, output_panel='unchanged C and D PBS >=1% detection; no IL33 target used',
        exclusion=dict(GSE218353='PBS IL33 background unresolved; QC complete but no response labels inferred',
                       GSE189285_liver='separate cross-tissue task', CGRP='not IL33 alone', combination='sealed'),
        expression='log1p(CP10K); denominator uses all original GEX features; missing features are NaN',
        roles='C + GSE102299 train; D development; GSE189285 lung external, previously evaluated')
    (DATA/'prepared.json').write_text(json.dumps(boundary, indent=2))
    (OUT/'data_boundary.json').write_text(json.dumps(boundary, indent=2))


def encode_controls():
    from train_il33_random_pairs import prepare as official_prepare
    from train_il33_fixed15 import external_embedding
    from encode_original_il33 import original_h
    sys.path.insert(0, str(ROOT/'src/downstream/compare'))
    from evaluate_tissue import fill_embeddings
    sig = json.loads((DATA/'signature.json').read_text())
    meta = pd.read_csv(DATA/'PBS_metadata.tsv.gz', sep='\t', index_col=0)
    inp = np.load(DATA/'PBS_inputs.npz')
    # Reuse only audited per-cell features. Each existing helper checks its source/encoding signature.
    official, _, osig = official_prepare()
    if osig['checkpoint'] != sig['checkpoints']['ILC2_official']:
        raise ValueError('Official encoder checkpoint differs')
    for name,checkpoint in CFG['models'].items():
        dest = DATA/name; dest.mkdir(exist_ok=True)
        if (dest/'PBS_h.npy').exists():
            print('ENCODER completed cache reused', name, flush=True); continue
        partial = dest/'PBS_h.partial.npy'
        exists = partial.exists()
        h = np.lib.format.open_memmap(partial, mode='r+' if exists else 'w+',
                                     dtype='float32', shape=(len(meta),768))
        if not exists:
            h[:] = np.nan
            if name == 'ILC2_official':
                for role,old_role in [('train','train'),('development','validation')]:
                    obs,_,pbs,features = official[old_role]
                    loc = meta.index.get_indexer(obs.index[pbs])
                    if (loc<0).any(): raise ValueError('Missing official PBS cells')
                    h[loc] = features.numpy()
                a = ad.read_h5ad(PATHS['external'])
                h[meta.evaluation_role.eq('external')] = external_embedding(a).numpy()
            elif name == 'GeneCompass':
                for role,key in [('train','train'),('development','validation'),('external','Lung')]:
                    a = ad.read_h5ad(PATHS[role]); a = a[a.obs.condition.eq('PBS')].copy()
                    loc = meta.index.get_indexer(a.obs_names)
                    if (loc<0).any(): raise ValueError('Missing original PBS cells')
                    h[loc] = original_h(a, key)
            else:
                old = ROOT/'data/processed/GSE136154_IL33_PBS_overlap_v1'
                oldsig = json.loads((old/name/'signature.json').read_text())
                if any(oldsig[k] != sig[k] for k in ['tokens','medians','encoding','max_gene_tokens']):
                    raise ValueError('Legacy tissue cache encoding differs')
                if oldsig['checkpoint'] != sig['checkpoints'][name]:
                    raise ValueError('Legacy tissue checkpoint differs')
                om = pd.read_csv(old/name/'PBS_metadata.tsv.gz',sep='\t',index_col=0)
                loc = meta.index.get_indexer(om.index)
                oi = np.load(old/'PBS_inputs.npz')
                if (loc<0).any() or any(not np.array_equal(inp[k][loc],oi[k]) for k in ['ids','vals','length']):
                    raise ValueError('Legacy tissue encoder inputs or cell order differ')
                h[loc] = np.load(old/name/'PBS_h.npy')
            h.flush()
        missing = np.flatnonzero(~np.isfinite(h).all(1))
        if len(missing):
            fill_embeddings(ROOT/checkpoint, h, missing, inp['ids'][missing], inp['vals'][missing],
                            inp['length'][missing], name+'_expanded_PBS')
        if not np.isfinite(h).all(): raise ValueError('Incomplete frozen features')
        h.flush(); del h; partial.replace(dest/'PBS_h.npy')
        print('ENCODER COMPLETE', name, flush=True)
    (DATA/'encoded.json').write_text(json.dumps(dict(models=list(CFG['models']), PBS_cells=len(meta),
        frozen=True, precision='FP32, TF32 disabled', pooling='masked gene mean without species CLS'),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','encode','all']);args=p.parse_args()
    torch.set_num_threads(CFG['cpu_threads'])
    with threadpool_limits(limits=CFG['cpu_threads']):
        if args.stage in ['prepare','all']: prepare()
        if args.stage in ['encode','all']: encode_controls()
