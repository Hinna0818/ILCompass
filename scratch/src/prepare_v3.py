"""Four-tissue counts -> shared Train-only preparation, tokens and input acceptance."""
import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

import prepare as preparation
import tokenize_corpus as tokenization
import validate_input_pipeline as pipeline

ROOT = preparation.ROOT
CONFIG = ROOT / 'scratch/configs/preparation_v3.json'
TOKEN_CONFIG = ROOT / 'scratch/configs/tokenization_v3.json'
INPUT_CONFIG = ROOT / 'scratch/configs/input_pipeline_v3.json'


def subset_corpus(cfg):
    destination = ROOT / cfg['corpus']
    if destination.exists():
        return
    temporary = destination.with_suffix('.partial.h5ad')
    if temporary.exists():
        raise FileExistsError(f'Incomplete corpus exists: {temporary}')
    old = ROOT / cfg['parent_prepared']
    manifest = pd.read_csv(old / 'split_manifest.tsv.gz', sep='\t', index_col='cell_id')
    expected = manifest.loc[manifest.tissue.isin(cfg['tissues'])]
    source = ROOT / cfg['parent_corpus']
    sources = pd.read_csv(old / 'source_manifest.tsv', sep='\t').set_index('path')
    digest = preparation.file_hash(source)
    if digest != sources.loc[cfg['parent_corpus'], 'sha256']:
        raise ValueError('Original corpus changed since frozen preparation')
    print('Reading original sparse counts; retaining four tissues', flush=True)
    with h5py.File(source) as handle:
        obs, var = read_elem(handle['obs']), read_elem(handle['var'])
        raw = read_elem(handle['X'])
        for key in ('data', 'indices', 'indptr'):
            array, saved = getattr(raw, key), handle[f'layers/counts/{key}']
            if saved.shape != array.shape:
                raise ValueError('Original X/counts structure differs')
            for start in range(0, len(array), 2_000_000):
                np.testing.assert_array_equal(array[start:start + 2_000_000], saved[start:start + 2_000_000])
    counts = preparation._raw_counts(raw)
    del raw
    rows = expected.source_row.to_numpy()
    if not obs.index[rows].equals(expected.index):
        raise ValueError('Frozen cell/source mapping differs')
    selected_obs = obs.iloc[rows].copy()
    selected_obs['parent_corpus_row'] = rows
    selected_obs['pretrain_v3_split'] = expected.split
    selected = counts[rows].copy()
    a = ad.AnnData(X=selected, obs=selected_obs, var=var)
    a.layers['counts'] = selected.copy()
    a.uns['v3_source_json'] = json.dumps(dict(parent=cfg['parent_corpus'], parent_sha256=digest,
        tissues=cfg['tissues'], split_source=cfg['parent_prepared'] + '/split_manifest.tsv.gz'))
    a.write_h5ad(temporary, compression='gzip')
    temporary.replace(destination)
    print('Wrote', destination, a.shape, flush=True)


def invoke(module, config):
    original = sys.argv
    sys.argv = [module.__file__, '--config', str(config)]
    try:
        module.main()
    finally:
        sys.argv = original


def verify_roles(cfg):
    previous = pd.read_csv(ROOT / cfg['parent_prepared'] / 'split_manifest.tsv.gz', sep='\t', index_col='cell_id')
    current = pd.read_csv(ROOT / cfg['output'] / 'split_manifest.tsv.gz', sep='\t', index_col='cell_id')
    expected = previous.loc[previous.tissue.isin(cfg['tissues'])]
    if not current.index.equals(expected.index) or not current.split.equals(expected.split):
        raise ValueError('Four-tissue split differs from original cell roles')
    old_genes = pd.read_csv(ROOT / cfg['parent_prepared'] / 'gene_vocab.tsv', sep='\t')
    genes = pd.read_csv(ROOT / cfg['output'] / 'gene_vocab.tsv', sep='\t')
    np.testing.assert_array_equal(genes.gene_id, old_genes.gene_id)
    np.testing.assert_array_equal(genes.token_id, old_genes.token_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'corpus', 'statistics', 'tokens', 'input'], default='all', nargs='?')
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if args.stage in ('all', 'corpus'):
        subset_corpus(cfg)
    prepared = ROOT / cfg['output']
    if args.stage in ('all', 'statistics'):
        if not (prepared / 'validation.json').exists():
            invoke(preparation, CONFIG)
        if json.loads((prepared / 'config.json').read_text()) != cfg:
            raise ValueError('Existing v3 preparation uses different settings')
        verify_roles(cfg)
    if args.stage in ('all', 'tokens'):
        token_cfg = json.loads(TOKEN_CONFIG.read_text())
        if not (ROOT / token_cfg['output'] / 'validation.json').exists():
            invoke(tokenization, TOKEN_CONFIG)
        report = ROOT / token_cfg['report'] / 'report.md'
        # The legacy tokenizer report has a historical cohort sentence; replace
        # only that sentence in this new output, keeping shared computed tables.
        text = report.read_text().replace('Scratch v0.3', 'Scratch v3')
        historical = '输入为冻结的113,159细胞含BM语料和95/2.5/2.5细胞划分，6研究、18文库、7组织。'
        text = text.replace(historical, '输入为96,426细胞四组织语料；沿用原Train/Validation/Test细胞角色，6研究、4组织。')
        report.write_text(text)
    if args.stage in ('all', 'input'):
        input_cfg = json.loads(INPUT_CONFIG.read_text())
        if not (ROOT / input_cfg['fixed_eval_masks'] / 'config.json').exists():
            pipeline.DEFAULT_CONFIG = INPUT_CONFIG
            pipeline.main()
    print('V3 data stage completed:', args.stage, flush=True)


if __name__ == '__main__':
    main()
