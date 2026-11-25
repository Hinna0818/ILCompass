"""GSE189285 within-tissue split, Train-selected panels and frozen Scratch inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from input_pipeline import ROOT, JointMLMCollator
from model import ILC2Transformer, ModelConfig, MODEL_INPUTS
from train import frozen_inputs

sys.path.append(str(ROOT / 'src/downstream/drugpred'))
import train_il33_mixed_delta as task
import train_il33_bidirectional_tissue as source
from train_il33_external_delta import new_roles
from token_data import verify_cell

CONFIG = ROOT / 'scratch/configs/il33_within_tissue_v1.json'
METHODS = ('Scratch_v1', 'ExpressionMLP')


def configuration():
    return json.loads(CONFIG.read_text())


def sources(cfg):
    return json.loads((ROOT / cfg['source_config']).read_text())['tissues']


def paths(cfg, tissue):
    return ROOT / cfg['data_directory'] / tissue, ROOT / cfg['result_directory'] / tissue


def encode_control_rows(counts, gene_ids, vocab, medians, max_length):
    """The shared encoder sorts ties by CSR column order, so order by frozen token ID."""
    order = np.argsort(np.asarray(gene_ids, dtype=str), kind='stable')
    counts = counts[:, order].tocsr()
    counts.sort_indices()
    gene_ids = pd.Index(gene_ids[order])
    loc = vocab.gene_id.map(pd.Series(np.arange(len(gene_ids)), index=gene_ids)).to_numpy()
    measured = ~pd.isna(loc)
    cols = loc[measured].astype(int)
    token_col = np.full(len(gene_ids), -1, dtype=np.int32)
    token_col[cols] = vocab.token_id.to_numpy()[measured]
    med = np.ones(len(gene_ids), dtype=float)
    med[cols] = medians.median_positive_cp10k.to_numpy()[measured]
    lengths = np.diff(counts[:, cols].indptr)
    if (lengths == 0).any() or lengths.max() > max_length:
        raise ValueError('Empty input or positional capacity exceeded; do not silently truncate')
    ids, values, actual = task.encode(counts, token_col, med, int(lengths.max()), 'cell_norm')
    np.testing.assert_array_equal(actual, lengths)
    for i, n in enumerate(actual):
        verify_cell(ids[i, :n], values[i, :n], counts[i], token_col, med, 2, 10000)
    return ids, values, actual


def prepare(cfg):
    base = ROOT / cfg['data_directory']
    base.mkdir(parents=True, exist_ok=True)
    snapshot = base / 'config.json'
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != cfg:
            raise ValueError('Existing data version has different settings')
        if (base / 'prepare_complete.json').exists():
            print('PREPARE already complete', flush=True)
            return
    task.atomic_json(snapshot, cfg)
    prepared = ROOT / cfg['prepared']
    corpus = pd.read_csv(prepared / 'split_manifest.tsv.gz', sep='\t', usecols=['cell_id', 'study'])
    if corpus.study.eq('GSE189285').any():
        raise ValueError('GSE189285 already present in pretraining corpus')
    functional = pd.read_csv(prepared / 'functional_gene_sets.csv')
    vocab = pd.read_csv(prepared / 'gene_vocab.tsv', sep='\t')
    split_counts, records = [], {}
    seen = set(corpus.cell_id)
    for tissue in sources(cfg):
        data, out = paths(cfg, tissue)
        data.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
        a, meta = source.load_tissue({'tissues': sources(cfg)}, tissue)
        if seen.intersection(meta.index):
            raise ValueError('Overlapping cell identities across sources or pretraining')
        seen.update(meta.index)
        meta['split_stratum'] = meta[['study', 'tissue', 'library_id', 'condition']].astype(str).agg('|'.join, axis=1)
        roles = task.assign_splits(meta, cfg['split_fractions'], cfg['split_seed'], labels=task.ROLES)
        meta = new_roles(meta, roles)
        background = f'GSE189285_{tissue}'
        var = a.var[['gene_symbol']].copy()
        var['present_measured'] = True
        panel = task.select_panel(a.X, var, meta, [background], ['measured'], functional.gene_id,
                                  cfg['minimum_train_detection'])
        panel['in_encoder_vocab'] = panel.gene_id.isin(vocab.gene_id)
        panel.to_csv(data / 'gene_panel.tsv', sep='\t', index=False)
        meta.to_csv(data / 'cell_manifest.tsv.gz', sep='\t', index_label='cell_id')
        for role in task.ROLES:
            folder = data / role; folder.mkdir(exist_ok=True)
            rows = meta.split.eq(role).to_numpy()
            m = meta.loc[rows]
            if set(m.condition) != {'PBS', 'IL33'}:
                raise ValueError('Each split needs both conditions in the same tissue')
            m.to_csv(folder / 'metadata.tsv.gz', sep='\t', index_label='cell_id')
            y = task.normalized(a[rows], panel.gene_id)
            if not np.isfinite(y).all():
                raise ValueError('Missing measured output gene; do not pad absent features as zero')
            np.save(folder / 'y.npy', y)
            if role == 'train':
                statistics = task.training_statistics(y, m, panel, [background], out / 'training_DEG')
                np.savez_compressed(folder / 'statistics.npz', **statistics)
            for condition, n in m.condition.value_counts().items():
                split_counts.append(dict(tissue=tissue, split=role, condition=condition, cells=int(n)))
        manifest = dict(config=cfg, tissue=tissue, background=background, cells=len(meta),
            source_sha256={p: task.file_hash(ROOT / p) for p in sources(cfg)[tissue].values()},
            raw_UMI=int(a.X.sum(dtype=np.int64)), output_genes=len(panel),
            functional_genes=int(panel.functional.sum()), in_encoder_vocab=int(panel.in_encoder_vocab.sum()),
            panel_sha256=task.file_hash(data / 'gene_panel.tsv'),
            cell_map_sha256={role: task.file_hash(data / role / 'metadata.tsv.gz') for role in task.ROLES},
            training_statistics_sha256=task.file_hash(data / 'train/statistics.npz'),
            panel_selection='Only Train detection >=1% in PBS or IL33, plus predefined functional genes',
            expression='log1p(CP10K), full original measured GEX denominator',
            limitation='Internal cell holdout; one pooled library per condition and tissue, no paired cells')
        task.atomic_json(data / 'prepare_manifest.json', manifest)
        records[tissue] = manifest
        print(f'PREPARED {tissue}: {len(meta)} cells, {len(panel)} output genes', flush=True)
    result = ROOT / cfg['result_directory']; result.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_counts).to_csv(result / 'split_counts.csv', index=False)
    task.atomic_json(base / 'prepare_complete.json', records)


def encode(cfg):
    checkpoint = torch.load(ROOT / cfg['checkpoint'], map_location='cpu', weights_only=False)
    if checkpoint['config'].get('tissue_auxiliary', False):
        raise ValueError('This benchmark is explicitly for v1')
    input_cfg = checkpoint['input_config']
    prepared, provenance, _ = frozen_inputs(input_cfg)
    if provenance != checkpoint['provenance'] or prepared != ROOT / cfg['prepared']:
        raise ValueError('Frozen pretraining configuration/data mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full frozen-encoder extraction')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ILC2Transformer(ModelConfig(**checkpoint['model_config'])).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    vocab = pd.read_csv(prepared / 'gene_vocab.tsv', sep='\t')
    medians = pd.read_csv(prepared / 'gene_medians_train.tsv', sep='\t')
    if not vocab.gene_id.equals(medians.gene_id):
        raise ValueError('Frozen gene median order mismatch')
    norm = json.loads((prepared / 'config.json').read_text())
    collator = JointMLMCollator({**norm, 'mask_probability': 0}, input_cfg, len(vocab), training=False)
    for tissue in sources(cfg):
        data, _ = paths(cfg, tissue)
        if (data / 'encoding.json').exists():
            saved = json.loads((data / 'encoding.json').read_text())
            if saved['checkpoint_sha256'] != task.file_hash(ROOT / cfg['checkpoint']):
                raise ValueError('Frozen feature checkpoint changed')
            print(f'ENCODING {tissue} cache reused', flush=True); continue
        manifest = json.loads((data / 'prepare_manifest.json').read_text())
        for p, digest in manifest['source_sha256'].items():
            if task.file_hash(ROOT / p) != digest:
                raise ValueError('Raw source changed since preparation')
        a, _ = source.load_tissue({'tissues': sources(cfg)}, tissue)
        meta = pd.read_csv(data / 'cell_manifest.tsv.gz', sep='\t', index_col='cell_id')
        np.testing.assert_array_equal(a.obs_names, meta.index)
        control_rows = np.flatnonzero(meta.condition.eq('PBS'))
        loc = a.var_names.get_indexer(vocab.gene_id)
        lengths = np.diff(a.X[control_rows][:, loc[loc >= 0]].tocsr().indptr)
        order = np.argsort(lengths, kind='stable')
        h = np.empty((len(control_rows), model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for left in range(0, len(order), cfg['encoding_batch_size']):
                idx = order[left:left + cfg['encoding_batch_size']]
                rows = control_rows[idx]
                ids, values, n = encode_control_rows(a.X[rows], a.var_names, vocab, medians, model.config.max_length)
                samples = [dict(gene_ids=ids[i, :n[i]], values=values[i, :n[i]], cell_id=meta.index[row],
                    source_row=int(row), sample_index=int(idx[i]), epoch=0, split='test') for i, row in enumerate(rows)]
                batch = collator(samples)
                encoded = model(**{key: batch[key].cuda() for key in MODEL_INPUTS})
                h[idx] = encoded['cell_embedding'].cpu().numpy()
                if left % 400 == 0:
                    print(f'ENCODE {tissue} PBS {min(left + len(idx), len(order))}/{len(order)}', flush=True)
        if not np.isfinite(h).all():
            raise ValueError('Nonfinite frozen features')
        features = {}
        cm = meta.iloc[control_rows]
        for role in task.ROLES:
            np.save(data / role / 'Scratch_v1_h.npy', h[cm.split.eq(role)])
            features[role] = task.file_hash(data / role / 'Scratch_v1_h.npy')
        task.atomic_json(data / 'encoding.json', dict(checkpoint=cfg['checkpoint'],
            checkpoint_sha256=task.file_hash(ROOT / cfg['checkpoint']), checkpoint_epoch=checkpoint['state']['epoch'],
            frozen_pretraining_provenance=provenance, features_sha256=features, control_cells=len(cm), dimensions=h.shape[1],
            max_nonzero_length=int(lengths.max()), median_nonzero_length=float(np.median(lengths)),
            all_nonzero=True, masking=False, precision='FP32, TF32 off', pooling='valid-token mean', optimizer_updates=0))


def dataset(cfg, tissue, role, method):
    data, _ = paths(cfg, tissue)
    manifest = json.loads((data / 'prepare_manifest.json').read_text())
    if manifest['config'] != cfg or task.file_hash(data / 'gene_panel.tsv') != manifest['panel_sha256']:
        raise ValueError('Data/panel configuration changed')
    m, y = task.load_expression_split(data, manifest, role)
    controls = m.condition.eq('PBS').to_numpy()
    h = np.load(data / role / 'Scratch_v1_h.npy') if method == 'Scratch_v1' else np.array(y[controls])
    if h.shape[0] != controls.sum() or not np.isfinite(h).all():
        raise ValueError('Feature row alignment or finite-value failure')
    return m, y, h
