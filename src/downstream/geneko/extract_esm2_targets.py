"""Frozen canonical mouse protein conditions for the existing three-KO benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REFERENCES = ROOT / 'data/references/esm2_three_KO_v1'
OUTPUT = ROOT / 'results/ILC2_three_KO_ESM2_v1'
CHECKPOINT = ROOT / 'model/esm2_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt'
TARGETS = [
    ('Piezo1', 'ENSMUSG00000014444', 'E2JF22'),
    ('Hilpda', 'ENSMUSG00000043421', 'Q9JLS0'),
    ('Nmur1', 'ENSMUSG00000026237', 'O55040'),
]
WINDOW, OVERLAP = 1022, 256


def cached_json(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            value = json.load(response)
        path.write_text(json.dumps(value, indent=2) + '\n')
    return json.loads(path.read_text())


def windows(length, size=WINDOW, overlap=OVERLAP):
    if length <= 0 or not 0 <= overlap < size:
        raise ValueError('Invalid sequence length/window/overlap')
    for start in range(0, length, size - overlap):
        end = min(start + size, length)
        yield start, end
        if end == length:
            break


def pool_windows(length, representations):
    """Each amino acid has equal final weight, including window overlaps."""
    total = np.zeros((length, representations[0][1].shape[1]), dtype=np.float32)
    coverage = np.zeros(length, dtype=np.int32)
    for start, values in representations:
        total[start:start + len(values)] += values
        coverage[start:start + len(values)] += 1
    if (coverage == 0).any():
        raise ValueError('Sequence is not fully covered')
    total /= coverage[:, None]
    return total.mean(axis=0), coverage


def prepare():
    REFERENCES.mkdir(parents=True, exist_ok=True)
    records = []
    for symbol, gene_id, accession in TARGETS:
        url = f'https://rest.uniprot.org/uniprotkb/{accession}.json'
        entry = cached_json(url, REFERENCES / f'{accession}.json')
        gene_names = {g['geneName']['value'] for g in entry.get('genes', []) if 'geneName' in g}
        if (entry['primaryAccession'] != accession or entry['organism']['taxonId'] != 10090
                or entry['entryType'] != 'UniProtKB reviewed (Swiss-Prot)' or symbol not in gene_names):
            raise ValueError(f'Unexpected protein identity for {symbol}')
        refs = entry.get('uniProtKBCrossReferences', [])
        ensembl_ids = {p['value'].split('.')[0] for r in refs if r['database'] == 'Ensembl'
                       for p in r.get('properties', []) if p['key'] == 'GeneId'}
        if gene_id in ensembl_ids:
            mapping = 'UniProt Ensembl GeneId cross-reference'
        elif symbol == 'Piezo1':
            # This reviewed entry currently lacks an Ensembl cross-reference.
            cross = cached_json(
                f'https://rest.ensembl.org/xrefs/id/{gene_id}?content-type=application/json;all_levels=1',
                REFERENCES / f'{gene_id}_xrefs.json')
            protein_mgi = {r['id'] for r in refs if r['database'] == 'MGI'}
            gene_mgi = {r['primary_id'] for r in cross
                        if r.get('dbname') == 'MGI' and r.get('display_id') == symbol}
            if not protein_mgi & gene_mgi:
                raise ValueError('Piezo1 Ensembl/UniProt MGI mapping mismatch')
            mapping = 'Ensembl and UniProt shared ' + ','.join(sorted(protein_mgi & gene_mgi))
        else:
            raise ValueError(f'Missing Ensembl mapping for {symbol}')
        sequence = entry['sequence']['value']
        if len(sequence) != entry['sequence']['length'] or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError(f'Invalid canonical sequence: {symbol}')
        records.append(dict(symbol=symbol, gene_id=gene_id, protein_id=accession,
                            species='Mus musculus', taxon_id=10090, length=len(sequence),
                            sequence=sequence, source_url=url, mapping_evidence=mapping,
                            selection='UniProt reviewed canonical protein; not all isoforms',
                            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                            uniprot_entry_audit=entry.get('entryAudit', {})))
    (REFERENCES / 'target_proteins.fasta').write_text(''.join(
        f">{r['gene_id']}|{r['symbol']}|{r['protein_id']}\n{r['sequence']}\n" for r in records))
    (REFERENCES / 'mapping.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def extract(records, device):
    import esm
    import torch
    from importlib.metadata import version

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly if needed')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'target_embeddings.npz').exists():
        raise FileExistsError('Embeddings already exist; preserve this version rather than overwrite it')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(CHECKPOINT))
    model = model.eval().requires_grad_(False).to(device)
    convert = alphabet.get_batch_converter()
    embeddings, summaries = [], []
    with torch.inference_mode():
        for r in records:
            parts = []
            for start, end in windows(r['length']):
                _, _, tokens = convert([(r['symbol'], r['sequence'][start:end])])
                output = model(tokens.to(device), repr_layers=[33], return_contacts=False)
                # Exclude BOS/EOS; one sequence per forward means no padding residues.
                values = output['representations'][33][0, 1:1 + end - start].cpu().numpy().copy()
                parts.append((start, values))
                del output
                print(f"{r['symbol']} residues {start + 1}-{end}/{r['length']}", flush=True)
            vector, coverage = pool_windows(r['length'], parts)
            if vector.shape != (1280,) or not np.isfinite(vector).all():
                raise ValueError('Invalid ESM2 embedding')
            embeddings.append(vector)
            summaries.append(dict(symbol=r['symbol'], gene_id=r['gene_id'], protein_id=r['protein_id'],
                                  length=r['length'], windows=list(windows(r['length'])),
                                  min_coverage=int(coverage.min()), max_coverage=int(coverage.max()),
                                  embedding_l2=float(np.linalg.norm(vector))))
    np.savez_compressed(OUTPUT / 'target_embeddings.npz',
                        gene_ids=np.array([r['gene_id'] for r in records]),
                        symbols=np.array([r['symbol'] for r in records]),
                        protein_ids=np.array([r['protein_id'] for r in records]),
                        embeddings=np.stack(embeddings))
    (OUTPUT / 'extraction.json').write_text(json.dumps(dict(
        model='esm2_t33_650M_UR50D', layer=33, dimensions=1280, dtype='float32', device=device,
        torch_version=torch.__version__, fair_esm_version=version('fair-esm'),
        checkpoint=str(CHECKPOINT), checkpoint_bytes=CHECKPOINT.stat().st_size,
        references=str(REFERENCES / 'mapping.json'), sequences=[{k: v for k, v in r.items()
            if k != 'sequence'} for r in records], window_size=WINDOW, overlap=OVERLAP,
        pooling='Mean across windows per residue, then equal mean over all residues; exclude BOS/EOS/PAD',
        limitations=['Windowed Piezo1 lacks full-length attention context',
                     'Canonical wild-type protein identity, not a mutant protein or cell-context embedding',
                     'No training data fitted; no functional KO severity encoded'],
        proteins=summaries), indent=2) + '\n')
    print(f'Saved {OUTPUT / "target_embeddings.npz"}: (3, 1280)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'extract', 'all'], default='all')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    records = prepare() if args.stage in ['prepare', 'all'] else json.loads((REFERENCES / 'mapping.json').read_text())
    if args.stage in ['extract', 'all']:
        extract(records, args.device)
