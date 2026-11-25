"""Build GSE202186 cytokine reference sequences and frozen ESM2 embeddings."""

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'data/references/GSE202186_cytokines'
OUT = ROOT / 'data/processed/GSE202186_perturbation_v1/ligands'
WORKBOOK = '41586_2023_6816_MOESM3_ESM.xlsx'
TABLE_URL = ('https://media.springernature.com/full/springer-static/esm/'
             'art%3A10.1038%2Fs41586-023-06816-9/MediaObjects/' + WORKBOOK)
METHODS_URL = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10781646/'
REFSEQ_URL = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'
              'db=protein&id=NP_665836.2&rettype=gp&retmode=xml')
VEGF_ISOFORM_URL = 'https://rest.uniprot.org/uniprotkb/Q00731-2.fasta'
VEGF_FORM = 'https://www.rndsystems.com/products/recombinant-mouse-vegf-164-protein_493-mv'
PSPN_FORM = 'https://resources.rndsystems.com/pdfs/datasheets/2479-ps_cf.pdf'
IL33_FORM = 'https://www.rndsystems.com/products/recombinant-mouse-il-33-protein_3626-ml'

# Explicit names disambiguate the biologically relevant product of a precursor.
NAMED_CHAINS = {
    'C3': ['C3a anaphylatoxin'], 'Hc': ['C5a anaphylatoxin'],
    'Egf': ['Epidermal growth factor'], 'Tgfb1': ['Transforming growth factor beta-1'],
    'Il33': ['Interleukin-33(109-266)'], 'Kitl': ['Soluble KIT ligand'],
    'Csf1': ['Macrophage colony-stimulating factor 1 43 kDa subunit'],
    'Hgf': ['Hepatocyte growth factor alpha chain', 'Hepatocyte growth factor beta chain'],
}
STOICHIOMETRY = {'LT-α1/β2': [1, 2], 'LT-α2/β1': [2, 1]}


def download(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        tmp = path.with_suffix('.part')
        tmp.write_bytes(data)
        tmp.replace(path)


def author_table():
    """The first worksheet contains the 86 conditions and mouse gene symbols."""
    with zipfile.ZipFile(REF / WORKBOOK) as z:
        strings = [''.join(t.text or '' for t in item.iterfind('.//{*}t'))
                   for item in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rows = ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iterfind('.//{*}row')
        records = {}
        for row in rows:
            content = {re.sub(r'\d+$', '', c.attrib['r']): strings[int(c.findtext('{*}v'))]
                       for c in row.findall('{*}c') if c.get('t') == 's'}
            if row.get('r') == '1':
                if (content['B'], content['E']) != ('Cytokine', 'Mouse gene symbol'):
                    raise ValueError('Unexpected Supplementary Table 1 header')
            elif 'B' in content and 'E' in content:
                records[content['B']] = (content['A'], content['C'],
                                         [s.strip() for s in content['E'].split(',')])
    if len(records) != 86:
        raise ValueError('Expected 86 distinct cytokines from Supplementary Table 1')
    return records


def fetch_references():
    """Fetch only author and reviewed Mus musculus records needed by this dataset."""
    REF.mkdir(parents=True, exist_ok=True)
    download(TABLE_URL, REF / WORKBOOK)
    download(REFSEQ_URL, REF / 'NP_665836.2.xml')
    download(VEGF_ISOFORM_URL, REF / 'Q00731-2.fasta')
    genes = sorted({gene for _, _, subunits in author_table().values() for gene in subunits})
    for i in range(0, len(genes), 18):
        destination = REF / f'uniprot_{1+i//18:02d}.json'
        if destination.exists():
            continue
        chunk = genes[i:i+18]
        query = '(' + ' OR '.join(f'gene_exact:{g}' for g in chunk) + ')'
        params = urllib.parse.urlencode({'query': query + ' AND organism_id:10090 AND reviewed:true',
                                         'size': 500, 'format': 'json'})
        with urllib.request.urlopen('https://rest.uniprot.org/uniprotkb/search?' + params,
                                    timeout=120) as response:
            data = json.load(response)
            if response.headers.get('Link'):
                raise ValueError('Unexpected UniProt pagination')
            data['release'] = response.headers.get('X-UniProt-Release')
        destination.write_text(json.dumps(data, ensure_ascii=False) + '\n')
        print(f'{destination.name}: {len(data["results"])} reviewed proteins', flush=True)


def selected_chains(gene, entry):
    """Return named mature chains, the secreted form, or the extracellular proxy."""
    sequences = entry['sequence']['value']
    accession = entry['primaryAccession']
    url = 'https://www.uniprot.org/uniprotkb/' + accession + '/entry'
    chains = [f for f in entry['features'] if f['type'] == 'Chain']
    if gene == 'Vegfa':
        with (REF / 'Q00731-2.fasta').open() as h:
            header = h.readline()
            sequence = ''.join(line.strip() for line in h)
        if ('|Q00731-2|' not in header or 'OX=10090 GN=Vegfa' not in header
                or len(sequence) != 190 or (sequence[26], sequence[-1]) != ('A', 'R')):
            raise ValueError('Unexpected mouse VEGF164 reference isoform')
        note = ('The author alias VEGF A165 corresponds to the mouse homolog VEGF164 (Q00731-2). '
                'The mouse product sheet supports the mature reference boundaries.')
        return [(accession + '-2', 27, 190, sequence[26:], 'Mouse VEGF164 mature chain',
                 url + '|' + VEGF_ISOFORM_URL + '|' + VEGF_FORM, note)]
    if gene == 'Pspn':
        if (sequences[60], sequences[-1], len(sequences)) != ('A', 'G', 156):
            raise ValueError('Unexpected reference Persephin coordinates')
        return [(accession, 61, 156, sequences[60:156], 'Mature mouse Persephin',
                 url + '|' + PSPN_FORM, 'Mouse product sheet supports the reference boundaries.')]

    note = 'Mouse UniProt chain annotation.'
    if gene in NAMED_CHAINS:
        features = []
        for name in NAMED_CHAINS[gene]:
            item, = [f for f in chains if f.get('description') == name]
            features.append(item)
        if gene == 'Il33':
            url += '|' + IL33_FORM
            note = 'Selected the supplier-supported 109–266 form. UniProt also annotates full-length and 102–266 forms.'
        elif gene == 'Csf1':
            note = 'Selected the short annotated soluble subunit among multiple processed CSF1 forms.'
        elif gene == 'Hgf':
            note = 'Mature alpha/beta chains from one precursor, one copy each.'
    else:
        soluble = [f for f in chains if re.search(r', (soluble|secreted) form$', f.get('description', ''))]
        extracellular = [f for f in entry['features'] if f['type'] == 'Topological domain'
                         and f.get('description') == 'Extracellular']
        if soluble:
            item, = soluble
        elif extracellular:
            item, = extracellular
            note = 'UniProt extracellular domain proxy. The recombinant extracellular boundaries may differ.'
        else:
            item, = chains
        features = [item]
    answer = []
    for f in features:
        start, end = (f['location'][side]['value'] for side in ['start', 'end'])
        if not (start and end and 1 <= start <= end <= len(sequences)):
            raise ValueError(f'Undefined {gene} chain coordinates')
        answer.append((accession, start, end, sequences[start-1:end], f['description'], url, note))
    return answer


def il17d_refseq():
    entry = ET.parse(REF / 'NP_665836.2.xml').getroot().find('GBSeq')
    features = entry.findall('GBSeq_feature-table/GBFeature')
    quals = [(q.findtext('GBQualifier_name'), q.findtext('GBQualifier_value')) for f in features
             for q in f.findall('GBFeature_quals/GBQualifier')]
    signal, = [f for f in features if f.findtext('GBFeature_key') == 'sig_peptide']
    if (entry.findtext('GBSeq_accession-version') != 'NP_665836.2'
            or not entry.findtext('GBSeq_comment').startswith('VALIDATED REFSEQ:')
            or ('db_xref', 'taxon:10090') not in quals or ('gene', 'Il17d') not in quals
            or signal.findtext('GBFeature_location') != '1..24'):
        raise ValueError('Unexpected IL-17D mouse RefSeq record')
    sequence = entry.findtext('GBSeq_sequence').upper()
    return [('NP_665836.2', 25, len(sequence), sequence[24:], 'Post-signal-peptide reference',
             'https://www.ncbi.nlm.nih.gov/protein/NP_665836.2',
             'Post-signal-peptide segment of a validated NCBI mouse RefSeq. The retrieved UniProt release lacks a reviewed mouse IL-17D entry.')]


def prepare():
    table = author_table()
    with (ROOT / 'results/GSE202186_overview/hashtag_map.tsv').open() as h:
        labels = {r['cyt_display'] for r in csv.DictReader(h, delimiter='\t') if int(r['author_cells']) > 0}
    if labels - {'PBS'} != set(table):
        raise ValueError('The GEO/SCP cytokine labels and Supplementary Table 1 differ')

    primary, aliases = {}, {}
    for path in sorted(REF.glob('uniprot_*.json')):
        for entry in json.loads(path.read_text())['results']:
            if entry['organism']['taxonId'] != 10090 or entry['entryType'] != 'UniProtKB reviewed (Swiss-Prot)':
                raise ValueError(f'Unexpected species or review status in {path}')
            for record in entry.get('genes', []):
                if 'geneName' in record:
                    primary.setdefault(record['geneName']['value'], {})[entry['primaryAccession']] = entry
                for alias in record.get('synonyms', []):
                    aliases.setdefault(alias['value'], {})[entry['primaryAccession']] = entry

    rows, sequences, ligand_chains = [], {}, {}
    for label, (family, other, genes) in sorted(table.items()):
        subunits = []
        for gene in genes:
            if gene == 'Il17d' and gene not in primary and gene not in aliases:
                subunits.extend(il17d_refseq())
            else:
                entries = primary.get(gene) or aliases.get(gene, {})
                if len(entries) != 1:
                    raise ValueError(f'Expected one reviewed mouse protein for {gene}, found {list(entries)}')
                subunits.extend(selected_chains(gene, next(iter(entries.values()))))
        copies = STOICHIOMETRY.get(label, [1] * len(subunits))
        if len(copies) != len(subunits):
            raise ValueError(f'Incomplete stoichiometry for {label}')
        parts, notes, evidence, segments, names, accessions = [], [], [], [], [], []
        for n, (accession, start, end, sequence, name, url, note) in zip(copies, subunits):
            if len(sequence) != end - start + 1 or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
                raise ValueError(f'Invalid reference sequence {accession}:{start}-{end}')
            digest = hashlib.sha256(sequence.encode('ascii')).hexdigest()
            sequences[digest] = sequence
            parts.append((digest, n))
            accessions.append(accession)
            segments.append(f'{accession}:{start}-{end}')
            names.append(name)
            notes.append(note)
            evidence.extend(url.split('|'))
        if label == 'IL-Y':
            notes.append('The authors list p28 and p40. Their encoded combination uses one copy of each reference chain.')
        ligand_chains[label] = parts
        if label in STOICHIOMETRY:
            biological_id = '+'.join(f'{g}*{n}' for g, n in zip(genes, copies))
        else:
            biological_id = '+'.join(genes)
        if label in {'C3a', 'C5a', 'VEGF'}:
            biological_id += '/' + {'C3a': 'C3a', 'C5a': 'C5a', 'VEGF': 'VEGF164'}[label]
        if label == 'HGF':
            biological_id += '/alpha+beta'
        rows.append(dict(cyt_display=label, canonical_id='taxon10090/' + biological_id,
                         input_key='+'.join(f'{digest}*{n}' for digest, n in sorted(parts)),
                         status='reference_proxy', taxon_id=10090, family=family, aliases=other,
                         gene='|'.join(genes), accession='|'.join(accessions), subunits='|'.join(names),
                         reference_segments='|'.join(segments), relative_chain_copies='|'.join(map(str, copies)),
                         author_evidence=TABLE_URL, reference_evidence='|'.join(dict.fromkeys(evidence)),
                         notes=' '.join(dict.fromkeys(notes)), unresolved_reason=''))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'ligand_map.tsv.part').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    (OUT / 'ligand_map.tsv.part').replace(OUT / 'ligand_map.tsv')
    (OUT / 'chain_sequences.fasta').write_text(''.join(
        f'>{segment}|sha256={digest}\n{sequences[digest]}\n'
        for segment, digest in sorted({(s, d) for r in rows for s, (d, _) in
                                      zip(r['reference_segments'].split('|'), ligand_chains[r['cyt_display']])})))
    return sequences, ligand_chains


def embed(sequences, ligand_chains):
    import esm
    import torch
    from importlib.metadata import version
    sys.path.insert(0, str(ROOT / 'src/downstream/geneko'))
    from extract_esm2_targets import CHECKPOINT, pool_windows, windows

    torch.set_num_threads(3)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(CHECKPOINT))
    model.eval().requires_grad_(False)
    convert = alphabet.get_batch_converter()
    vectors = {}
    with torch.inference_mode():
        for i, (key, seq) in enumerate(sequences.items(), 1):
            parts = []
            for start, end in windows(len(seq)):
                _, _, tokens = convert([(key, seq[start:end])])
                out = model(tokens, repr_layers=[33], return_contacts=False)['representations'][33]
                parts.append((start, out[0, 1:1+end-start].numpy().copy()))
            vectors[key], _ = pool_windows(len(seq), parts)
            if i == 1 or i % 10 == 0 or i == len(sequences):
                print(f'ESM2 residue pooling: {i}/{len(sequences)} distinct chains', flush=True)
    labels = list(ligand_chains)
    embeddings = np.stack([np.average(np.stack([vectors[k] for k, _ in ligand_chains[label]]),
                                      weights=[n for _, n in ligand_chains[label]], axis=0)
                           for label in labels]).astype(np.float32)
    if embeddings.shape != (len(labels), 1280) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid ESM2 matrix')
    with (OUT / 'esm2.npz.part').open('wb') as h:
        np.savez_compressed(h, cyt_display=np.array(labels), embeddings=embeddings)
    (OUT / 'esm2.npz.part').replace(OUT / 'esm2.npz')
    meta = dict(model='esm2_t33_650M_UR50D', layer=33, frozen=True, normalization='none',
                tensor_shape=list(embeddings.shape), dtype=str(embeddings.dtype), device='cpu',
                torch_version=torch.__version__, fair_esm_version=version('fair-esm'),
                checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_bytes=CHECKPOINT.stat().st_size,
                unique_chain_sequences=len(sequences), uniprot_release=json.loads((REF/'uniprot_01.json').read_text())['release'],
                protein_pooling='Mean over real amino acid residues after excluding BOS, EOS and PAD.',
                complex_pooling='Mean of constituent chain vectors weighted by relative chain copy number. Repeating identical chains does not change a weighted mean.',
                author_methods=METHODS_URL, author_table=TABLE_URL,
                source_species='Recombinant mouse cytokines as reported by Cui et al.',
                construct_status='Mouse reference proxies. The study identifies the suppliers collectively. Individual order IDs and exact reagent constructs remain unverified.',
                limitations='A sequence mean has no explicit glycosylation, tags, receptor partners, dose, quaternary structure, or subunit-interface representation.')
    (OUT/'extraction.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['fetch', 'prepare', 'all'], default='all')
    stage = parser.parse_args().stage
    if stage == 'fetch':
        fetch_references()
    else:
        records, compositions = prepare()
        if stage == 'all':
            embed(records, compositions)
