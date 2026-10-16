import csv, json, re
from pathlib import Path

root = Path(__file__).resolve().parents[2].parent / "ILC2_rotation_proposal/scbasecount_audit"
def read(name):
    with (root / name).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))

def unique_experiments(rows):
    merged = {}
    for r in rows:
        key = r['experiment_accession']
        if key not in merged:
            merged[key] = dict(r)
            merged[key]['runs'] = set()
        merged[key]['runs'].add(r['run_accession'])
    for r in merged.values():
        r['run_accession'] = ', '.join(sorted(r.pop('runs')))
    return list(merged.values())

priority = read('ILC2_priority_35_samples.tsv')
old = unique_experiments(read('ENA_previous_studies_raw.tsv'))
gut = unique_experiments(read('ENA_GSE124880_raw.tsv'))
series = {}
for p in root.glob('*_series.soft.txt'):
    d = {}
    for line in p.read_text().splitlines():
        if line.startswith('!Series_') and ' = ' in line:
            key, val = line.split(' = ', 1)
            d.setdefault(key[8:], []).append(val)
    assert 'title' in d, f'Not valid GEO SOFT: {p}'
    series[p.name.split('_')[0]] = d

selected = []
for r in old:
    m = re.search(r'GSM(\d+)', r['experiment_title'])
    n = int(m.group(1)) if m else 0
    if ((r['study_accession'] == 'PRJNA397374' and 2733478 <= n <= 2733491)
        or (r['study_accession'] == 'PRJNA561400' and 4041754 <= n <= 4041761)
        or r['study_accession'] in ('PRJNA482472', 'PRJNA593191')
        or (r['study_accession'] == 'PRJNA545529' and 4192253 <= n <= 4192257)):
        selected.append(r)
selected.extend(r for r in gut if int(re.search(r'GSM(\d+)', r['experiment_title']).group(1)) >= 3557852)
coverage = read('priority_study_coverage.tsv')
missing = [r for r in coverage if r['experiment_accession'] == 'SRX27568893']
assert len(priority) == 35 and sum(int(r['obs_count']) for r in priority) == 216236
assert len(selected) == 254 and len(missing) == 1
assert len({r['experiment_accession'] for r in selected + priority + missing}) == 290
out = {'priority': priority, 'studies9': read('ILC2_priority_9_studies.tsv'),
       'previous_selected': selected, 'missing': missing, 'series': series,
       'provenance': json.loads((root / 'query_provenance.json').read_text())}
(root / 'workbook_inputs.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps({'selected_records': 290, 'scBaseCount_RNA': 35, 'observations': 216236,
                  'GEO_series_snapshots': len(series)}, ensure_ascii=False))
