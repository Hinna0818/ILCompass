"""Verify Scanpy on one real library and GeneCompass on a small synthetic model."""
from pathlib import Path
import importlib, importlib.metadata, json, os, platform, subprocess, sys

PROJECT = Path(__file__).resolve().parents[2]
HERE = PROJECT.parent / "ILC2_rotation_proposal/environment"
REPO = HERE.parent / 'software' / 'GeneCompass'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/ilc2_model_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/ilc2_model_numba')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
import scanpy as sc
import anndata as ad
import torch
from transformers import BertConfig
import genecompass
sc.settings.n_jobs = 2

report = {'python': sys.version, 'executable': sys.executable,
          'platform': platform.platform(), 'genecompass_path': genecompass.__file__,
          'genecompass_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
          'cuda_available': torch.cuda.is_available(), 'mps_built': torch.backends.mps.is_built(),
          'mps_available': torch.backends.mps.is_available()}
packages = ['scanpy','anndata','numpy','scipy','pandas','h5py','torch','torchvision','transformers',
            'accelerate','datasets','pyarrow','igraph','leidenalg','scrublet','ipykernel','jupyterlab',
            'hyperopt','loompy','ray','tdigest','scikit-misc']
report['versions'] = {p: importlib.metadata.version(p) for p in packages}
for module in ['torchvision','hyperopt','loompy','ray','tdigest','scrublet','igraph','leidenalg']:
    importlib.import_module(module)
assert Path(sys.prefix).name == 'ilc2_model'
assert sys.flags.no_user_site, 'Use conda run/activate or python -s to isolate user packages'
assert Path(genecompass.__file__).resolve().is_relative_to(REPO)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

# The old 10x gzip/prefix layout is read explicitly to avoid features.tsv assumptions.
data = PROJECT / 'data/GSE117567_RAW'
prefix = 'GSM3303970_BAM_'
genes = pd.read_csv(data / (prefix + 'genes.tsv.gz'), sep='\t', header=None, names=['gene_id','gene_symbol'])
barcodes = pd.read_csv(data / (prefix + 'barcodes.tsv.gz'), sep='\t', header=None)[0].astype(str)
x = mmread(data / (prefix + 'matrix.mtx.gz')).T.tocsr()
a = ad.AnnData(x, obs=pd.DataFrame(index='GSM3303970:' + barcodes), var=genes.set_index('gene_id'))
assert a.shape == (4990,27998) and a.obs_names.is_unique and a.var_names.is_unique
a.var['mt'] = a.var.gene_symbol.str.startswith('mt-')
sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=[], inplace=True)
assert float(a.obs.total_counts.median()) == 2944
assert float(a.obs.n_genes_by_counts.median()) == 1298
report['scanpy_real_matrix'] = {'shape': list(a.shape), 'median_UMI': 2944, 'median_genes': 1298, 'QC_metrics': 'passed'}
del a, x

# Check preprocessing, neighbor graph, UMAP and Leiden on synthetic data only.
rng = np.random.default_rng(42)
s = ad.AnnData(sp.csr_matrix(rng.poisson(np.linspace(.1,10,512), (128,512)).astype(np.float32)))
sc.pp.highly_variable_genes(s, flavor='seurat_v3', n_top_genes=100)
assert int(s.var.highly_variable.sum()) == 100
sc.pp.normalize_total(s)
sc.pp.log1p(s)
sc.pp.pca(s, n_comps=10)
sc.pp.neighbors(s, n_neighbors=10, n_pcs=10)
sc.tl.leiden(s, flavor='igraph', directed=False, n_iterations=2, random_state=42)
sc.tl.umap(s, random_state=42)
assert s.obsm['X_umap'].shape == (128,2) and s.obs.leiden.notna().all()
report['scanpy_synthetic_pipeline'] = 'Seurat v3 HVG/PCA/neighbors/Leiden/UMAP passed'

torch.set_num_threads(2)
config = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                    intermediate_size=64, max_position_embeddings=32, hidden_dropout_prob=0,
                    attention_probs_dropout_prob=0, pad_token_id=0,
                    use_values=True, use_promoter=True, use_co_exp=True, use_gene_family=True,
                    use_peca_grn=True, use_cls_token=True, warmup_steps=2, emb_warmup_steps=2)
knowledge = {k: torch.randn(64,8) for k in ['promoter','co_exp','gene_family','peca_grn']}
knowledge['homologous_gene_human2mouse'] = {2: 3, 3: 2}
model = genecompass.BertForMaskedLM(config=config, knowledges=knowledge)
ids = torch.randint(2,64,(2,16))
values = torch.rand(2,16)
mask = torch.ones_like(ids)
species = torch.ones(2,1,dtype=torch.long)
labels = torch.full_like(ids,-100)
labels[:,2] = ids[:,2]
masked_ids = ids.clone()
masked_ids[:,2] = 1
report['genecompass_smoke'] = {}
for device in ['cpu'] + (['mps'] if report['mps_available'] else []):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    result = model(input_ids=masked_ids.to(device), values=values.to(device),
                   attention_mask=mask.to(device), species=species.to(device),
                   labels=labels.to(device), labels_values=values.to(device))
    assert result.logits.shape == (2,16,64) and torch.isfinite(result.loss).item()
    result.loss.backward()
    assert model.bert.embeddings.word_embeddings.weight.grad is not None
    optimizer.step()
    model.eval()
    with torch.no_grad():
        cell_embedding = model.bert(input_ids=ids.to(device), values=values.to(device),
                                    attention_mask=mask.to(device), species=species.to(device)).last_hidden_state[:,0,:]
    assert cell_embedding.shape == (2,32) and torch.isfinite(cell_embedding).all().item()
    model.train()
    report['genecompass_smoke'][device] = {'forward_backward_optimizer': 'passed', 'cell_embedding': 'passed',
                                         'loss': result.loss.detach().cpu().item()}

report['checkpoint_tested'] = False
report['prior_embedding_lfs_pointers'] = [str(p.relative_to(REPO)) for p in (REPO / 'prior_knowledge').rglob('*.pickle')
                                        if p.stat().st_size < 200 and p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1')]
report['note'] = 'Small random model with synthetic prior tensors; no pretrained checkpoint or real-data training.'
(HERE / 'environment-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('ALL CHECKS PASSED', flush=True)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
