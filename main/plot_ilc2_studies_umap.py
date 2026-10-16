"""两个研究各自及联合UMAP：只读当前外周counts，不重新QC或改写训练对象。

输入：data/train_data/GSE117567.h5ad（strict_v1）与GSE153748.h5ad（balanced_v1）。
输出：results/ILC2_studies_umap/ 的PNG/PDF、坐标、HVG、交叉表与参数，无重复大矩阵。
运行：conda activate ilc2_model; python main/plot_ilc2_studies_umap.py
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/ilc2_model_numba')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/ilc2_model_mpl')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from plot_gse117567_umap import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ILC2_studies_umap'
STUDIES = ['GSE117567', 'GSE153748']
TISSUE_COLORS = {'Lung':'#4477AA', 'SI':'#EE6677', 'LI':'#66CCEE',
                 'Skin':'#AA3377', 'Fat':'#CCBB44', 'Pancreas':'#228833'}
STUDY_COLORS = {'GSE117567':'#4477AA', 'GSE153748':'#EE7733'}
PARAMS = {'target_sum':10000, 'hvg_flavor':'seurat', 'n_top_genes':2500,
          'n_pcs':30, 'n_neighbors':15, 'min_dist':0.5, 'leiden_resolution':0.6,
          'seed':0, 'batch_correction':'none', 'downsampling':False,
          'joint_hvg_batch_key':'study', 'normalization':'total_UMI_then_log1p_no_scaling'}


def embed(raw: ad.AnnData, name: str, joint: bool = False) -> pd.DataFrame:
    print(f'{name}: {raw.n_obs:,} cells; normalize / HVG / PCA / neighbors', flush=True)
    a = ad.AnnData(raw.X.astype(np.float32), obs=raw.obs.copy(), var=raw.var.copy())
    sc.pp.normalize_total(a, target_sum=PARAMS['target_sum'])
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor='seurat', n_top_genes=PARAMS['n_top_genes'],
                              batch_key='study' if joint else None)
    a.var.loc[a.var.highly_variable].to_csv(OUT / f'{name}_hvg.tsv',sep='\t')
    a = a[:, a.var.highly_variable].copy()
    gc.collect()
    sc.pp.pca(a, n_comps=PARAMS['n_pcs'], svd_solver='arpack', random_state=0)
    sc.pp.neighbors(a, n_neighbors=PARAMS['n_neighbors'], n_pcs=PARAMS['n_pcs'], random_state=0)
    print(f'{name}: UMAP / Leiden', flush=True)
    sc.tl.umap(a, min_dist=PARAMS['min_dist'], random_state=0)
    sc.tl.leiden(a, resolution=PARAMS['leiden_resolution'], flavor='igraph',
                 directed=False, n_iterations=2, random_state=0)
    xy = a.obsm['X_umap']
    assert xy.shape == (raw.n_obs,2) and np.isfinite(xy).all()
    coords = pd.DataFrame(xy,index=a.obs_names,columns=['UMAP1','UMAP2']).join(a.obs)
    coords.index.name = 'cell_id'
    assert coords.index.equals(raw.obs_names) and coords.leiden.notna().all()
    path = OUT / f'{name}_coordinates.tsv.gz'
    coords.to_csv(path,sep='\t')
    reloaded = pd.read_csv(path,sep='\t',index_col=0)
    assert reloaded.index.equals(coords.index) and np.allclose(reloaded[['UMAP1','UMAP2']],xy)
    pd.crosstab(coords.leiden, [coords.study,coords.tissue]).to_csv(OUT / f'{name}_cluster_counts.tsv',sep='\t')
    draw(coords,name,joint)
    return coords


def draw(coords: pd.DataFrame, name: str, joint: bool) -> None:
    keys = ['study','tissue','leiden'] if joint else ['tissue','leiden']
    titles = {'study':'Study', 'tissue':'Tissue', 'leiden':'Leiden clusters (0.6)'}
    fig, axes = plt.subplots(1,len(keys),figsize=(5.7*len(keys),5.3),layout='constrained')
    order = np.random.default_rng(0).permutation(len(coords))
    points = coords.iloc[order]
    for ax,key in zip(axes,keys):
        values = points[key].astype(str)
        if key == 'tissue':
            colors = {k:v for k,v in TISSUE_COLORS.items() if k in set(values)}
        elif key == 'study':
            colors = STUDY_COLORS
        else:
            labels = sorted(set(values),key=int)
            colors = {k:plt.get_cmap('tab20')(i%20) for i,k in enumerate(labels)}
        assert set(values) <= set(colors)
        ax.scatter(points.UMAP1,points.UMAP2,c=[colors[v] for v in values],s=2.2,
                   linewidths=0,alpha=.8,rasterized=True)
        if key == 'leiden':
            for label in colors:
                xy = points.loc[values.eq(label),['UMAP1','UMAP2']].median()
                ax.text(*xy,label,ha='center',va='center',fontsize=10,
                        bbox={'boxstyle':'round,pad=.15','fc':'white','ec':'none','alpha':.85})
        else:
            handles = [plt.Line2D([],[],marker='o',linestyle='',color=color,markersize=5,
                        label=f'{label} ({int(values.eq(label).sum()):,})') for label,color in colors.items()]
            ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=9)
        ax.set(title=titles[key],xlabel='UMAP1',ylabel='UMAP2',xticks=[],yticks=[])
        ax.spines[['top','right']].set_visible(False)
    version = '117567: strict_v1; 153748: balanced_v1' if joint else str(coords.qc_version.iloc[0])
    fig.suptitle(f'{name} | n={len(coords):,} | {version}',fontsize=12)
    fig.supxlabel('No batch correction | Peripheral cells only | Independent embeddings have independent cluster labels',fontsize=9)
    fig.savefig(OUT / f'{name}_umap.png',dpi=180,facecolor='white')
    fig.savefig(OUT / f'{name}_umap.pdf',facecolor='white')
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    sources = {s:ROOT/f'data/train_data/{s}.h5ad' for s in STUDIES}
    hashes = {s:file_hash(p) for s,p in sources.items()}
    resource = {'available_memory_gib':round(psutil.virtual_memory().available/2**30,2),
                'logical_cores':psutil.cpu_count(), 'strategy':'sequential_sparse_no_redundant_h5ad'}
    parts, summary = [], []
    for study,path in sources.items():
        source = ad.read_h5ad(path,backed='r')
        metadata = source.uns['metadata']
        version = metadata.get('version',metadata.get('dataset_version','unknown'))
        obs = source.obs[['sample_id','tissue','n_genes_by_counts','total_counts','pct_counts_mt']].copy()
        obs['tissue_original'] = obs.tissue.astype(str)
        obs['tissue'] = obs.tissue.astype(str).replace({'Gut':'SI'})
        obs['study'] = study
        obs['qc_version'] = version
        raw = ad.AnnData(source.X[:],obs=obs,var=source.var[['gene_symbol']].copy())
        source.file.close()
        assert raw.obs_names.is_unique and raw.var_names.is_unique and not raw.obs.tissue.eq('BM').any()
        assert np.issubdtype(raw.X.dtype,np.integer) and (raw.X.data>=0).all()
        assert np.array_equal(raw.X.sum(axis=1).A1,raw.obs.total_counts)
        if parts:
            assert raw.var_names.equals(parts[0].var_names), '共同基因顺序必须一致，不能静默丢弃不同feature'
        coordinates = embed(raw,study)
        summary.append({'embedding':study,'n_cells':raw.n_obs,'n_input_genes':raw.n_vars,
                        'n_clusters':coordinates.leiden.nunique(),'qc_version':version})
        parts.append(raw)
        del source,coordinates
        gc.collect()
    joint = ad.concat(parts,join='inner',merge='same')
    del parts
    gc.collect()
    assert joint.n_obs == sum(s['n_cells'] for s in summary) and joint.obs_names.is_unique
    coords = embed(joint,'Joint',joint=True)
    summary.append({'embedding':'Joint','n_cells':joint.n_obs,'n_input_genes':joint.n_vars,
                    'n_clusters':coords.leiden.nunique(),'qc_version':'mixed_versions_exploratory_only'})
    assert all(file_hash(path)==hashes[study] for study,path in sources.items())
    pd.DataFrame(summary).to_csv(OUT/'summary.tsv',sep='\t',index=False)
    payload = {**PARAMS,'sources':{s:{'path':str(p.relative_to(ROOT)),'sha256':hashes[s]} for s,p in sources.items()},
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'software':{n:importlib.metadata.version(n) for n in ['scanpy','anndata','numpy','matplotlib']},
               'resources':resource,'input_files_unchanged':True,'coordinates_reloaded_checked':True,
               'tissue_mapping':{'GSE117567: Gut':'SI, source metadata confirms small-intestine lamina propria'},
               'qc_version_warning':'GSE117567 strict_v1 and GSE153748 balanced_v1; not a harmonized training merge'}
    (OUT/'parameters.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(f'Figures and coordinates saved: {OUT}',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
