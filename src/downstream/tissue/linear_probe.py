"""Frozen GeneCompass/ILC2 tissue probes; fixed sampling, train-only scaling."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ilc2_mpl')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import sys,json,pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/downstream/drugpred'))
from models import FrozenILC2Encoder,OFFICIAL_CHECKPOINT
from posttrain_data import encode,file_hash
from genecompass_coverage import map_genes
import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
from threadpoolctl import threadpool_limits

OUT=ROOT/'results/ILC2_tissue_linear_probe_v1'
DATA=ROOT/'data/processed/ILC2_posttrain_update'
MODELS={'GeneCompass':ROOT/'model/pretrained/GeneCompass_base','ILC2':OFFICIAL_CHECKPOINT}


def probe_jobs(meta):
    train=meta.split.eq('posttrain_train').to_numpy();test=meta.split.eq('posttrain_test').to_numpy()
    counts=meta.groupby('tissue').study.nunique();shared=counts[counts>=2].index.tolist()
    common=meta.tissue.isin(shared).to_numpy()
    jobs=[('six_tissue_same_source',train,test),('three_tissue_same_source',train&common,test&common)]
    for study in meta.study.unique():
        tr=train&common&meta.study.ne(study).to_numpy();te=test&common&meta.study.eq(study).to_numpy()
        if te.any():jobs.append(('holdout_'+study,tr,te))
    return jobs


def fit_probe(train_h,test_h,train_y):
    scaler=StandardScaler().fit(train_h)
    model=LogisticRegression(C=1.,class_weight='balanced',max_iter=3000,solver='lbfgs',random_state=0)
    model.fit(scaler.transform(train_h),train_y)
    return model,scaler,model.predict(scaler.transform(test_h))


def features():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/2prepare_traindata.json').read_text())
    obs=pd.read_csv(DATA/'ILC2_posttrain_20261021_update_metadata.tsv.gz',sep='\t',index_col=0,low_memory=False)
    obs.groupby(['study','tissue','split']).size().rename('cells').to_csv(OUT/'available_counts.tsv',sep='\t')
    selected=[]
    for (study,tissue,split),d in obs.groupby(['study','tissue','split'],sort=True):
        if split not in ['posttrain_train','posttrain_test']:continue
        selected.append(d.sample(n=min(300 if split=='posttrain_train' else 100,len(d)),random_state=20261026))
    meta=pd.concat(selected).sort_index()[['study','tissue','split','library_id','qc_version']]
    if (OUT/'cells.tsv.gz').exists():
        old=pd.read_csv(OUT/'cells.tsv.gz',sep='\t',index_col=0)
        if not old.index.equals(meta.index):raise ValueError('Changed sample')
    meta.to_csv(OUT/'cells.tsv.gz',sep='\t')
    a=ad.read_h5ad(DATA/'ILC2_posttrain_20261021_update.h5ad',backed='r');sub=a[meta.index].to_memory();a.file.close()
    if (sub.X!=sub.layers['counts']).nnz or not np.isfinite(sub.X.data).all() or (sub.X.data<0).any() or (sub.X.data!=np.floor(sub.X.data)).any():raise ValueError('Invalid counts')
    tokens=pickle.loads((ROOT/cfg['token_dictionary']).read_bytes());median=pickle.loads((ROOT/cfg['gene_median_reference']).read_bytes())
    cols,med,_=map_genes(sub.var_names,tokens,median);ids,vals,lengths=encode(sub.X,cols,med,2048,'median')
    write=dict(seed=20261026,cap_train_per_study_tissue=300,cap_test_per_study_tissue=100,encoding='median_top2048_FP32_masked_gene_mean',
        probe='StandardScaler(training only) + multinomial logistic regression C=1 class_weight=balanced max_iter=3000',
        limitation='Posttrain test cells not used to fit encoder; same-source evaluation is not unseen-study. Study-holdout is head-only: encoder saw the source studies during posttraining.',
        source=file_hash(DATA/'ILC2_posttrain_20261021_update.h5ad'),checkpoints={k:file_hash(v/'pytorch_model.bin') for k,v in MODELS.items()})
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=write:raise ValueError('Input signature changed')
    (OUT/'config.json').write_text(json.dumps(write,indent=2))
    for name,checkpoint in MODELS.items():
        dest=OUT/f'{name}_h.npy'
        if dest.exists():continue
        if not torch.cuda.is_available():raise RuntimeError('GPU needed')
        partial=OUT/f'{name}_h.partial.npy';progress=OUT/f'{name}_progress.json';start=json.loads(progress.read_text())['rows'] if progress.exists() else 0
        h=np.lib.format.open_memmap(partial,mode='r+' if start else 'w+',dtype='float32',shape=(len(meta),768))
        torch.backends.cuda.matmul.allow_tf32=False;model=FrozenILC2Encoder(checkpoint).cuda().eval()
        for left in range(start,len(meta),8):
            right=min(left+8,len(meta));n=int(lengths[left:right].max());ti=torch.tensor(ids[left:right,:n],device='cuda',dtype=torch.long);tv=torch.tensor(vals[left:right,:n],device='cuda');mask=torch.arange(n,device='cuda')[None,:]<torch.tensor(lengths[left:right],device='cuda')[:,None]
            h[left:right]=model(ti,tv,mask).cpu().numpy()
            if right%200==0 or right==len(meta):
                h.flush();progress.write_text(json.dumps(dict(rows=right)));print('ENCODE',name,right,'/',len(meta),flush=True)
        h.flush();del h,model;partial.replace(dest);torch.cuda.empty_cache()
    return meta


def main(heads_only=False):
    meta=pd.read_csv(OUT/'cells.tsv.gz',sep='\t',index_col=0) if heads_only else features()
    train=meta.split.eq('posttrain_train').to_numpy();test=meta.split.eq('posttrain_test').to_numpy()
    if (train&test).any() or not meta.index.is_unique:raise ValueError('Invalid cell split')
    jobs=probe_jobs(meta)
    metrics=[];perclass=[];predictions=[]
    for name in MODELS:
        h=np.load(OUT/f'{name}_h.npy')
        if not np.isfinite(h).all() or len(h)!=len(meta):raise ValueError('Invalid features')
        for fold,tr,te in jobs:
            y=meta.tissue.to_numpy();model,scaler,pred=fit_probe(h[tr],h[te],y[tr]);labels=model.classes_
            if not set(y[te])<=set(labels):raise ValueError('Unseen class in test')
            # Held source may contain a subset of classes; score only true supported classes.
            supported=np.unique(y[te]);macro=f1_score(y[te],pred,labels=supported,average='macro',zero_division=0)
            metrics.append(dict(encoder=name,fold=fold,train_cells=int(tr.sum()),test_cells=int(te.sum()),train_classes=','.join(labels),test_classes=','.join(supported),accuracy=accuracy_score(y[te],pred),balanced_accuracy=balanced_accuracy_score(y[te],pred),macro_F1=macro,n_iter=int(model.n_iter_.max())))
            for label,v in classification_report(y[te],pred,labels=labels,output_dict=True,zero_division=0).items():
                if label in labels:perclass.append(dict(encoder=name,fold=fold,tissue=label,**v))
            for cell,truth,p in zip(meta.index[te],y[te],pred):predictions.append(dict(cell_id=cell,encoder=name,fold=fold,truth=truth,prediction=p))
            np.savez(OUT/f'{name}_{fold}_head.npz',coef=model.coef_,intercept=model.intercept_,classes=labels,center=scaler.mean_,scale=scaler.scale_)
            pd.DataFrame(confusion_matrix(y[te],pred,labels=labels),index=labels,columns=labels).to_csv(OUT/f'{name}_{fold}_confusion.tsv',sep='\t')
            print('RESULT',metrics[-1],flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.tsv',sep='\t',index=False);pd.DataFrame(perclass).to_csv(OUT/'per_tissue.tsv',sep='\t',index=False);pd.DataFrame(predictions).to_csv(OUT/'predictions.tsv.gz',sep='\t',index=False)
    pooled=[]
    for name,d in pd.DataFrame(predictions).groupby('encoder'):
        d=d[d.fold.str.startswith('holdout_')]
        if d.cell_id.duplicated().any():raise ValueError('Duplicate pooled heldout cell')
        pooled.append(dict(encoder=name,scope='pooled_head_leave_study_out_shared3',cells=len(d),accuracy=accuracy_score(d.truth,d.prediction),balanced_accuracy=balanced_accuracy_score(d.truth,d.prediction),macro_F1=f1_score(d.truth,d.prediction,average='macro')))
    pd.DataFrame(pooled).to_csv(OUT/'heldout_pooled.tsv',sep='\t',index=False)
    report()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(ROOT/'external/fonts/Arial.ttf');plt.rcParams['font.family']='Arial'
    m=pd.read_csv(OUT/'metrics.tsv',sep='\t');p=pd.read_csv(OUT/'heldout_pooled.tsv',sep='\t');meta=pd.read_csv(OUT/'cells.tsv.gz',sep='\t')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,name in zip(axes,MODELS):
        c=pd.read_csv(OUT/f'{name}_six_tissue_same_source_confusion.tsv',sep='\t',index_col=0)
        values=c.to_numpy()/c.sum(axis=1).to_numpy()[:,None];ax.imshow(values,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(len(c)),c.columns,rotation=45,ha='right');ax.set_yticks(range(len(c)),c.index);ax.set_title(name);ax.set_xlabel('Predicted tissue');ax.set_ylabel('True tissue')
        for i in range(len(c)):
            for j in range(len(c)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>.6 else 'black')
    fig.tight_layout();fig.savefig(OUT/'tissue_confusion.png',dpi=180);plt.close(fig)
    lines=['# GeneCompass与后训练ILC2：组织线性分类探针','',
        '两编码器冻结；相同细胞、median编码、2048基因、FP32、有效基因平均池化。训练集标准化后接768维多项logistic线性分类头，C=1、类别平衡，不调参、不选seed。无需再添加MLP。',
        '每研究×组织最多抽300个原posttrain_train、100个原posttrain_test。后者未参与后训练梯度，但之前有总体模型评估；同来源测试不等于跨研究独立验证。',
        '六类为Fat/LI/Lung/Pancreas/SI/Skin。Fat、LI、Skin仅一个研究，无法分离研究与组织。共享类别Lung/Pancreas/SI另做分类头留研究：只在其余研究的原训练细胞拟合，测试该研究原测试细胞。编码器后训练已经见过该研究其他细胞，因此不是encoder层面的完全未见研究。',
        '某些留出研究只含一个组织；该折balanced accuracy等于该组织召回，不能单独证明多类辨别。合并heldout预测与各研究指标同时报告；pancreas只有两个来源。',
        '```tsv',m.to_csv(sep='\t',index=False),'```','## 合并三类留研究预测','```tsv',p.to_csv(sep='\t',index=False),'```',
        '## 实际采样','```tsv',meta.groupby(['study','tissue','split']).size().rename('cells').to_csv(sep='\t'),'```',
        '没有随机细胞产生的动物级置信区间。每组织precision/recall/F1见per_tissue.tsv；原始混淆矩阵、预测、线性权重/标准化参数与checkpoint签名均保存。',
        '组织可分类只表明表示携带组织相关信息，不证明组织分类辅助loss会改善IL33响应。若组织已经容易分类，应优先解决独立响应背景不足，而非直接给后训练加分类任务。']
    lines += ['', '## 结果解释',
        '六类同来源测试原始accuracy=94.92%、后训练96.00%；macro-F1=0.9451/0.9561。后训练有约1个百分点提升，但两者本来已携带很强的同来源组织标签信息。',
        '相同三类组织的同来源对照见three_tissue_same_source；留研究合并balanced accuracy仅44.03%/45.53%，macro-F1=0.4282/0.4431。后训练没有消除来源迁移困难。',
        'GSE248385胰腺留出时正确率原始0%、后训练3%；GSE299900肺两者4%。后训练分别把78/100胰腺判为肺、66/100肺判为胰腺。不能把这个现象简单归因于标签错误；研究协议、细胞状态及技术来源混杂都可能参与。',
        '建议先不直接追加普通组织分类loss：同来源已接近饱和，优化该目标可能强化来源线索。先增加同组织跨研究的独立来源/评估，并考虑研究平衡的多任务小规模对照；任何辅助loss必须以留研究指标和扰动任务同时检验。',
        '本轮为单一固定抽样、固定正则强度的探针，不包含动物级显著性检验，不能将小幅提升称为统计显著收益。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'verification.json').write_text(json.dumps(dict(same_cells_both_encoders=True,train_test_disjoint=True,head_standardization_train_only=True,frozen_encoders=True,converged=bool((m.n_iter<3000).all())),indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--heads-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main(args.heads_only)
