"""Local experiment entry points reuse project GeneCompass loading and masking."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(k,'4')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ilc2_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/ilc2_numba')
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
for p in ('src','external/GeneCompass','src/downstream/geneko'):sys.path.insert(0,str(ROOT/p))
from posttrain_data import file_hash
DEFAULT=Path(__file__).with_name('config.json')
def config(path=DEFAULT):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def signature(cfg):
    paths={k:cfg[k] for k in ('source','split_manifest','token_dictionary','median_reference','functional_panel')}
    paths['weights']=cfg['checkpoint']+'/pytorch_model.bin'
    paths['model_config']=cfg['checkpoint']+'/config.json'
    return {k:file_hash(ROOT/v) for k,v in paths.items()}
