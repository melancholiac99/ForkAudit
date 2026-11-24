import json,os,subprocess,sys
from pathlib import Path
from common import ROOT,WORK,json_save
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from launch import train,evaluate,sha
env={**os.environ,'CUDA_HOME':str(ROOT/'cuda-12.8'),'TORCH_CUDA_ARCH_LIST':'8.0',
     'HF_HOME':str(ROOT/'hf-cache'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
     'WANDB_MODE':'offline','WANDB_DIR':str(ROOT/'logs'),'PYTHONNOUSERSITE':'1',
     'TORCH_EXTENSIONS_DIR':str(ROOT/'torch-extensions'),'TRITON_CACHE_DIR':str(ROOT/'triton-cache')}
if sys.argv[1]=='train':
    assert json.loads((WORK/'target_check.json').read_text())['invariant']
    train(ROOT,{**env,'OPSD_PROJECTOR':str(WORK/'projector.pt')},'P')
else:
    assert (ROOT/'run/P/complete.json').exists()
    sequence=[('P@100','aime24')]+[(m,b) for m in ('B','R@100','P@100') for b in ('aime25','hmmt25')]
    cells={}
    for model,bench in sequence:
        cells[f'{model}/{bench}']=evaluate(ROOT,env,model,bench,1001)
    json_save(WORK/'evaluation_receipts.json',cells)
    from report import report
    report()
