"""Copy the deployed R source, replacing only the loss call and disabling clipping."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from launch import read, save, sha, validate_checkpoint

root=Path('/mnt/data/ForkAudit/math-opsd')
work=root/'run/cond-kl'
src=work/'source'
def main(arm):
    assert arm in ('R-noclip','R-cond','R-condA')
    if arm!='R-condA':assert 'preflight' in read(HERE/'results.json')
    src=work/('condA-source' if arm=='R-condA' else 'source')
    out=root/'run'/arm
    if (out/'complete.json').exists():
        return
    shutil.copytree(root/'source',src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
    file=src/'opsd_trainer.py'
    code=file.read_text()
    a=code.index('            loss = self.generalized_jsd_loss(')
    b=code.index('            del student_logits_for_loss',a)
    file.write_text(code[:a]+'            from objective import trainer_loss\n            loss = trainer_loss(self,student_logits_for_loss,teacher_logits_for_loss,shifted_labels)\n'+code[b:])
    shutil.copy((HERE/'condA' if arm=='R-condA' else HERE)/'objective.py',src)
    env={**os.environ,'OPSD_MODEL':str(root/'models/Qwen3-1.7B'),'OPSD_OUTPUT':str(out),'OPSD_ARM':arm,
         'HF_HOME':str(root/'hf-cache'),'HF_DATASETS_OFFLINE':'1','HF_HUB_OFFLINE':'1','WANDB_MODE':'offline','WANDB_DIR':str(root/'logs'),
         'CUDA_HOME':str(root/'cuda-12.8'),'TORCH_EXTENSIONS_DIR':str(root/'torch-extensions'),'TRITON_CACHE_DIR':str(root/'triton-cache'),
         'PATH':str(root/'conda-env/bin')+':'+str(root/'cuda-12.8/bin')+':'+os.environ['PATH']}
    args=['bash','scripts/run_opsd_1b.sh','--jsd_token_clip','0','--stop_after_updates','100','--seed','42','--data_seed','42','--local_train_glob',str(root/'datasets/train/data/*.parquet')]
    start=time.monotonic()
    with (root/'logs'/f'{arm}-train.log').open('a') as log:
        subprocess.run(args,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    wall=time.monotonic()-start
    hashes={}
    for step in (25,50,75,100):
        path=out/f'checkpoint-{step}'
        validate_checkpoint(path,step)
        hashes[str(step)]=sha(path/'adapter_model.safetensors')
    save(out/'complete.json',dict(arm=arm,wall_seconds=wall,adapters=hashes))
if __name__=='__main__':
    main(sys.argv[1])
