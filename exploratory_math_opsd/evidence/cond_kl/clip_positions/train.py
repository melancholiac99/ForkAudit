"""Use the deployed R trainer, changing only the maximum of the elementwise clamp."""
import os,sys,subprocess,shutil,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from launch import validate_checkpoint,sha,save,read
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
ARMS=('R-clip-boundary','R-clip-interior')
def main(arm):
    assert arm in ARMS
    out=ROOT/'run'/arm
    if (out/'complete.json').is_file():return
    src=ROOT/'run/cond-kl/clip-prefix-source'
    shutil.copytree(ROOT/'source',src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
    file=src/'opsd_trainer.py';code=file.read_text()
    pattern='            loss = self.generalized_jsd_loss('
    assert code.count(pattern)==1 and code.count('token_clip=self.jsd_token_clip,')==1
    code=code.replace(pattern,"            from prefix_positions import cap_tensor\n            assert self.fixed_teacher and self.beta==0 and self.jsd_token_clip==.05\n"+pattern)
    code=code.replace('token_clip=self.jsd_token_clip,','token_clip=cap_tensor(self,inputs),').replace('jsd.clamp(max=token_clip)','jsd.clamp(max=token_clip.to(jsd.dtype))')
    file.write_text(code);shutil.copy(HERE/'positions.py',src/'prefix_positions.py')
    env=dict(os.environ,OPSD_ARM=arm,OPSD_OUTPUT=str(out),OPSD_MODEL=str(ROOT/'models/Qwen3-1.7B'))
    assert not any(env.get(k) for k in ('OPSD_PROJECTOR','OPSD_SCREEN_ARM','OPSD_RRES_PROJECTOR'))
    start=time.monotonic()
    with (ROOT/'logs'/f'{arm}-train.log').open('a') as log:
        subprocess.run(['bash','scripts/run_opsd_1b.sh','--stop_after_updates','100','--seed','42','--data_seed','42',
                       '--local_train_glob',str(ROOT/'datasets/train/data/*.parquet')],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    hashes={}
    for k in (25,50,75,100):
        ck=out/f'checkpoint-{k}';validate_checkpoint(ck,k);hashes[str(k)]=sha(ck/'adapter_model.safetensors')
    save(out/'complete.json',dict(arm=arm,wall_seconds=time.monotonic()-start,adapters=hashes))
if __name__=='__main__':main(sys.argv[1])
