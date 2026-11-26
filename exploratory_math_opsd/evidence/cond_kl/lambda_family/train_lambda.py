"""Run the R launch command; restore the full optimizer checkpoint for follow-up."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from launch import read, save, sha, validate_checkpoint

ROOT = Path('/mnt/data/ForkAudit/math-opsd')
ARMS = {'R-lambda0':0., 'R-lambda05':.5}


def train(arm, end):
    assert arm in (*ARMS, 'R') and end in (100,200)
    if arm=='R':
        assert end==200
    name = 'R-extension' if arm=='R' else arm
    out = ROOT/'run'/name
    if not out.exists():
        storage = Path('/mnt/data/ForkAudit/lambda-runs')/name
        storage.mkdir(parents=True,exist_ok=True)
        out.symlink_to(storage, target_is_directory=True)
    receipt = out/('complete.json' if end==100 else 'extended.json')
    if receipt.exists():
        assert sha(out/f'checkpoint-{end}/adapter_model.safetensors')==read(receipt)['adapters'][str(end)]
        return
    src = ROOT/'run/cond-kl/lambda-source'
    shutil.copytree(ROOT/'source',src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
    if arm in ARMS:
        trainer = src/'opsd_trainer.py'
        code = trainer.read_text()
        pattern = '            loss = self.generalized_jsd_loss('
        assert code.count(pattern)==1
        start = code.index(pattern)
        stop = code.index('            del student_logits_for_loss',start)
        trainer.write_text(code[:start]+'            from lambda_objective import trainer_loss\n            loss = trainer_loss(self,student_logits_for_loss,teacher_logits_for_loss,shifted_labels)\n'+code[stop:])
        shutil.copy(HERE/'objective.py',src/'lambda_objective.py')
    # Keep adapter checkpoints at the usual cadence, plus full state for @100.
    launch = src/'opsd_train.py'
    text = launch.read_text()
    anchor = '        if state.global_step == 10:'
    assert text.count(anchor)==1
    launch.write_text(text.replace(anchor,'        args.save_only_model = state.global_step != 100\n'+anchor))
    args = ['bash','scripts/run_opsd_1b.sh','--stop_after_updates',str(end),'--seed','42','--data_seed','42',
            '--local_train_glob',str(ROOT/'datasets/train/data/*.parquet')]
    if end==200:
        previous=ROOT/'run'/arm/'checkpoint-100'
        validate_checkpoint(previous,100)
        assert len(list((previous/'global_step100').glob('*optim_states.pt')))==4
        args+=['--resume_from_checkpoint',str(previous)]
    env=dict(os.environ,OPSD_ARM=name,OPSD_MODEL=str(ROOT/'models/Qwen3-1.7B'),OPSD_OUTPUT=str(out),
             OPSD_LAMBDA=str(ARMS.get(arm,1.)))
    start=time.monotonic()
    with (ROOT/'logs'/f'{name}-{end}-train.log').open('a') as f:
        subprocess.run(args,cwd=src,env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
    adapters={}
    for k in range(25 if end==100 else 125,end+1,25):
        path=out/f'checkpoint-{k}'
        validate_checkpoint(path,k)
        adapters[str(k)]=sha(path/'adapter_model.safetensors')
    save(receipt,dict(arm=arm,wall_seconds=time.monotonic()-start,adapters=adapters))


if __name__=='__main__':
    train(sys.argv[1],int(sys.argv[2]))
