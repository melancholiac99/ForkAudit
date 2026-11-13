"""Durable serial full-P1 queue. Stop on failure; never silently rerun an arm."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from prepare_campaign import sha, source_paths
from native_p1 import read,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    folder=args.campaign.resolve()
    lock=(folder/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registration=read(folder/'registration.json')
    assert read(folder/'publication.json')['verified_remote_head']
    assert {str(p.relative_to(ROOT)):sha(p) for p in source_paths()}==registration['source_sha256']
    assert sha(folder/'tasks.json')==registration['tasks_sha256']
    names={arm:f'{folder.name}_{arm}' for arm in read(folder/'PLAN.json')['arms']}
    write(folder/'run_names.json',names)
    runner=ROOT/'exploratory_ctha/verl_agent_migration/run_native.py'
    def status(**values):
        temp=folder/'status.tmp';temp.write_text(json.dumps(dict(time=time.time(),**values),indent=2)+'\n');temp.replace(folder/'status.json')
    def execute(command,label):
        with (folder/f'{label}.log').open('x') as stream:
            code=subprocess.call(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if code:raise RuntimeError(f'{label} exited{code}; no automatic retry')
    try:
        for index,(arm,name) in enumerate(names.items()):
            assert {str(p.relative_to(ROOT)):sha(p) for p in source_paths()}==registration['source_sha256'],'Source drift'
            assert shutil.disk_usage(ROOT).free>24*1024**3,'Insufficient free disk before arm'
            status(state='RUNNING',arm=arm,phase='training_and_tune',run=name)
            command=[sys.executable,str(runner),'--name',name,'--algorithm','grpo' if arm=='grpo' else 'gigpo',
                '--updates','12','--gpus','1','--device',str(index%2),'--frozen-base-bf16','--correction','bypass_ppo_clip',
                '--micro-batch','16','--logprob-micro','16','--ray-cpus','12','--model-seed','0',
                '--bounded-cuda-graphs','--rollout-session','--p1-campaign',str(folder),'--p1-arm',arm]
            execute(command,arm)
            run=ROOT/'.runtime/runs'/name
            status(state='RUNNING',arm=arm,phase='independent_audit',run=name)
            execute([sys.executable,str(HERE/'audit.py'),'--run',str(run)],arm+'_audit')
            status(state='RUNNING',arm=arm,phase='native_checkpoint_restore',run=name)
            restore_name=name+'_restore'
            execute([sys.executable,str(runner),'--name',restore_name,'--gpus','1','--device',str(index%2),
                '--frozen-base-bf16','--resume-from',str(run/'checkpoints/global_step_12'),'--val-only',
                '--restore-checkpoint-only','--micro-batch','16','--ray-cpus','12'],arm+'_restore')
            restored=read(ROOT/'.runtime/runs'/restore_name/'restore_rank_0.json')
            assert restored['passed'] and restored['actual']==read(run/'checkpoints/global_step_12/actor/live_fingerprint_rank_0.json')
            write(run/'final_restore_verified.json',restored)
        execute([sys.executable,str(HERE/'audit.py'),'--campaign',str(folder)],'comparison')
        status(state='COMPLETE',scope='Full three-arm one-seed P1; inspect comparison and limitations')
    except BaseException as error:
        status(state='FAILED_OR_INCOMPLETE',error=str(error),arm=locals().get('arm'))
        raise
    finally:
        os.sync()


if __name__=='__main__':main()
