"""Collect two on-policy trajectories on the frozen shared task schedule for each arm."""
import os
import sys
import time
from .b_collect import batch
from .d_data import folder, state_dir, PARENT, read
from .s21_store import ARMS
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.grpo_forced_audit.common import save, digest_file


def collect(k,rank):
    arm=ARMS[rank];path=folder(k,arm)
    if (path/'collection.json').exists():return
    checkpoint=state_dir(k-1,arm)/'adapter'
    sha=digest_file(checkpoint/'adapter_model.safetensors')
    assert sha==read(folder(k-1,arm)/'optimizer.json')['safetensors_sha256']
    os.environ['SHOP_GPU_IDS']=str(rank)
    tasks=read(path.parent/'tasks.json')
    goals=read(PARENT/'snapshots/goals_frozen.json')
    start=time.monotonic()
    runner=Runner(path,checkpoint)
    try:
        for rep in range(2):
            dest=path/f'rollout_{rep}.jsonl'
            if dest.exists():continue
            tmp=dest.with_suffix(f'.{time.time_ns()}.pending')
            batch(runner,tasks,goals,tmp,rep,stream=f'T_opd_D_{k}',include_legal=True)
            tmp.replace(dest)
    finally:
        runner.close()
    save(path/'collection.json',dict(checkpoint=sha,seconds=time.monotonic()-start,
         rollout_sha256=[digest_file(path/f'rollout_{i}.jsonl') for i in range(2)]))


if __name__=='__main__':
    os.umask(0o077)
    collect(int(sys.argv[1]),int(os.environ['LOCAL_RANK']))
