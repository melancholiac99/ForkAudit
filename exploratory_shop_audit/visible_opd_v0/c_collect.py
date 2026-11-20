"""Collect the authorized initial on-policy batch on four native student replicas."""
import os
import time

from .b_collect import batch, PARENT, RUN as STAGE_B
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, append, read
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner

RUN = ROOT / '.runtime/runs/shop_visible_opd_c_smoke_20261120'

def tasks():
    split = read(PARENT/'asin_split_manifest.json')
    excluded = {r['goal_index'] for k in ('T_opd_eval','T_pre','T_cal','V') for r in split[k]}
    selected = [r for r in split['T_opd'] if r['goal_index'] not in excluded][:32]
    assert len({r['goal_index'] for r in selected}) == 32
    return selected

def collect():
    os.umask(0o077)
    RUN.mkdir(exist_ok=True)
    assert read(STAGE_B/'reports/acceptance.json')['passed']
    link = RUN/'snapshots'
    if not link.exists():
        link.symlink_to(PARENT/'snapshots', target_is_directory=True)
    goals = read(link/'goals_frozen.json')
    os.environ['SHOP_GPU_IDS'] = '0,1,2,3'
    start = time.monotonic()
    runner = Runner(RUN)
    try:
        for rep in range(2):
            path = RUN/f'rollout_{rep}.jsonl'
            if not path.exists():
                tmp = path.with_suffix(f'.{time.time_ns()}.partial')
                batch(runner,tasks(),goals,tmp,rep=rep,stream='T_opd_smoke')
                tmp.rename(path)
    finally:
        runner.close()
        append(RUN/'cost.jsonl', dict(stage='collection',gpu_hours=4*(time.monotonic()-start)/3600))

if __name__ == '__main__':
    collect()
