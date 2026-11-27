"""Advance registered half-dose shards on GPUs released by contrast generation."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import late_switch
import launch

WORK=Path('/mnt/data2/ForkAudit/contrast')
FOLDER=late_switch.ROOT/'run/evaluation/R-lambda05@200/aime24/1001-lanes'


def wait_free(lane):
    while True:
        batches=list(WORK.glob(f'batch-{lane}-*.json'))
        memory=subprocess.check_output(['nvidia-smi','-i',str(lane),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if len(batches)==25 and int(memory.strip())<512:
            return
        time.sleep(3)


def main():
    with (WORK/'half-gpu-handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        original=launch.call
        def dispatch(root,args,log,env):
            wait_free(int(env['CUDA_VISIBLE_DEVICES']))
            return original(root,args,log,env)
        launch.call=dispatch
        def fixed(lane):
            while not (FOLDER/f'lane{lane}.json').exists():
                time.sleep(3)
            wait_free(lane)
            subprocess.run([sys.executable,str(late_switch.HERE/'late_switch.py'),'probe','half',str(lane)],
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':str(lane)},check=True)
        with ThreadPoolExecutor(4) as pool:
            jobs=[pool.submit(fixed,i) for i in range(4)]
            try:
                late_switch.contest('half')
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(1)
            for job in jobs:job.result()
        late_switch.collect(('R-lambda05',),200,'late_half_fixed.json')


if __name__=='__main__':main()
