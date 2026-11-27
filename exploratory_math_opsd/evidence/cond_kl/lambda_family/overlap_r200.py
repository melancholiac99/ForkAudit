"""Start R@200 work on each GPU as its preceding shard finishes."""
import fcntl
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from endpoint200 import HERE, ROOT, aime
import launch


def after_shard(folder, lane):
    while not (folder / f'lane{lane}.json').is_file():
        time.sleep(2)


def run():
    work = ROOT/'run/cond-kl/endpoint200'
    work.mkdir(exist_ok=True)
    with (work/'r200-pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (HERE/'r200_probe.json').exists() and (HERE/'r200_aime24.json').exists():
            return
        previous = ROOT/'run/math500-long-seed1001-n2/nonthinking-R-lambda15@100/shards'
        contest = ROOT/'run/evaluation/R-extension@200/aime24/1001-lanes'
        original = launch.call

        def gated(root, args, log, env):
            after_shard(previous, int(env['CUDA_VISIBLE_DEVICES']))
            return original(root, args, log, env)

        def fixed(lane):
            after_shard(previous, lane)
            after_shard(contest, lane)
            subprocess.run([sys.executable, str(HERE/'dose15.py'), 'probe', 'r', str(lane)],
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES':str(lane)}, check=True)

        launch.call = gated
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                # Start monitoring fixed-probe dependencies after contest workers
                # start. Abort the process on contest failure to release waiters.
                jobs = [pool.submit(fixed, i) for i in range(4)]
                try:
                    aime()
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
                for job in jobs:
                    job.result()
            subprocess.run([sys.executable, str(HERE/'dose15.py'), 'collect', 'r'], check=True)
        finally:
            launch.call = original


if __name__ == '__main__':
    run()
