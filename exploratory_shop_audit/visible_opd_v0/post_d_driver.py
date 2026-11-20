"""Run the post-pilot checks only after both behavior endpoints have been sealed."""
import fcntl
import os
import subprocess
import sys
import time

from .post_d_common import POST, RUN, required_stage
from exploratory_shop_audit.grpo_forced_audit.common import append, utc


def main():
    os.umask(0o077)
    POST.mkdir(parents=True, exist_ok=True)
    with (POST / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (RUN / 'reports/delivery.json').exists():
            time.sleep(30)
        required_stage()
        if (POST / 'reports/delivery.json').exists():
            return
        stages = [('post_d_snapshot', [], 4), ('post_d_report', ['snapshot'], 0),
                  ('post_d_reset', ['train'], 4), ('post_d_reset', ['measure'], 4),
                  ('post_d_gradients', [], 4), ('post_d_report', ['final'], 0)]
        for module, args, gpus in stages:
            name = '_'.join([module, *args])
            cmd = ['torchrun', '--standalone', '--nproc-per-node=4'] if gpus else [sys.executable]
            print(utc(), name, flush=True)
            start = time.monotonic()
            with (POST / f'{name}.log').open('a') as log:
                result = subprocess.run(cmd + ['-m', f'{__package__}.{module}', *args], stdout=log, stderr=log)
            append(POST / 'allocation.jsonl', dict(stage=name, time=utc(), gpus=gpus,
                   seconds=time.monotonic()-start, exit_code=result.returncode))
            result.check_returncode()


if __name__ == '__main__':
    main()
