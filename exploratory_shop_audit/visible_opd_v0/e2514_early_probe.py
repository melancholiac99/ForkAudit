"""Score the frozen starting actor on group A while group B trains the next actor."""
import fcntl
import os
import subprocess
import sys
import time

from .e25_paths import PUBLIC, RUN, SERIAL
from exploratory_shop_audit.grpo_forced_audit.common import read, save


def check_lane():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert os.environ['OPD_MIGRATION_GROUP'] == 'A'
    assert os.environ['OPD_READONLY_FP32'] == '1'
    assert (PUBLIC / 'migration_2514/R-matched/endpoint.json').is_file()


def await_optimizer(path, update, timeout=1200):
    deadline = time.monotonic() + timeout
    receipt = path / 'optimizer.json'
    while not receipt.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Waiting for the signed R update {update}')
        time.sleep(0.5)
    meta = read(receipt)
    assert (meta['arm'], meta['update']) == ('R', update)
    assert meta['before_sha'] == read(path / 'target_manifest.json')['scorer_sha']


def measure_rank(update, rank, early):
    check_lane()
    path = RUN / SERIAL / 'pilot' / f'{update:02}' / 'R'
    # The regular worker takes this same lock. A failed early worker releases it
    # automatically; a completed one has already written the identical rank result.
    with (path / f'fp32_rank_{rank}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (path / f'fp32_part_{rank}.json').exists():
            return
        start = time.time()
        from .e25_layout import Layout
        from .e25_measure import per_update
        from .e25_probe import configure
        assert configure() == rank
        layout = Layout(SERIAL)
        previous = read(path.parents[1] / f'{update-1:02}' / 'R/optimizer.json')
        assert previous['after_sha'] == read(path / 'target_manifest.json')['scorer_sha']
        timing = dict(rank=rank, update=update, start=start, early=early)

        def new_snapshot(p):
            timing['before_forward_finished'] = time.time()
            await_optimizer(p, update)
            timing['after_forward_ready'] = time.time()

        per_update(rank, layout, update, ['R'], before_new_snapshot=new_snapshot)
        timing['end'] = time.time()
        save(path / f'fp32_timing_{rank}.json', timing)


def run(first):
    check_lane()
    root = RUN / SERIAL
    report_dir = RUN / 'migration_2514/early_fp32'
    report_dir.mkdir(parents=True, exist_ok=True)
    terminal = PUBLIC / 'migration_2514/R/endpoint.json'
    with (report_dir / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for update in range(first, 51):
            path = root / 'pilot' / f'{update:02}' / 'R'
            logfile = root / 'logs/R' / f'{update:03}_train.log'
            result = report_dir / f'{update:03}.json'
            while not terminal.is_file():
                if (path / 'optimizer.json').is_file() or result.is_file():
                    break
                if not logfile.is_file() or not (path / 'target_manifest.json').is_file():
                    time.sleep(1)
                    continue
                start = time.time()
                print(f'Starting frozen FP32 scoring during R update {update}', flush=True)
                with (report_dir / f'{update:03}.log').open('a') as log:
                    proc = subprocess.run(['torchrun', '--standalone', '--nproc-per-node=4', '-m',
                        __package__ + '.e2512_worker', 'per_update_early', str(update), 'R'],
                        stdout=log, stderr=subprocess.STDOUT)
                save(result, dict(update=update, start=start, end=time.time(), exit_code=proc.returncode,
                                  physical_devices='0,1,2,3'))
                if proc.returncode:
                    print(f'Early scoring exited {proc.returncode}; the regular worker owns recovery', flush=True)
                break
            if terminal.is_file():
                return


def verify_archived(update):
    """Compare the completed early shard with an ordinary two-snapshot evaluation."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from .e25_layout import Layout
    from .e25_measure import per_update
    from .e25_probe import configure
    from exploratory_shop_audit.grpo_forced_audit.common import digest_file
    check_lane()
    rank = configure()
    layout = Layout(SERIAL)
    original = layout.folder(update, 'R')
    part = f'fp32_part_{rank}.json'
    assert read(original / f'fp32_timing_{rank}.json')['early']
    with TemporaryDirectory(dir=RUN/'migration_2514/early_fp32') as tmp:
        scratch = Path(tmp)
        for name in ('target_manifest.json', 'bf16_before.pt'):
            (scratch/name).symlink_to(original/name)
        view = SimpleNamespace(name=layout.name, batch=layout.batch, snapshot=layout.snapshot,
                               folder=lambda k, arm: scratch)
        per_update(rank, view, update, ['R'])
        assert (scratch/part).read_bytes() == (original/part).read_bytes()
    save(PUBLIC/'migration_2514/early_probe_verification'/f'{update:03}_rank{rank}.json',
         dict(update=update, rank=rank, arm='R', identical_serialized_metrics=True,
              result_sha256=digest_file(original/part), target_manifest_sha256=digest_file(original/'target_manifest.json'),
              prior_adapter_sha256=read(layout.folder(update-1,'R')/'optimizer.json')['safetensors_sha256'],
              updated_adapter_sha256=read(original/'optimizer.json')['safetensors_sha256']))


if __name__ == '__main__':
    if sys.argv[1]=='verify':
        verify_archived(int(sys.argv[2]))
    else:
        run(int(sys.argv[1]))
