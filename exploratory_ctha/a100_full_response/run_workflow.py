"""Bounded new-protocol engineering, then four rounds per baseline arm."""
import argparse
import concurrent.futures
import fcntl
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from protocol_common import ENGINEERING, FAST, HERE, MODEL, PILOT, R2, ROOT, fingerprint, hardware, load, save, sha, sources

LOCK = threading.Lock()


def status(root, stage, **values):
    with LOCK:
        value = dict(stage=stage, time=time.time(), pid=os.getpid(), **values)
        save(root / 'status.json', value)
        print(json.dumps(value), flush=True)


def job(root, name, args, gpu):
    assert sources() == load(root / 'registration.json')['source_sha256']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TRITON_F32_DEFAULT='tf32x3', NVIDIA_TF32_OVERRIDE='0',
        VLLM_BATCH_INVARIANT='1', FULL_RESPONSE_ENGINEERING=str(ENGINEERING), FULL_RESPONSE_PILOT=str(PILOT),
        CTHA_A100_ROOT=str(ENGINEERING), CTHA_A100_GROUP='a100-full-response-r3',
        PYTHONPATH=str(ROOT), WANDB_MODE='offline')
    command = [sys.executable, '-u', *map(str, args)]
    status(root, 'running', job=name, gpu=gpu)
    with (root / (name + '.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(root / (name + '_launch.json'), dict(pid=process.pid, command=command, gpu=gpu, time=time.time(), inventory=hardware()), immutable=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}: {root / (name + ".log")}')


def register(root, **extra):
    save(root / 'registration.json', dict(time=time.time(), scope=load(HERE / 'PLAN.json'),
        source_sha256=sources(), hardware=hardware(), config_fingerprint=fingerprint(load(root / 'frozen_config.json')),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'triton', 'transformers', 'peft', 'numpy', 'alfworld', 'textworld']},
        model_path=str(MODEL), assets_manifest_sha256=sha(ROOT / '.runtime/assets_manifest.json'),
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), **extra), immutable=True)


def engineering():
    ENGINEERING.mkdir(parents=True, exist_ok=False)
    assert load(R2 / 'failure_review.json')['Engineering'] == 'FAIL'
    names = ['frozen_config.json', 'selection.json', 'train_games.json', 'repeat_games.json', 'cpu_validation.json',
             'repeat_a', 'repeat_b', 'initial_train', 'initial_prepared', 'repeat_verification.json', 'cross_batch_verification.json']
    reused = {}
    for name in names:
        src, dest = R2 / name, ENGINEERING / name
        if src.is_dir():
            shutil.copytree(src, dest)
            for file in src.rglob('*'):
                if file.is_file():
                    reused[str(file.relative_to(R2))] = sha(file)
        else:
            shutil.copy2(src, dest)
            reused[name] = sha(src)
    register(ENGINEERING, reused_evidence_sha256=reused,
        note='New acceptance protocol and NEW probe checkpoint; explicitly reused frozen-base A100 evidence only. R2 remains FAIL.')
    for name, folder, pattern in [('ctha', HERE.parent, 'test_ctha*.py'), ('v3', HERE.parent / 'v3', 'test_v3.py'),
            ('fp32', HERE.parent / 'v3_fp32', 'test_*.py'), ('fp32_fast', FAST, 'test_*.py'), ('protocol', HERE, 'test_protocol.py')]:
        job(ENGINEERING, 'cpu_' + name, ['-m', 'unittest', 'discover', '-s', folder, '-p', pattern, '-v'], '')
    worker = HERE / 'engineering_worker.py'
    for action, gpu in [('train', 0), ('checkpoint', ''), ('adapter_sample', 1), ('adapter_score', 0)]:
        job(ENGINEERING, action, [worker, action], gpu)
    job(ENGINEERING, 'independent_verify', [HERE / 'verify_engineering.py'], '')
    status(ENGINEERING, 'complete', Engineering='PASS', protocol=load(HERE / 'PLAN.json')['version'], R2='FAIL', Learning='NOT_RUN', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')


def pilot():
    assert load(ENGINEERING / 'verification.json')['Engineering'] == 'PASS'
    publication = load(ROOT / '.runtime/publication_staging/full_response_protocol_publication.json')
    assert publication['verified_remote'] and publication['source_sha256'] == sources(), 'Publish and verify protocol/code before learning updates'
    PILOT.mkdir(parents=True, exist_ok=False)
    for name in ['frozen_config.json', 'train_games.json']:
        shutil.copy2(ENGINEERING / name, PILOT / name)
    register(PILOT, engineering_verification_sha256=sha(ENGINEERING / 'verification.json'),
        protocol_publication=publication, arm_devices={'grpo': hardware()[0], 'gigpo': hardware()[1]},
        initial_cache_origin=str(R2 / 'initial_train'),
        initial_cache_sha256={p.name: sha(p) for p in (R2 / 'initial_train').glob('*.json')})
    worker = HERE / 'pilot_worker.py'
    job(PILOT, 'base_fixed_train_eval', [worker, 'collect', '--arm', 'grpo', '--base-eval'], 0)

    def arm_run(arm, gpu):
        try:
            for i in range(4):
                rnd = PILOT / arm / f'round_{i:03d}'
                rnd.mkdir(parents=True)
                if i == 0:
                    # Identical model/tasks/seeds/config; explicit shared frozen-base cache.
                    shutil.copytree(R2 / 'initial_train', rnd / 'rollouts')
                    for p in (rnd / 'rollouts').glob('*.json'):
                        assert sha(p) == load(PILOT / 'registration.json')['initial_cache_sha256'][p.name]
                    save(rnd / 'shared_initial_cache.json', dict(origin=str(R2 / 'initial_train'),
                        config_fingerprint=fingerprint(load(PILOT / 'frozen_config.json')), adapter=None), immutable=True)
                else:
                    job(PILOT, f'{arm}_r{i}_collect', [worker, 'collect', '--arm', arm, '--iteration', i], gpu)
                for action in ['prepare', 'train']:
                    job(PILOT, f'{arm}_r{i}_{action}', [worker, action, '--arm', arm, '--iteration', i], gpu)
            job(PILOT, arm + '_final_fixed_train_eval', [worker, 'collect', '--arm', arm, '--iteration', 3, '--eval'], gpu)
            save(PILOT / arm / 'execution.json', dict(Execution='COMPLETE', rounds=4), immutable=True)
            return None
        except Exception:
            error = traceback.format_exc()
            save(PILOT / arm / 'failure.json', dict(error=error, time=time.time()))
            return error

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(arm_run, arm, gpu) for arm, gpu in [('grpo', 0), ('gigpo', 1)]]
        errors = [error for future in concurrent.futures.as_completed(futures) if (error := future.result())]
    if errors:
        raise RuntimeError('\n'.join(errors))
    job(PILOT, 'independent_verify', [HERE / 'verify_pilot.py'], '')
    status(PILOT, 'complete', Execution='COMPLETE', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['engineering', 'pilot'])
    a = p.parse_args()
    root = ENGINEERING if a.phase == 'engineering' else PILOT
    lock = ROOT / '.runtime/locks/full_response_workflow.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            (engineering if a.phase == 'engineering' else pilot)()
        except Exception:
            if root.exists():
                save(root / 'failure.json', dict(error=traceback.format_exc(), time=time.time()))
                status(root, 'failed', Engineering='FAIL' if a.phase == 'engineering' else 'PASS',
                    Learning='NOT_RUN' if a.phase == 'engineering' else 'INCOMPLETE',
                    Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')
            raise


if __name__ == '__main__':
    main()
