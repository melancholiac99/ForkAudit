"""Two isolated four-round arms; exact registered base-cache subsets."""
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
from small_common import ENGINEERING, HERE, OLD, PILOT, ROOT, fingerprint, hardware, load, save, sha, sources

LOCK = threading.Lock()
OLD_RUN = ROOT / '.runtime/runs/ctha_a100_baseline_rl_r1'
CACHE = ROOT / '.runtime/runs/ctha_a100_eng_20261113_r2/initial_train'


def state(stage, **kw):
    with LOCK:
        value = dict(stage=stage, time=time.time(), pid=os.getpid(), **kw)
        save(PILOT / 'status.json', value)
        print(json.dumps(value), flush=True)


def job(name, args, gpu):
    assert sources() == load(PILOT / 'registration.json')['source_sha256']
    env = os.environ.copy()
    env.update(FULL_RESPONSE_PILOT=str(PILOT), CUDA_VISIBLE_DEVICES=str(gpu),
        TRITON_F32_DEFAULT='tf32x3', NVIDIA_TF32_OVERRIDE='0', VLLM_BATCH_INVARIANT='1', PYTHONPATH=str(ROOT))
    command = [sys.executable, '-u', *map(str, args)]
    state('running', job=name, gpu=gpu)
    with (PILOT / (name + '.log')).open('x') as log:
        proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save(PILOT / (name + '_launch.json'), dict(pid=proc.pid, command=command, gpu=gpu, time=time.time()), immutable=True)
        code = proc.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}; {PILOT / (name + ".log")}')


def subset(src, dest, games):
    dest.mkdir(parents=True, exist_ok=False)
    hashes, episodes = {}, []
    for game in games:
        for rep in range(4):
            name = f's{game["seed"]}_r{rep}.json'
            shutil.copy2(src / name, dest / name)
            hashes[name] = sha(src / name)
            assert sha(dest / name) == hashes[name]
            episode = load(dest / name)
            assert episode['game_file'] == game['game_file']
            assert episode['provenance']['adapter_sha256'] is None
            episodes.append(episode)
    original = load(src / 'summary.json')
    assert all(e['provenance'] == original['provenance'] for e in episodes)
    summary = dict(original, n_games=4, n_episodes=16,
        success_rate=sum(e['won'] for e in episodes) / 16,
        env_steps=sum(len(e['steps']) for e in episodes),
        derived_subset=dict(source=str(src), summary_sha256=sha(src / 'summary.json'), episode_sha256=hashes,
            note='Original episode bytes and provenance retained; selected subset summary derived explicitly.'))
    save(dest / 'summary.json', summary, immutable=True)
    return summary


def main():
    PILOT.mkdir(parents=True, exist_ok=False)
    assert load(ENGINEERING / 'verification.json')['Engineering'] == 'PASS'
    publication = load(ROOT / '.runtime/publication_staging/small_baseline_publication.json')
    assert publication['verified_remote'] and publication['source_sha256'] == sources()
    plan = load(HERE / 'PLAN.json')
    games = [g for g in load(ENGINEERING / 'train_games.json') if g['seed'] in plan['pilot']['task_seeds']]
    assert len(games) == 4
    save(PILOT / 'train_games.json', games, immutable=True)
    shutil.copy2(ENGINEERING / 'frozen_config.json', PILOT / 'frozen_config.json')
    save(PILOT / 'registration.json', dict(time=time.time(), scope=plan, source_sha256=sources(),
        config_fingerprint=fingerprint(load(PILOT / 'frozen_config.json')), hardware=hardware(),
        games_sha256=sha(PILOT / 'train_games.json'), engineering_sha256=sha(ENGINEERING / 'verification.json'),
        protocol_publication=publication, packages={p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'triton', 'transformers', 'peft']},
        old_pilot='Stopped at zero updates; reference scoring only', selected_on='Original-base outcomes; enriched diagnostic only'), immutable=True)
    baseline = subset(OLD_RUN / 'base_train_eval', PILOT / 'base_train_eval', games)
    cache = subset(CACHE, PILOT / 'shared_initial_cache', games)
    assert cache['env_steps'] == 582
    save(PILOT / 'subset_verification.json', dict(passed=True, initial_rows=582,
        cache=cache['derived_subset'], baseline=baseline['derived_subset']), immutable=True)
    job('cpu_scope', ['-m', 'unittest', 'discover', '-s', HERE, '-p', 'test_small.py', '-v'], '')

    def arm_run(arm, gpu):
        try:
            for i in range(4):
                rnd = PILOT / arm / f'round_{i:03d}'
                rnd.mkdir(parents=True)
                if i == 0:
                    shutil.copytree(PILOT / 'shared_initial_cache', rnd / 'rollouts')
                else:
                    job(f'{arm}_r{i}_collect', [HERE / 'worker.py', 'collect', '--arm', arm, '--iteration', i], gpu)
                for action in ['prepare', 'train']:
                    job(f'{arm}_r{i}_{action}', [HERE / 'worker.py', action, '--arm', arm, '--iteration', i], gpu)
            job(arm + '_final_eval', [HERE / 'worker.py', 'collect', '--arm', arm, '--iteration', 3, '--eval'], gpu)
            return None
        except Exception:
            error = traceback.format_exc()
            save(PILOT / arm / 'failure.json', dict(error=error, time=time.time()))
            return error

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs = [pool.submit(arm_run, arm, gpu) for arm, gpu in [('grpo', 0), ('gigpo', 1)]]
        errors = [err for f in concurrent.futures.as_completed(fs) if (err := f.result())]
    if errors:
        raise RuntimeError('\n'.join(errors))
    job('independent_verify', [HERE / 'verify.py'], '')
    state('complete', Execution='COMPLETE', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')


if __name__ == '__main__':
    lock = ROOT / '.runtime/locks/full_response_workflow.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception:
            if PILOT.exists():
                save(PILOT / 'failure.json', dict(error=traceback.format_exc(), time=time.time()))
                state('failed', Learning='INCOMPLETE', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')
            raise
