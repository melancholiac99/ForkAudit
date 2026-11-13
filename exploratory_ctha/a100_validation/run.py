"""Fresh A100 engineering queue; no inherited process takeover or learning pilots."""
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

from a100_common import DATA, FAST, HERE, MODEL, PARENT, ROOT, fingerprint, hardware, load, save, sha, sources

STATE_LOCK = threading.Lock()


def state(stage, **values):
    result = dict(stage=stage, pid=os.getpid(), time=time.time(), **values)
    with STATE_LOCK:
        save(DATA / 'status.json', result)
    print(json.dumps(result), flush=True)


def compare(left, right, output):
    fields = ['response_token_ids', 'prompt_token_ids', 'sample_seed', 'action', 'obs_before', 'obs_after', 'won_after']
    rows = []
    paths = sorted(left.glob('s*_r*.json'))
    assert len(paths) == 8
    for path in paths:
        a, b = load(path), load(right / path.name)
        exact = (a['game_file'] == b['game_file'] and a['won'] == b['won']
                 and len(a['steps']) == len(b['steps'])
                 and all(all(x[k] == y[k] for k in fields) for x, y in zip(a['steps'], b['steps'])))
        rows.append(dict(file=path.name, exact=exact))
    result = dict(passed=all(r['exact'] for r in rows), episodes=8, comparisons=rows)
    save(output, result, immutable=True)
    if not result['passed']:
        raise RuntimeError(f'Full real-trajectory repeat mismatch: {output}')


def select_config():
    plan = load(HERE / 'PLAN.json')
    rows = []
    for config in plan['candidates'][1:]:
        for folder in sorted((DATA / 'candidates').glob(config['name'] + '_gpu*')):
            registration = load(folder / 'registration.json')
            parity = load(folder / 'parity.json')
            performance = load(folder / 'performance.json')
            assert registration['input_sha256'] == sha(DATA / 'reference/prompts.json')
            rows.append(dict(candidate=config['name'], path=str(folder),
                             config=registration['config'], parity=parity, performance=performance))
    assert {r['candidate'] for r in rows} == {c['name'] for c in plan['candidates'][1:]}
    eligible = [r for r in rows if r['parity']['passed'] and r['performance']['warmup_timed_token_repeat']]
    by_name = {}
    for row in eligible:
        by_name.setdefault(row['candidate'], []).append(row)
    speed = {name: sum(r['performance']['tokens_per_second'] for r in group) / len(group)
             for name, group in by_name.items()}
    if not speed:
        raise RuntimeError('No FP32 candidate passed original numeric gates')
    selected = max(speed, key=speed.get)
    group = by_name[selected]
    assert len(group) == 2, 'Selected config must be measured on both A100s before freeze'
    first, second = [load(Path(r['path']) / 'replies.json') for r in group]
    assert len(first) == len(second) == 16
    same = all(a['prompt_ids'] == b['prompt_ids'] and a['completion_ids'] == b['completion_ids']
               and a['behavior_logprobs'] == b['behavior_logprobs'] for a, b in zip(first, second))
    if not same:
        raise RuntimeError('Selected config does not reproduce exactly across devices')
    config = group[0]['config']
    assert all(fingerprint(r['config']) == fingerprint(config) for r in group)
    save(DATA / 'selection.json', dict(candidates=rows, mean_speed_by_config=speed,
         selected=selected, cross_gpu_exact=True, scoring_rule='Highest mean generated tokens/s among passing candidate measurements'), immutable=True)
    save(DATA / 'frozen_config.json', config, immutable=True)
    return config


def job(name, arguments, gpu=None):
    reg = load(DATA / 'registration.json')
    assert sources() == reg['source_sha256'], 'Source changed after freeze; new revision required'
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env.update(TRITON_F32_DEFAULT='tf32x3', NVIDIA_TF32_OVERRIDE='0', VLLM_BATCH_INVARIANT='1',
               CTHA_A100_GROUP=DATA.name,
               PYTHONPATH=str(ROOT) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
    state('running', job=name, gpu=gpu)
    command = [sys.executable, '-u', *map(str, arguments)]
    with (DATA / (name + '.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(DATA / (name + '_launch.json'), dict(pid=process.pid, command=command, gpu=gpu,
             gpu_inventory=reg['hardware'], time=time.time()), immutable=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}; see {DATA / (name + ".log")}')


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (DATA / 'registration.json').exists():
            raise RuntimeError('Queue already registered; inspect existing process/evidence before restarting')
        config = select_config()
        unchanged = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--',
            'exploratory_ctha/v3_fp32_fast', 'exploratory_ctha/v3_engineering_fast_20261113'], text=True)
        assert not unchanged, 'Legacy implementations must remain byte-identical to upstream'
        old = DATA.parent / 'ctha_a100_eng_20261113_r1/cpu_validation.json'
        cold = load(old)
        assert cold['cold_start_gate']['passed'] and cold['offline_nonzero_aux_samples'] > 0
        assert all(cold['checks'].values())
        assert cold['import_verification_sha256'] == sha(ROOT / '.runtime/imports/import_verification.json')
        save(DATA / 'cpu_validation.json', cold, immutable=True)
        sys.path.insert(0, str(PARENT))
        from run import select_games
        games = select_games(load(ROOT / 'exploratory_E4/games.json')['train'], 0, 0, 16)
        save(DATA / 'train_games.json', games, immutable=True)
        save(DATA / 'repeat_games.json', games[:4], immutable=True)
        registration = dict(time=time.time(), scope=load(HERE / 'PLAN.json'), hardware=hardware(),
             config=config, config_fingerprint=fingerprint(config), source_sha256=sources(),
             model_path=str(MODEL), model_manifest_sha256=sha(ROOT / '.runtime/assets_manifest.json'),
             games_sha256=sha(ROOT / 'exploratory_E4/games.json'), train_games_sha256=sha(DATA / 'train_games.json'),
             cpu_evidence_origin=str(old), cpu_evidence_sha256=sha(old),
             repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             packages={p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'triton', 'transformers', 'peft', 'numpy', 'alfworld', 'textworld', 'wandb']})
        save(DATA / 'registration.json', registration, immutable=True)
        try:
            for name, folder, pattern in [('ctha', PARENT, 'test_ctha*.py'),
                    ('v3', PARENT / 'v3', 'test_v3.py'), ('fp32', PARENT / 'v3_fp32', 'test_*.py'),
                    ('fp32_fast', FAST, 'test_*.py')]:
                job('cpu_' + name, ['-m', 'unittest', 'discover', '-s', folder, '-p', pattern, '-v'], '')
            base = [HERE / 'backend.py']
            config_args = ['--config', DATA / 'frozen_config.json']
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(job, 'repeat_' + label,
                    [*base, 'collect', *config_args, '--games', DATA / 'repeat_games.json',
                     '--out', DATA / ('repeat_' + label), '--reps', 2], gpu)
                    for label, gpu in [('a', 0), ('b', 1)]]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(str(error))
            if errors:
                raise RuntimeError('; '.join(errors))
            compare(DATA / 'repeat_a', DATA / 'repeat_b', DATA / 'repeat_verification.json')
            job('initial_train', [*base, 'collect', *config_args, '--games', DATA / 'train_games.json',
                                 '--out', DATA / 'initial_train', '--reps', 4], 0)
            compare(DATA / 'repeat_a', DATA / 'initial_train', DATA / 'cross_batch_verification.json')
            job('initial_prepare', [*base, 'prepare', *config_args])
            job('direction_probe', [*base, 'train', *config_args, '--out', DATA / 'direction_probe'], 1)
            job('checkpoint_verify', [*base, 'checkpoint', *config_args], '')
            job('adapter_parity_sample', [*base, 'adapter_sample', *config_args], 0)
            job('adapter_parity_score', [*base, 'adapter_score', *config_args], 1)
            job('independent_verify', [HERE / 'verify.py', '--data', DATA], '')
            state('engineering_complete', Engineering='PASS', Learning='NOT_RUN',
                  Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED', publication='pending')
        except Exception:
            error = traceback.format_exc()
            save(DATA / 'failure.json', dict(error=error, time=time.time()))
            state('failed', Engineering='FAIL', Learning='NOT_RUN', Generalization='NOT_RUN',
                  CTHA_online_efficacy='NOT_TESTED', error=error)
            raise


if __name__ == '__main__':
    main()
