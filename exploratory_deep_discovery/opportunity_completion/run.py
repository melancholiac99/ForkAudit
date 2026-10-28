"""Run 2 arms x 2 information conditions, with per-continuation checkpointing.

No GPU is used with --dry or --check-only. The GPU path reuses scripts/policy.py.
Native prompts and continuations use the original common.py and 50-step budget.
"""
import argparse
from collections import deque
import fcntl
from pathlib import Path
import random
import re
import time

from protocol import (HERE, ROOT, VERSION, ARMS, CONDITIONS, admissible, common, digest,
                      file_hash, load_design, presented_observation, read, seed_for,
                      verify_environment, verify_inputs, write)


class DryPolicy:
    """Random admissible actions, deliberately ignores cue; plumbing only."""
    vllm_version = 'dry-random-admissible'

    def sample(self, prompts, seeds, n=1, logprobs=False):
        result = []
        for prompt, seed in zip(prompts, seeds):
            tail = prompt.split('Your admissible actions of the current situation are: [', 1)[1]
            choices = re.findall(r"'([^']+)'", tail.split('].', 1)[0])
            action = random.Random(seed).choice(choices)
            result.append([dict(text=f'<think>CPU plumbing test.</think>\n<action>{action}</action>',
                                token_ids=[], finish_reason='stop')])
        return result

    def close(self):
        pass


def result_key(point, arm, condition, rep):
    return f"{point['id']}__{arm}__{condition}__r{rep:03d}"


def validate_record(record, fingerprint, key):
    if (record.get('run_fingerprint') != fingerprint or record.get('key') != key
            or record.get('status') != 'complete' or not isinstance(record.get('won'), bool)):
        raise ValueError(f'Incompatible/incomplete checkpoint: {key}')


def advance(c, action, valid, fmt, response, seed, result, forced=False, token_count=0):
    obs, adm, won, done = result
    shown_before = c['obs']
    c['memory'].append(dict(text_obs=shown_before, action=action))
    c['steps_taken'] += 1
    c['obs'] = obs
    if forced:
        c['obs'] = presented_observation(obs, c['condition'], c['point']['clue']['text'])
    c['transitions'].append(dict(step=c['steps_taken'], forced=forced, action=action,
                                 valid=bool(valid), fmt=fmt, response=response, seed=seed,
                                 generated_tokens=token_count,
                                 observation_before=c['raw_obs'], shown_observation_before=shown_before,
                                 observation_after=obs, shown_observation_after=c['obs'],
                                 won=bool(won), environment_done=bool(done)))
    c['raw_obs'], c['admissible'] = obs, adm
    c['won'] = bool(won)
    c['done'] = bool(won or done or c['steps_taken'] >= common.MAX_STEPS)
    c['stop_reason'] = ('success' if won else 'environment_done' if done else
                        'step_cap' if c['steps_taken'] >= common.MAX_STEPS else None)


def execute(args, design, points, out, fingerprint):
    pending = deque()
    existing = 0
    for point in points:
        for rep in range(args.reps):
            for arm in ARMS:
                for condition in CONDITIONS:
                    key = result_key(point, arm, condition, rep)
                    path = out / 'results' / f'{key}.json'
                    if path.exists():
                        validate_record(read(path), fingerprint, key)
                        existing += 1
                    else:
                        pending.append(dict(key=key, point=point, arm=arm, condition=condition,
                                            rep=rep, path=path))
    total = len(points) * len(ARMS) * len(CONDITIONS) * args.reps
    print(f'{len(points)} points; {total} continuations; {existing} complete; {len(pending)} pending', flush=True)
    if not pending:
        print('Already complete; model was not loaded.', flush=True)
        return
    import alfenv
    from envpool import EnvPool
    pool = None
    policy = None
    live = {}
    start = time.monotonic()
    last_update = start
    n_finished = existing
    generated_steps = 0
    generated_tokens = 0
    try:
        # Fork CPU workers before CUDA initialization, matching original replay implementation.
        pool = EnvPool(args.workers)
        if args.dry:
            policy = DryPolicy()
        else:
            from policy import Policy
            policy = Policy(model_path=common.MODEL_PATH,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            max_model_len=args.max_model_len)
        write(out / 'runtime.json', dict(vllm=policy.vllm_version, dry=args.dry,
                                        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        while pending or live:
            new = []
            for _ in range(min(args.concurrency - len(live), len(pending))):
                c = pending.popleft()
                point = c['point']
                c.update(memory=[dict(m) for m in point['memory']], steps_taken=point['turn'] - 1,
                         transitions=[], won=False, done=False, first_prompt=None)
                new.append(c)
            if new:
                jobs = [(c['key'], str(Path(alfenv.ALFWORLD_DATA) / c['point']['game_file']),
                         c['point']['prefix'], c['point']['expected_prefix_observations']) for c in new]
                opened = pool.open(jobs)
                for c in new:
                    obs, adm, won, div = opened[c['key']]
                    p = c['point']
                    if div or won or obs != p['observation_before'] or admissible(adm) != p['admissible_before']:
                        raise ValueError(f"Prefix restore mismatch: {c['key']}")
                    c.update(obs=obs, raw_obs=obs, admissible=adm)
                    live[c['key']] = c
                forced = {c['key']: c['point']['arms'][c['arm']]['action'] for c in new}
                returned = pool.step(forced)
                for c in new:
                    expected = c['point']['arms'][c['arm']]
                    obs, adm, won, done = returned[c['key']]
                    if (obs != expected['observation_after'] or won or done
                            or admissible(adm) != expected['admissible_after']):
                        raise ValueError(f"Forced branch no longer matches peek: {c['key']}")
                    advance(c, forced[c['key']], True, 'forced', None, None, returned[c['key']], forced=True)
            active = [c for c in live.values() if not c['done']]
            if active:
                prompts = [common.build_prompt(c['point']['task'], c['memory'], c['obs'], c['admissible'])
                           for c in active]
                seeds = [seed_for(c['point']['index'], c['arm'], c['rep'], c['steps_taken']) for c in active]
                for c, prompt in zip(active, prompts):
                    if c['first_prompt'] is None:
                        c['first_prompt'] = prompt
                outputs = policy.sample(prompts, seeds, logprobs=False)
                if len(outputs) != len(active):
                    raise ValueError('Model returned wrong batch size')
                parsed = [common.parse(o[0]['text']) for o in outputs]
                returned = pool.step({c['key']: parsed[i][0] for i, c in enumerate(active)})
                for c, output, (action, valid, fmt), seed in zip(active, outputs, parsed, seeds):
                    count = len(output[0]['token_ids'])
                    advance(c, action, valid, fmt, output[0]['text'], seed, returned[c['key']], token_count=count)
                    generated_tokens += count
                generated_steps += len(active)
            finished = [key for key, c in live.items() if c['done']]
            if finished:
                pool.close(finished)
                for key in finished:
                    c = live.pop(key)
                    record = dict(status='complete', protocol=VERSION, run_fingerprint=fingerprint,
                                  design_sha256=design['design_sha256'], dry=args.dry, key=key,
                                  point_id=c['point']['id'], point_index=c['point']['index'],
                                  traj_id=c['point']['traj_id'], task_id=c['point']['task_id'],
                                  turn=c['point']['turn'], arm=c['arm'], condition=c['condition'], rep=c['rep'],
                                  forced_action=c['point']['arms'][c['arm']]['action'],
                                  clue=c['point']['clue']['text'], first_prompt=c['first_prompt'],
                                  won=c['won'], stop_reason=c['stop_reason'], final_episode_step=c['steps_taken'],
                                  generated_steps=len(c['transitions']) - 1,
                                  generated_tokens=sum(t['generated_tokens'] for t in c['transitions']),
                                  transitions=c['transitions'])
                    write(c['path'], record)
                    n_finished += 1
            now = time.monotonic()
            if now - last_update > 20 or not pending and not live:
                progress = dict(complete=n_finished, total=total, active=len(live), queued=len(pending),
                                generated_steps_this_launch=generated_steps,
                                generated_tokens_this_launch=generated_tokens,
                                elapsed_seconds_this_launch=round(now - start, 1))
                write(out / 'progress.json', progress)
                print(progress, flush=True)
                last_update = now
    finally:
        # Completed continuations are durable even when a launch is interrupted.
        if pool is not None:
            pool.shutdown()
        if policy is not None:
            policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=HERE / 'design.json')
    parser.add_argument('--run-name', default='main_32x16')
    parser.add_argument('--reps', type=int, default=16)
    parser.add_argument('--limit-points', type=int)
    parser.add_argument('--point-id', action='append', help='Run selected design IDs; may be repeated')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.85)
    parser.add_argument('--max-model-len', type=int, default=6144)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('--run-name must be a simple name, not a path')
    if not (0 < args.reps < 1000 and args.workers > 0 and args.concurrency > 0):
        parser.error('reps must be 1..999; workers and concurrency must be positive')
    if args.limit_points is not None and args.limit_points < 1:
        parser.error('--limit-points must be positive')
    design = load_design(args.design)
    verify_inputs(design)
    points = design['points']
    if args.point_id:
        if args.limit_points:
            parser.error('Use either --point-id or --limit-points')
        unknown = set(args.point_id) - {p['id'] for p in points}
        if unknown:
            parser.error(f'Unknown point IDs: {sorted(unknown)}')
        points = [p for p in points if p['id'] in args.point_id]
    elif args.limit_points:
        points = points[:args.limit_points]
    if args.check_only:
        verify_environment(points)
        print('CPU preflight passed; no model loaded.')
        return
    if not args.dry and not (Path(common.MODEL_PATH) / 'config.json').exists():
        parser.error(f'Model not found: {common.MODEL_PATH}; set M0_MODEL before starting')
    out = HERE / 'runs' / ('dry_' + args.run_name if args.dry else args.run_name)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), HERE / 'protocol.py'] + [ROOT / 'scripts' / name for name in
             ('common.py', 'policy.py', 'alfenv.py', 'envpool.py')]
    config = dict(protocol=VERSION, design_sha256=design['design_sha256'],
                  point_ids=[p['id'] for p in points], reps=args.reps, dry=args.dry,
                  model='random-admissible' if args.dry else common.MODEL_PATH,
                  temperature=common.TEMPERATURE, max_steps=common.MAX_STEPS,
                  history_length=common.HISTORY_LENGTH, max_response_tokens=common.MAX_RESPONSE_TOKENS,
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  paired_condition_seeds=True,
                  code_sha256={str(f.relative_to(ROOT)): file_hash(f) for f in files})
    if not args.dry:
        config['model_config_sha256'] = file_hash(Path(common.MODEL_PATH) / 'config.json')
    fingerprint = digest(config)
    with (out / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Another process is writing {out}')
        manifest = out / 'run_manifest.json'
        if manifest.exists():
            if read(manifest)['run_fingerprint'] != fingerprint:
                parser.error('Run configuration or runtime code changed; use a new --run-name')
        else:
            write(manifest, dict(run_fingerprint=fingerprint, config=config, design=design))
        execute(args, design, points, out, fingerprint)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
