"""P1 action worker: unchanged v3 loss, local datasets and explicit cold start."""
import argparse
import contextlib
import fcntl
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from p1_common import DATA, ENGINEERING, FAST, HERE, MODEL, PARENT, ROOT, VALIDATION, load, save, sha, sources


@contextlib.contextmanager
def cpu_model_lock():
    path = ROOT / '.runtime/locks/full_cpu_model.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
        fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'prepare', 'train'])
    parser.add_argument('--arm', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--split', default='train', choices=['train', 'tune'])
    parser.add_argument('--base-eval', action='store_true')
    args = parser.parse_args()
    plan = load(HERE / 'PLAN.json')
    arm = next(a for a in plan['arms'] if a['name'] == args.arm)
    arm_root = DATA / (args.arm + '_s0')
    rnd = arm_root / f'round_{args.iteration:03d}'
    prior = arm_root / f'round_{args.iteration - 1:03d}/adapter' if args.iteration else None
    assert load(ENGINEERING / 'verification.json')['Engineering'] == 'PASS'
    registration = load(DATA / 'registration.json')
    assert sources() == registration['source_sha256'], 'P1 sources changed after registration'
    os.environ['CTHA_A100_ROOT'] = str(arm_root)
    sys.path.insert(0, str(VALIDATION))
    import a100_common as common
    common.DATA = arm_root
    import backend
    backend.DATA = arm_root
    util = backend.configure()
    util.DATA = arm_root
    util.source_hashes = sources
    config = load(DATA / 'frozen_config.json')

    def metric(values):
        arm_root.mkdir(parents=True, exist_ok=True)
        row = dict(time=time.time(), arm=args.arm, round=args.iteration, **values)
        with (arm_root / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    if args.action == 'collect':
        import merged_policy
        original_export = merged_policy.export_policy

        def locked_export(base, adapter):
            with cpu_model_lock():
                return original_export(base, adapter)

        merged_policy.export_policy = locked_export
        module = backend.legacy('collect')
        module.SamplingPolicyOriginal = module.SamplingPolicy
        if args.base_eval:
            assert args.split == 'tune' and args.iteration == 0
        adapter = None if args.base_eval else (prior if args.split == 'train' else rnd / 'adapter')
        out = DATA / 'base_tune' if args.base_eval else rnd / ('rollouts' if args.split == 'train' else 'eval_tune')
        module.SamplingPolicy = lambda model, adp, length, temperature, gpu_util: backend.Policy(
            module, config, model, adp, length, temperature, gpu_util, out)
        original_provenance = module.provenance

        def provenance(a, games):
            value = original_provenance(a, games)
            value.pop('fingerprint')
            value.update(script='exploratory_ctha/a100_p1/worker.py', source_sha256=sha(Path(__file__)),
                         a100_config=config, a100_config_fingerprint=common.fingerprint(config),
                         precision=f'float32_tf32x3_stages{config["stages"]}',
                         worker_sha256=sha(VALIDATION / 'worker.py'),
                         visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), gpu=common.hardware(),
                         arm='shared_base' if args.base_eval else args.arm, p1_protocol_sha256=sha(HERE / 'PLAN.json'))
            value['fingerprint'] = common.fingerprint(value)
            return value

        module.provenance = provenance
        games = rnd / 'games.json' if args.split == 'train' else DATA / 'tune_games.json'
        sys.argv = ['collect.py', '--model', str(MODEL), '--games', str(games), '--split', args.split,
                    '--iteration', str(args.iteration if args.split == 'train' else 0),
                    '--out', str(out), '--reps', str(plan['reps'] if args.split == 'train' else 2),
                    '--workers', '16', '--chunk-games', '16', '--max-steps', '50',
                    '--max-model-len', '6144', '--temperature', '0.7',
                    '--seed-namespace', 'train' if args.split == 'train' else 'eval',
                    '--gpu-util', str(config['gpu_memory_utilization'])]
        if adapter:
            sys.argv += ['--adapter', str(adapter)]
        module.main()
        summary = load(out / 'summary.json')
        if args.split == 'train':
            metric(dict(phase='rollout', **{'rollout/success': summary['success_rate'], 'rollout/env_steps': summary['env_steps']}))
        elif args.base_eval:
            save(DATA / 'base_tune/measurement.json', dict(success=summary['success_rate'],
                 episodes=summary['n_episodes'], scope='Frozen shared base development evaluation'), immutable=True)
        else:
            log = load(rnd / 'adapter/train_log.json')
            metric(dict(phase='tune', optimizer_step=log['total_updates'], **{'tune/success': summary['success_rate']}))
    elif args.action == 'prepare':
        from habits import load_episodes
        from habits_v2 import make_batch, oof_audit
        from prepare import validate_train, effective_omega
        current = load_episodes(rnd / 'rollouts', args.iteration)
        validate_train(current, args.iteration)
        expected_adapter = sha(prior / 'adapter_model.safetensors') if prior else None
        summary = load(rnd / 'rollouts/summary.json')
        assert summary['provenance']['adapter_sha256'] == expected_adapter
        assert summary['provenance']['a100_config_fingerprint'] == registration['config_fingerprint']
        history = []
        historical = []
        if arm['mode'] == 'ctha':
            for i, folder in enumerate(sorted(Path(os.environ['CTHA_P0_ROOT']).glob('round_*'))):
                episodes = load_episodes(folder / 'rollouts', -1000 + i)
                assert all(e['provenance']['split'] == 'train' and e['provenance']['adapter_sha256'] is None for e in episodes)
                historical.extend(episodes)
            assert len(historical) == 768
            for i in range(max(0, args.iteration - plan['history_rounds']), args.iteration):
                old = load_episodes(arm_root / f'round_{i:03d}/rollouts', i)
                validate_train(old, i)
                history.extend(old)
        opts = plan['auxiliary']
        omega = effective_omega(opts['omega'], args.iteration, opts['anneal_start'], opts['anneal_end'])
        rows, report = make_batch(current, historical + history, mode=arm['mode'], base=arm['base'], family=arm['family'],
            feature_window=8, train_horizon=0, aux_horizon=8, alpha=opts['ridge_alpha'],
            min_mixed=opts['min_mixed'], min_pairs=opts['min_pairs'], permutations=opts['permutations'],
            bootstraps=opts['bootstraps'], min_delta=opts['min_delta'], sign_fraction=opts['sign_fraction'],
            omega=omega, clip=opts['clip'], baseline=opts['baseline'], gamma=1., local_weight=1.)
        index = {(e['episode_id'], step['turn']): step for e in current for step in e['steps']}
        for row in rows:
            source = index[row['episode_id'], row['turn']]
            for key in ['behavior_logprobs', 'prompt_token_ids', 'finish_reason']:
                row[key] = source[key]
        report.update(schema='ctha-v3-fp32', train_only=True, iteration=args.iteration,
                      historical_p0_episodes=len(historical), history_episodes=len(history),
                      current_policy_gate=oof_audit(current, family=arm['family'], min_mixed=20, min_pairs=200,
                                                    permutations=12, bootstraps=300),
                      code_sha=sources(), source_hashes={p.name: sha(p) for p in (rnd / 'rollouts').glob('s*_r*.json')})
        out = rnd / 'prepared'
        out.mkdir(exist_ok=False)
        (out / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        report['samples_sha256'] = sha(out / 'samples.jsonl')
        save(out / 'advantage_report.json', report, immutable=True)
        metric(dict(phase='prepare', **{'aux/nonzero_rows': report['nonzero_aux_samples'],
                                      'aux/omega': omega, 'aux/gate_passed': report['gate']['passed']}))
    else:
        import torch
        from transformers import AutoModelForCausalLM
        original_load = AutoModelForCausalLM.from_pretrained

        def locked_gpu_load(*a, **kw):
            with cpu_model_lock():
                model = original_load(*a, **kw)
                return model.to('cuda')

        AutoModelForCausalLM.from_pretrained = staticmethod(locked_gpu_load)
        original_save = util.save

        def instrument(path, value):
            original_save(path, value)
            if Path(path).name == 'progress.json':
                row = value['last']
                metric(dict(phase='train', optimizer_step=row['total_updates'], **{
                    'train/loss': row['loss'], 'train/grad_norm': row['grad_norm'],
                    'stability/k3': row['sampled_behavior_k3'], 'stability/clip_fraction': row['clip_fraction'],
                    'stability/reference_k3': row['sampled_reference_k3'], 'train/elapsed_s': value['elapsed_s']}))
            elif Path(path).name == 'parity.json':
                metric(dict(phase='parity', **{'parity/' + k: value[k] for k in ['mean_abs', 'p99_abs', 'max_abs', 'passed']}))

        util.save = instrument
        trainer = backend.legacy('trainer')
        assert load(FAST / 'PLAN.json')['optimizer'] == plan['optimizer']
        trainer.train(SimpleNamespace(data=rnd / 'prepared/samples.jsonl', summary=rnd / 'rollouts/summary.json',
                                      out=rnd / 'adapter', adapter=prior, seed=args.iteration, probe=False))


if __name__ == '__main__':
    main()
