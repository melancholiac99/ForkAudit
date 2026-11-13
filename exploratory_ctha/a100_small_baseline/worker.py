"""One isolated baseline-RL action; no auxiliary predictor or CTHA path."""
import argparse
import math
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from small_common import ENGINEERING, FAST, HERE, MODEL, PILOT, ROOT, configure, cpu_model_lock, emit, fingerprint, hardware, load, save, sha, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['collect', 'prepare', 'train'])
    p.add_argument('--arm', choices=['grpo', 'gigpo'], required=True)
    p.add_argument('--iteration', type=int, default=0)
    p.add_argument('--eval', action='store_true')
    p.add_argument('--base-eval', action='store_true')
    a = p.parse_args()
    assert load(ENGINEERING / 'verification.json')['Engineering'] == 'PASS'
    reg = load(PILOT / 'registration.json')
    assert reg['source_sha256'] == sources(), 'Source changed after registration'
    assert 0 <= a.iteration < 4
    arm = PILOT / a.arm
    rnd = arm / f'round_{a.iteration:03d}'
    prior = arm / f'round_{a.iteration - 1:03d}/adapter' if a.iteration else None
    backend, util = configure(arm)
    config = load(PILOT / 'frozen_config.json')
    if a.action == 'collect':
        module = backend.legacy('collect')
        module.SamplingPolicyOriginal = module.SamplingPolicy
        evaluation = a.eval or a.base_eval
        assert not a.base_eval or (a.arm == 'grpo' and a.iteration == 0)
        adapter = None if a.base_eval else rnd / 'adapter' if evaluation else prior
        out = PILOT / 'base_train_eval' if a.base_eval else arm / 'final_train_eval' if evaluation else rnd / 'rollouts'
        module.SamplingPolicy = lambda model, adp, length, temp, util: backend.Policy(module, config, model, adp, length, temp, util, out)
        original = module.provenance

        def provenance(args, games):
            value = original(args, games)
            value.pop('fingerprint')
            value.update(script=str(Path(__file__).relative_to(ROOT)), source_sha256=sha(Path(__file__)),
                a100_config=config, a100_config_fingerprint=fingerprint(config),
                precision=f'float32_tf32x3_stages{config["stages"]}',
                worker_sha256=sha(backend.HERE / 'worker.py'),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=hardware(),
                baseline_rl_protocol_sha256=sha(HERE / 'PLAN.json'), arm='shared_base' if a.base_eval else a.arm)
            value['fingerprint'] = fingerprint(value)
            return value

        module.provenance = provenance
        sys.argv = ['collect.py', '--model', str(MODEL), '--games', str(PILOT / 'train_games.json'),
            '--split', 'train', '--iteration', str(0 if evaluation else a.iteration), '--out', str(out),
            '--reps', '4', '--workers', '16', '--chunk-games', '16', '--max-steps', '50',
            '--max-model-len', '6144', '--temperature', '0.7', '--gpu-util', str(config['gpu_memory_utilization']),
            '--seed-namespace', 'baseline_rl_fixed_train_eval_v1' if evaluation else 'train']
        if adapter:
            sys.argv += ['--adapter', str(adapter)]
        module.main()
        summary = load(out / 'summary.json')
        emit(arm, 'shared_base' if a.base_eval else a.arm,
             {'eval/fixed_train_success' if evaluation else 'rollout/train_success': summary['success_rate'],
              'rollout/env_steps': summary['env_steps']}, 0 if a.base_eval else 4 if evaluation else a.iteration)
    elif a.action == 'prepare':
        # No make_batch/CTHA predictor: original base-advantage function + exact stored behavior.
        from habits import load_episodes
        from habits_v2 import action_base_advantages
        from prepare import validate_train
        current = load_episodes(rnd / 'rollouts', a.iteration)
        validate_train(current, a.iteration)
        summary = load(rnd / 'rollouts/summary.json')
        expected = sha(prior / 'adapter_model.safetensors') if prior else None
        assert summary['provenance']['adapter_sha256'] == expected
        assert summary['provenance']['a100_config_fingerprint'] == fingerprint(config)
        assert len(current) == 16
        adv, failed, diag = action_base_advantages(current, base=a.arm, local_weight=1., gamma=1.)
        rows = []
        for e in current:
            for i, step in enumerate(e['steps']):
                value = adv[(e['episode_id'], i)]
                assert math.isfinite(value)
                rows.append(dict(episode_id=e['episode_id'], turn=step['turn'], prompt=step['prompt'],
                    response_token_ids=step['response_token_ids'], behavior_logprobs=step['behavior_logprobs'],
                    prompt_token_ids=step['prompt_token_ids'], finish_reason=step['finish_reason'],
                    advantage=value, advantage_episode=value, advantage_habit=0.))
        out = rnd / 'prepared'
        out.mkdir(parents=True, exist_ok=False)
        import json
        (out / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        save(out / 'advantage_report.json', dict(schema='ctha-v3-fp32', mode='grpo', base=a.arm,
            samples_sha256=sha(out / 'samples.jsonl'), train_only=True, nonzero_aux_samples=0,
            total_rows=len(rows), zero_advantage_rows=sum(r['advantage'] == 0 for r in rows),
            source_hashes={p.name: sha(p) for p in (rnd / 'rollouts').glob('s*_r*.json')},
            diagnostic=diag, current_ctha_gate='NOT_RUN', P0='NOT_USED'), immutable=True)
        emit(arm, a.arm, {'batch/all_rows': len(rows), 'batch/zero_advantage_rows': sum(r['advantage'] == 0 for r in rows)}, a.iteration)
    else:
        import torch
        from transformers import AutoModelForCausalLM
        original_load = AutoModelForCausalLM.from_pretrained

        def load_gpu(*args, **kw):
            with cpu_model_lock():
                return original_load(*args, **kw).to('cuda')

        AutoModelForCausalLM.from_pretrained = staticmethod(load_gpu)
        original_save = util.save

        def instrument(path, value):
            original_save(path, value)
            if Path(path).name == 'progress.json':
                row = value['last']
                emit(arm, a.arm, {**{'train/' + key: row[key] for key in ['loss', 'grad_norm', 'batch_rows']},
                    **{'stability/' + key: row[key] for key in ['sampled_behavior_k3', 'sampled_reference_k3', 'clip_fraction']},
                    'train/round': a.iteration, 'train/elapsed_s': value['elapsed_s']}, row['total_updates'])
            elif Path(path).name == 'parity.json':
                emit(arm, a.arm, {'parity/' + k: value[k] for k in ['mean_abs', 'p99_abs', 'max_abs', 'passed']}, a.iteration)

        util.save = instrument
        from engineering_worker import new_trainer
        trainer = new_trainer()
        trainer.train(SimpleNamespace(data=rnd / 'prepared/samples.jsonl', summary=rnd / 'rollouts/summary.json',
            out=rnd / 'adapter', adapter=prior, seed=a.iteration, probe=False))
        log = load(rnd / 'adapter/train_log.json')
        expected_updates = math.ceil(log['n_source_steps'] / 64)
        assert log['n_trained_rows'] == log['n_source_steps'] and log['updates_this_round'] == expected_updates
        assert log['optimizer_resumed'] == bool(prior)
        if prior:
            prior_log = load(prior / 'train_log.json')
            assert log['total_updates'] == prior_log['total_updates'] + expected_updates


if __name__ == '__main__':
    main()
