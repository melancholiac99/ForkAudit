"""Explicitly authorized, ten-minute BF16 bypass probe; preserve the live pilot."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / '.runtime/runs/a100_bf16_bypass_r2'
PILOT = ROOT / '.runtime/runs/ctha_a100_small_baseline_r1'


def load(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    controller = load(PILOT / 'status.json')['pid']
    assert 'a100_small_baseline/run.py' in Path(f'/proc/{controller}/cmdline').read_bytes().decode()
    publication = load(ROOT / '.runtime/publication_staging/bf16_bypass_publication.json')
    assert publication['verified_remote']
    import probe
    assert publication['source_sha256'] == probe.sources()
    record = dict(start=time.time(), controller=controller, stages=[], paused_children=[],
        protocol_sha256=hashlib.sha256((HERE / 'PLAN.json').read_bytes()).hexdigest(),
        publication=publication)
    (OUT / 'admission.json').write_text(json.dumps(dict(user_authorized_new_protocol=True,
        authorization='Explicit user answer approving the new four-step BF16 bypass protocol',
        raw_probability_error_is_diagnostic=True, original_BF16_result='FAIL retained'), indent=2) + '\n')
    paused = False
    try:
        os.kill(controller, signal.SIGSTOP)
        paused = True
        for path in PILOT.glob('gigpo_r*_train_launch.json'):
            pid = load(path)['pid']
            proc = Path(f'/proc/{pid}')
            if not proc.exists() or 'State:\tZ' in (proc / 'status').read_text():
                continue
            command = (proc / 'cmdline').read_bytes().decode().replace('\x00', ' ')
            if 'a100_small_baseline/worker.py train --arm gigpo' in command:
                os.kill(pid, signal.SIGSTOP)
                record['paused_children'].append(pid)
        assert record['paused_children'], 'Need a safely paused GPU1 training worker; inspect live stage before retry'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='1', NVIDIA_TF32_OVERRIDE='0',
            TRITON_F32_DEFAULT='tf32x3', PYTHONPATH=str(ROOT))
        for action in ['collect', 'prepare', 'train', 'checkpoint', 'updated_sample', 'updated_score']:
            remaining = 600 - (time.time() - record['start'])
            if remaining <= 0:
                raise TimeoutError('Ten-minute diagnostic execution ceiling')
            with (OUT / (action + '.log')).open('x') as log:
                proc = subprocess.Popen([sys.executable, '-u', str(HERE / 'probe.py'), action],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                    raise
            record['stages'].append(dict(action=action, code=code, time=time.time()))
            print(action, code, flush=True)
            if code:
                raise RuntimeError(f'{action} exited {code}; gates unchanged')
        # Independent arithmetic/file checks, separate from the trainer's verdict flags.
        import numpy as np
        import torch
        d = OUT / 'direction_probe'
        train = load(d / 'train_log.json')
        assert train['complete'] and train['total_updates'] == train['updates_this_round'] == 4
        assert train['n_trained_rows'] == 8 and train['nonzero_advantage_rows'] == 8
        direction = load(d / 'post_update.json')['direction']
        pos = [r['delta_mean_response_logprob'] for r in direction if r['advantage'] > 0]
        neg = [r['delta_mean_response_logprob'] for r in direction if r['advantage'] < 0]
        assert len(pos) == len(neg) == 4 and np.mean(pos) > 0 > np.mean(neg)
        assert load(d / 'direction_check.json')['passed']
        assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) and r['grad_norm'] > 0 for r in train['logs'])
        for row in train['logs'] + [load(d / 'post_update.json')['stability']]:
            assert np.isfinite(row['sampled_behavior_k3']) and row['sampled_behavior_k3'] <= .02
            assert np.isfinite(row['clip_fraction']) and row['clip_fraction'] <= .30
        drifts, clipped = [], []
        for row in load(d / 'final_scoring.json'):
            lp, old = [torch.tensor(row[k]) for k in ['logprobs', 'behavior']]
            delta = old - lp
            drifts.extend((delta.clamp(-20, 20).exp() - delta - 1).tolist())
            ratio = (lp - old).exp()
            clipped.extend(((ratio < .8) | (ratio > 1.2)).float().tolist())
        post = load(d / 'post_update.json')['stability']
        assert abs(np.mean(drifts) - post['sampled_behavior_k3']) < 1e-10
        assert abs(np.mean(clipped) - post['clip_fraction']) < 1e-10
        live, restored = load(d / 'live_checkpoint_verification.json'), load(OUT / 'checkpoint_verification.json')
        assert live['passed'] and restored['passed'] and restored['moments_equal'] and restored['next_step_equal']
        assert restored['restored_updates'] == 4 and restored['next_test_updates'] == 5
        for name, key in [('adapter_model.safetensors', 'weights_sha'), ('optimizer.pt', 'optimizer_sha')]:
            assert hashlib.sha256((d / name).read_bytes()).hexdigest() == train[key]
        replies = load(OUT / 'updated_replies.json')
        scores = load(OUT / 'updated_hf_logprobs.json')
        assert len(replies) == len(scores) == 8
        ds, cs, errors = [], [], []
        seq_ratios = []
        for row, values in zip(replies, scores):
            lp, old = torch.tensor(values), torch.tensor(row['behavior_logprobs'])
            diff = old - lp
            ds.extend((diff.clamp(-20, 20).exp() - diff - 1).tolist())
            ratio = (lp - old).exp()
            cs.extend(((ratio < .8) | (ratio > 1.2)).float().tolist())
            errors.extend((lp - old).abs().tolist())
            seq_ratios.append(float((lp - old).sum().exp()))
        assert np.mean(ds) <= .02 and np.mean(cs) <= .30
        updated = dict(sampled_behavior_k3=float(np.mean(ds)), clip_fraction=float(np.mean(cs)),
            mean_abs=float(np.mean(errors)), p99_abs=float(np.quantile(errors, .99)), max_abs=float(np.max(errors)),
            sequence_ratio_min=min(seq_ratios), sequence_ratio_max=max(seq_ratios),
            sequence_ESS_fraction=float(sum(seq_ratios)**2 / (8 * sum(r*r for r in seq_ratios))))
        record.update(Limited_BF16_probe='PASS', Engineering='INCOMPLETE: new-precision cross-GPU/cross-batch repeat not run',
            Original_strict_BF16='FAIL retained; not reclassified', Learning='NOT_RUN', CTHA_online_efficacy='NOT_TESTED',
            adopted=False, optimizer_updates=4, direction=load(d / 'direction_check.json'),
            train_seconds=train['elapsed_s'], post_update_stability=post, updated_parity=updated,
            checkpoint_weights_sha256=train['weights_sha'], checkpoint_optimizer_sha256=train['optimizer_sha'])
    except Exception as e:
        record.update(Limited_BF16_probe='FAIL_OR_INCOMPLETE', error=repr(e), adopted=False)
        print(record['error'], flush=True)
    finally:
        for pid in record['paused_children']:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        if paused:
            os.kill(controller, signal.SIGCONT)
        record.update(finished=time.time(), old_controller_resumed=True)
        (OUT / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'publication'}), flush=True)


if __name__ == '__main__':
    main()
