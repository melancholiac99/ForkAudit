"""Independently recompute saved probability diagnostics; never infer efficacy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]


def ess(log_weights):
    weights = np.exp(log_weights - np.max(log_weights))
    return float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum()))


def recompute(actor, rollout, mask):
    mask = mask.astype(bool)
    delta = actor.astype(np.float64) - rollout.astype(np.float64)
    x = delta[mask]
    assert len(x) and np.isfinite(x).all()
    ratio = np.exp(x)
    lengths = mask.sum(-1)
    response_logs = np.where(mask, delta, 0).sum(-1)[lengths > 0]
    result = dict(token_count=len(x), response_count=len(response_logs),
        sampled_behavior_k3=float(np.mean(np.expm1(-x) + x)),
        clip_fraction=float(np.mean((ratio < .8) | (ratio > 1.2))),
        token_ESS_fraction=ess(x), response_ESS_fraction=ess(response_logs),
        token_ratio_min=float(ratio.min()), token_ratio_max=float(ratio.max()),
        response_log_ratio_min=float(response_logs.min()), response_log_ratio_max=float(response_logs.max()),
        long_response_count=int((lengths >= 256).sum()))
    for name, q in [('p01', .01), ('p50', .5), ('p99', .99), ('p999', .999)]:
        result['token_ratio_' + name] = float(np.quantile(ratio, q))
    long_mask = mask & (lengths >= 256)[:, None]
    if long_mask.any():
        d = delta[long_mask]
        r = np.exp(d)
        result.update(long_sampled_k3=float(np.mean(np.expm1(-d) + d)),
            long_clip_fraction=float(np.mean((r < .8) | (r > 1.2))), long_token_ESS_fraction=ess(d))
    extra = dict(raw_logprob_error_mean=float(np.abs(x).mean()), raw_logprob_error_p99=float(np.quantile(np.abs(x), .99)),
        raw_logprob_error_max=float(np.abs(x).max()),
        sampled_KL_rollout_to_actor_k1=float(-x.mean()),
        opposite_orientation_k3=float(np.mean(np.expm1(x) - x)),
        response_geometric_ratio_ESS_fraction=ess(response_logs / lengths[lengths > 0]),
        token_ESS_absolute=ess(x) * len(x), response_ESS_absolute=ess(response_logs) * len(response_logs),
        longest_response_tokens=int(lengths.max()))
    return result, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    exit_status = json.loads((run / 'exit.json').read_text())
    registration = json.loads((run / 'registration.json').read_text())
    rows = []
    if (run / 'mismatch.jsonl').exists():
        for line in (run / 'mismatch.jsonl').read_text().splitlines():
            reported = json.loads(line)
            path = run / f"probabilities_step_{reported['step']}.npz"
            with np.load(path) as values:
                actual, extra = recompute(values['actor'], values['rollout'], values['mask'])
            for key, value in actual.items():
                assert math.isclose(reported[key], value, rel_tol=1e-9, abs_tol=1e-10), (key, reported[key], value)
            rows.append(dict(step=reported['step'], reported_equals_independent=True,
                gates_passed=actual['sampled_behavior_k3'] <= .02 and actual['clip_fraction'] <= .30,
                **actual, **extra, evidence=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    logdir = ROOT / '.runtime/tensorboard/ctha-a100-r2' / run.name
    scalars = {}
    if logdir.exists():
        accumulator = EventAccumulator(str(logdir), size_guidance={'scalars': 0}).Reload()
        scalars = {tag: [dict(step=e.step, value=e.value, wall_time=e.wall_time) for e in accumulator.Scalars(tag)]
                   for tag in accumulator.Tags().get('scalars', [])}
    training_steps = [r['value'] for r in scalars.get('training/global_step', [])]
    finite_gradients = scalars.get('actor/grad_norm', [])
    execution_pass = (exit_status['returncode'] == 0 and bool(training_steps) and bool(rows)
                      and all(row['gates_passed'] for row in rows)
                      and bool(finite_gradients) and all(math.isfinite(e['value']) for e in finite_gradients))
    result = dict(run=run.name, configuration_sha256=registration['configuration_sha256'],
        native_one_iteration_pipeline='PASS' if execution_pass else 'FAIL_OR_INCOMPLETE',
        Engineering='INCOMPLETE', Learning='NOT_RUN', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED',
        interpretation='Engineering iteration only. Full-model cross-GPU/batch reproducibility, post-update alignment and actual native checkpoint recovery remain unverified.',
        importance_ratio='HF actor / actual sampler; inherited k3 is a diagnostic, not exact distribution KL. Response product ESS differs from geometric length-normalized ESS.',
        probability_checks=rows, optimizer_steps='No native checkpoint in one-iteration smoke; outer trainer iteration is not an AdamW step count.',
        tensorboard=scalars)
    (run / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    with (run / 'metrics.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['metric', 'step', 'value', 'wall_time'])
        for key, events in scalars.items():
            for event in events:
                writer.writerow([key, event['step'], event['value'], event['wall_time']])
    print(json.dumps({k: v for k, v in result.items() if k != 'tensorboard'}, indent=2))


if __name__ == '__main__':
    main()
