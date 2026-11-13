"""Independently recompute acceptance under the new full-response protocol."""
import math
import re
import sys
from protocol_common import ENGINEERING, FAST, HERE, R2, ROOT, fingerprint, load, save, sha, sources


def verify():
    import numpy as np
    import torch
    registration = load(ENGINEERING / 'registration.json')
    assert registration['source_sha256'] == sources()
    for relative, expected in registration['reused_evidence_sha256'].items():
        assert sha(R2 / relative) == expected == sha(ENGINEERING / relative)
    assert load(R2 / 'failure_review.json')['Engineering'] == 'FAIL'
    assert load(ENGINEERING / 'frozen_config.json') == load(R2 / 'frozen_config.json')
    original = load(FAST / 'PLAN.json')
    plan = load(HERE / 'PLAN.json')
    for field in ['optimizer', 'parity_gate', 'stability']:
        assert plan[field] == original[field]
    checks = {}
    for name in ['ctha', 'v3', 'fp32', 'fp32_fast', 'protocol']:
        text = (ENGINEERING / ('cpu_' + name + '.log')).read_text()
        assert re.search(r'Ran (\d+) tests?', text) and '\nOK\n' in text
    checks['cpu_regressions'] = True
    # Reconstruct R2 cross-GPU/cross-batch comparisons rather than trusting flags.
    sys.path.insert(0, str(ROOT / 'scripts'))
    import common
    episode_sets = {}
    config_fp = fingerprint(load(ENGINEERING / 'frozen_config.json'))
    for label, count in [('repeat_a', 8), ('repeat_b', 8), ('initial_train', 64)]:
        folder = ENGINEERING / label
        episodes = {p.name: load(p) for p in folder.glob('s*_r*.json')}
        assert len(episodes) == count == load(folder / 'summary.json')['n_episodes']
        for e in episodes.values():
            assert e['provenance']['a100_config_fingerprint'] == config_fp
            assert e['provenance']['adapter_sha256'] is None
            assert e['provenance']['split'] == 'train' and not e['provenance']['dry']
            memory = []
            for s in e['steps']:
                assert s['prompt'] == common.build_prompt(e['task'], memory, s['obs_before'], s['admissible'])
                assert len(s['response_token_ids']) == len(s['behavior_logprobs']) > 0
                assert all(math.isfinite(x) for x in s['behavior_logprobs'])
                memory.append(dict(text_obs=s['obs_before'], action=s['action']))
            assert e['won'] == e['steps'][-1]['won_after']
        episode_sets[label] = episodes
    for label in ['repeat_b', 'initial_train']:
        for name, a in episode_sets['repeat_a'].items():
            b = episode_sets[label][name]
            assert a['won'] == b['won'] and len(a['steps']) == len(b['steps'])
            for x, y in zip(a['steps'], b['steps']):
                assert all(x[k] == y[k] for k in ['response_token_ids', 'prompt_token_ids', 'behavior_logprobs', 'sample_seed', 'action', 'obs_before', 'obs_after', 'won_after'])
    checks['hash_verified_real_A100_repeated_cross_batch_evidence'] = True
    probe = ENGINEERING / 'direction_probe'
    log = load(probe / 'train_log.json')
    assert log['complete'] and log['n_trained_rows'] == 8
    assert log['updates_this_round'] == log['total_updates'] == 4
    assert load(probe / 'probe_indices.json') == load(R2 / 'direction_probe/probe_indices.json')
    assert log['weights_sha'] == sha(probe / 'adapter_model.safetensors')
    assert log['optimizer_sha'] == sha(probe / 'optimizer.pt')
    live = load(probe / 'live_checkpoint_verification.json')
    restored = load(ENGINEERING / 'checkpoint_verification.json')
    assert live['passed'] and live['live_vs_disk_moments_equal'] and live['live_vs_disk_weights_equal']
    assert restored['passed'] and restored['moments_equal'] and restored['next_step_equal']
    assert restored['restored_updates'] == 4 and restored['next_test_updates'] == 5
    assert restored['weights_sha256'] == log['weights_sha'] == live['weights_sha256']
    assert restored['optimizer_sha256'] == log['optimizer_sha'] == live['optimizer_sha256']
    checks['live_disk_weights_moments_and_actual_cpu_restore'] = True
    post = load(probe / 'post_update.json')
    means = {}
    for field in ['delta_mean_response_logprob', 'delta_mean_action_logprob']:
        for label, sign in [('positive', 1), ('negative', -1)]:
            values = [r[field] for r in post['direction'] if r['advantage'] * sign > 0]
            assert len(values) == 4 and all(math.isfinite(v) for v in values)
            means[label + '_' + field] = float(np.mean(values))
    assert means['positive_delta_mean_response_logprob'] > 0 > means['negative_delta_mean_response_logprob']
    recorded = load(probe / 'direction_check.json')
    assert recorded['passed'] and recorded['full_response_passed'] and not recorded['action_span_blocking']
    for key, value in means.items():
        assert math.isclose(value, recorded[key], abs_tol=1e-12)
    checks['signed_full_response_direction'] = True
    for row in [*log['logs'], post['stability']]:
        assert math.isfinite(row['sampled_behavior_k3']) and row['sampled_behavior_k3'] <= .02
        assert math.isfinite(row['clip_fraction']) and row['clip_fraction'] <= .30
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['grad_norm'] > 0 and math.isfinite(r['sampled_reference_k3']) for r in log['logs'])
    checks['finite_gradients_behavior_KL_and_clip_stability'] = True
    from verify import parity_from_rows

    def check(rows, reported):
        actual = parity_from_rows(rows)
        for key in ['mean_abs', 'p99_abs', 'max_abs']:
            assert math.isclose(actual[key], reported[key], rel_tol=1e-7, abs_tol=1e-12)
        assert actual['mean_abs'] <= .01 and actual['p99_abs'] <= .05 and actual['max_abs'] <= .25
        assert reported['passed'] and actual['tokens'] == reported['tokens']
        return actual

    base = check(load(probe / 'base_parity_replies.json'), load(probe / 'parity.json'))
    rows = load(ENGINEERING / 'adapter_parity/replies.json')
    hf = load(ENGINEERING / 'adapter_parity/hf_logprobs.json')
    merged = load(ENGINEERING / 'adapter_parity/hf_merged_logprobs.json')
    assert len(rows) == len(hf) == len(merged) == 8
    updated = check([{**r, 'hf_logprobs': v} for r, v in zip(rows, hf)], load(ENGINEERING / 'adapter_parity/verification.json'))
    merge = check([{**r, 'behavior_logprobs': m, 'hf_logprobs': v} for r, m, v in zip(rows, merged, hf)], load(ENGINEERING / 'adapter_parity/merge_verification.json'))
    checks['base_updated_HF_vLLM_and_merged_unmerged_parity'] = True
    # Independently recompute final behavior k3, clip fraction and reference estimator.
    scored = load(probe / 'final_scoring.json')
    assert len(scored) == 8
    drifts, clips, refs = [], [], []
    for row in scored:
        lp, old, ref = [torch.tensor(row[k]) for k in ['logprobs', 'behavior', 'reference']]
        assert lp.shape == old.shape == ref.shape
        assert torch.isfinite(lp).all() and torch.isfinite(old).all() and torch.isfinite(ref).all()
        d = old - lp
        drifts.extend((d.clamp(-20, 20).exp() - d - 1).tolist())
        ratio = (lp - old).exp()
        clips.extend(((ratio < .8) | (ratio > 1.2)).float().tolist())
        d = ref - lp
        refs.extend((d.clamp(-20, 20).exp() - d - 1).tolist())
    assert math.isclose(float(np.mean(drifts)), post['stability']['sampled_behavior_k3'], abs_tol=1e-12)
    assert math.isclose(float(np.mean(clips)), post['stability']['clip_fraction'], abs_tol=1e-12)
    checks['sampled_KL_and_clip_independently_recomputed'] = True
    result = dict(Engineering='PASS', protocol=plan['version'], R2='FAIL (unchanged)', Learning='NOT_RUN',
         Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED', checks=checks,
         action_span_diagnostic=means, action_span_blocking=False,
         base_parity=base, updated_parity=updated, merged_unmerged_parity=merge,
         weights_sha256=log['weights_sha'], optimizer_sha256=log['optimizer_sha'],
         final_sampled_reference_k3=float(np.mean(refs)),
         source_sha256=sources(), config_fingerprint=config_fp,
         checkpoint_scope='Actual CPU restore + fixed engineering gradient next-step equality; no second online round.')
    save(ENGINEERING / 'verification.json', result, immutable=True)
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
