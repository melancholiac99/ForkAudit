"""Independent reconstruction of acceptance evidence; no learning inference."""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from a100_common import FAST, HERE, ROOT, fingerprint, load, save, sha, sources


def parity_from_rows(rows):
    import numpy as np
    import torch
    errors = []
    for row in rows:
        actual = torch.tensor(row['hf_logprobs'], dtype=torch.float32)
        behavior = torch.tensor(row['behavior_logprobs'], dtype=torch.float32)
        assert len(actual) == len(behavior) == len(row['completion_ids']) > 0
        assert torch.isfinite(actual).all() and torch.isfinite(behavior).all()
        errors.extend((actual - behavior).abs().tolist())
    return dict(mean_abs=float(np.mean(errors)), p99_abs=float(np.quantile(errors, .99)),
                max_abs=float(np.max(errors)), tokens=len(errors))


def verify(data):
    registration = load(data / 'registration.json')
    assert sources() == registration['source_sha256']
    config = load(data / 'frozen_config.json')
    assert fingerprint(config) == registration['config_fingerprint']
    assert config['dtype'] == 'float32'
    plan = registration['scope']
    original = load(FAST / 'PLAN.json')
    assert plan['parity_gate'] == original['parity_gate']
    for key, value in original['stability'].items():
        if key in plan['stability']:
            assert plan['stability'][key] == value
    assert plan['optimizer'] == original['optimizer']
    checks = {}
    test_counts = {}
    for name in ['ctha', 'v3', 'fp32', 'fp32_fast']:
        text = (data / ('cpu_' + name + '.log')).read_text()
        match = re.search(r'Ran (\d+) tests?', text)
        assert match and '\nOK\n' in text
        test_counts[name] = int(match.group(1))
    checks['cpu_regressions'] = True
    cold = load(data / 'cpu_validation.json')
    assert cold['p0_episodes'] == 768 and cold['legacy_round_episodes'] == 64
    assert cold['cold_start_gate']['passed'] and cold['offline_nonzero_aux_samples'] > 0
    assert all(cold['checks'].values())
    assert cold['import_verification_sha256'] == sha(ROOT / '.runtime/imports/import_verification.json')
    checks['original_cold_start_feasibility'] = True
    sys.path.insert(0, str(ROOT / 'scripts'))
    import common
    episode_sets = {}
    allowed_games = {g['game_file'] for g in load(data / 'train_games.json')}
    for label, count in [('repeat_a', 8), ('repeat_b', 8), ('initial_train', 64)]:
        folder = data / label
        episodes = {p.name: load(p) for p in folder.glob('s*_r*.json')}
        summary = load(folder / 'summary.json')
        assert len(episodes) == count == summary['n_episodes']
        expected = {(g['game_file'], r) for g in load(data / ('train_games.json' if count == 64 else 'repeat_games.json'))
                    for r in range(4 if count == 64 else 2)}
        assert {(e['game_file'], e['rep']) for e in episodes.values()} == expected
        for episode in episodes.values():
            provenance = episode['provenance']
            assert provenance['split'] == 'train' and provenance['schema'] == 'ctha-v3-fp32'
            assert not provenance['dry'] and provenance['adapter_sha256'] is None
            assert provenance['a100_config_fingerprint'] == registration['config_fingerprint']
            assert provenance['fingerprint'] == summary['provenance']['fingerprint']
            assert episode['game_file'] in allowed_games
            assert episode['n_turns'] == len(episode['steps']) and 0 < len(episode['steps']) <= 50
            memory = []
            for step in episode['steps']:
                assert step['prompt'] == common.build_prompt(episode['task'], memory, step['obs_before'], step['admissible'])
                assert len(step['response_token_ids']) == len(step['behavior_logprobs']) > 0
                assert all(math.isfinite(x) for x in step['behavior_logprobs'])
                if memory:
                    assert step['obs_before'] == previous['obs_after']
                memory.append(dict(text_obs=step['obs_before'], action=step['action']))
                previous = step
            assert episode['won'] == episode['steps'][-1]['won_after']
        episode_sets[label] = episodes
        checks[label + '_complete'] = True
    fields = ['response_token_ids', 'prompt_token_ids', 'sample_seed', 'action', 'obs_before', 'obs_after', 'won_after']
    for other in ['repeat_b', 'initial_train']:
        for name, a in episode_sets['repeat_a'].items():
            b = episode_sets[other][name]
            assert a['won'] == b['won'] and len(a['steps']) == len(b['steps'])
            assert all(all(x[k] == y[k] for k in fields) for x, y in zip(a['steps'], b['steps']))
        checks['exact_comparison_' + other] = True
    prepared = load(data / 'initial_prepared/advantage_report.json')
    assert prepared['mode'] == 'grpo' and prepared['base'] == 'grpo' and prepared['nonzero_aux_samples'] == 0
    assert prepared['samples_sha256'] == sha(data / 'initial_prepared/samples.jsonl')
    probe = data / 'direction_probe'
    log = load(probe / 'train_log.json')
    assert log['complete'] and log['n_trained_rows'] == 8
    assert log['updates_this_round'] == log['total_updates'] == 4
    assert log['weights_sha'] == sha(probe / 'adapter_model.safetensors')
    assert log['optimizer_sha'] == sha(probe / 'optimizer.pt')
    gate = plan['parity_gate']
    def check_parity(actual, recorded):
        for key in ['mean_abs', 'p99_abs', 'max_abs']:
            assert math.isclose(actual[key], recorded[key], rel_tol=1e-7, abs_tol=1e-12)
        assert actual['tokens'] == recorded['tokens']
        assert actual['mean_abs'] <= gate['mean_absolute_sampled_logprob_error_max']
        assert actual['p99_abs'] <= gate['p99_absolute_sampled_logprob_error_max']
        assert actual['max_abs'] <= gate['max_absolute_sampled_logprob_error_max']
        assert recorded['passed']
    base = parity_from_rows(load(probe / 'base_parity_replies.json'))
    check_parity(base, load(probe / 'parity.json'))
    checks['base_parity_recomputed'] = True
    updated_rows = load(data / 'adapter_parity/replies.json')
    scored = load(data / 'adapter_parity/hf_logprobs.json')
    assert len(updated_rows) == len(scored) == 8
    updated = parity_from_rows([{**row, 'hf_logprobs': lp} for row, lp in zip(updated_rows, scored)])
    check_parity(updated, load(data / 'adapter_parity/verification.json'))
    checks['updated_adapter_parity_recomputed'] = True
    post = load(probe / 'post_update.json')
    assert load(probe / 'direction_check.json')['passed']
    for field in ['delta_mean_response_logprob', 'delta_mean_action_logprob']:
        positive = [row[field] for row in post['direction'] if row['advantage'] > 0]
        negative = [row[field] for row in post['direction'] if row['advantage'] < 0]
        assert len(positive) == len(negative) == 4 and sum(positive) / 4 > 0 and sum(negative) / 4 < 0
    for row in [*log['logs'], post['stability']]:
        assert row['sampled_behavior_k3'] <= plan['stability']['sampled_behavior_k3_stop']
        assert row['clip_fraction'] <= plan['stability']['clip_fraction_stop']
    assert all(math.isfinite(row['grad_norm']) and row['grad_norm'] > 0 for row in log['logs'])
    checks['direction_gradients_and_stability'] = True
    checkpoint = load(data / 'checkpoint_verification.json')
    assert checkpoint['passed'] and checkpoint['moments_equal'] and checkpoint['next_step_equal']
    assert checkpoint['restored_updates'] == 4 and checkpoint['next_test_updates'] == 5
    assert not checkpoint['online_second_round_tested']
    assert checkpoint['weights_sha256'] == log['weights_sha'] and checkpoint['optimizer_sha256'] == log['optimizer_sha']
    from safetensors.torch import load_file
    weights = load_file(str(probe / 'adapter_model.safetensors'))
    assert any('lora_B' in name and bool(tensor.count_nonzero()) for name, tensor in weights.items())
    checks['actual_checkpoint_restore_and_nonzero_weights'] = True
    assert not any((data / name).exists() for name in ['baseline_tune', 'grpo', 'gigpo'])
    result = dict(Engineering='PASS', Learning='NOT_RUN', Generalization='NOT_RUN',
                  CTHA_online_efficacy='NOT_TESTED', checks=checks, cpu_test_counts=test_counts,
                  base_parity=base, updated_adapter_parity=updated,
                  weights_sha256=log['weights_sha'], optimizer_sha256=log['optimizer_sha'],
                  checkpoint_restore_scope='CPU fixed-gradient next-step equality, not a second online RL round',
                  config_fingerprint=registration['config_fingerprint'])
    save(data / 'verification.json', result, immutable=True)
    (data / 'REPORT.txt').write_text('CTHA v3 A100 ENGINEERING ACCEPTANCE\nEngineering: PASS\nLearning: NOT_RUN\nGeneralization: NOT_RUN\nCTHA online efficacy: NOT_TESTED\n\nOnly eight real balanced responses and four engineering optimizer steps updated parameters.\nActual CPU optimizer restore used fixed engineering gradients; a second online RL round was not tested.\nAll probability, stability and direction gates are the original gates. Historical P0 records were used only for offline feasibility.\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().data), indent=2))
