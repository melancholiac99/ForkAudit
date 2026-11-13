"""Fixed-training-task learning diagnostic; no Tune/generalization inference."""
import math
from protocol_common import ENGINEERING, HERE, PILOT, fingerprint, load, save, sha, sources


def measurements(folder, adapter_sha):
    games = load(PILOT / 'train_games.json')
    summary = load(folder / 'summary.json')
    prov = summary['provenance']
    assert prov['split'] == 'train' and prov['seed_namespace'] == 'baseline_rl_fixed_train_eval_v1'
    assert prov['adapter_sha256'] == adapter_sha
    assert prov['a100_config_fingerprint'] == fingerprint(load(PILOT / 'frozen_config.json'))
    values = []
    for game in games:
        episodes = [load(folder / f's{game["seed"]}_r{r}.json') for r in range(4)]
        assert all(e['game_file'] == game['game_file'] and e['provenance'] == prov for e in episodes)
        values.append(sum(e['won'] for e in episodes) / 4)
    assert summary['n_episodes'] == 64 and math.isclose(sum(values) / 16, summary['success_rate'])
    return values


def verify():
    import numpy as np
    assert load(ENGINEERING / 'verification.json')['Engineering'] == 'PASS'
    assert load(PILOT / 'registration.json')['source_sha256'] == sources()
    base = np.asarray(measurements(PILOT / 'base_train_eval', None))
    results = {}
    for arm in ['grpo', 'gigpo']:
        prior_sha = None
        prior_updates = 0
        rounds = []
        for i in range(4):
            rnd = PILOT / arm / f'round_{i:03d}'
            summary = load(rnd / 'rollouts/summary.json')
            assert summary['provenance']['adapter_sha256'] == prior_sha
            assert summary['provenance']['seed_namespace'] == 'train'
            assert summary['provenance']['iteration'] == i
            report = load(rnd / 'prepared/advantage_report.json')
            assert report['base'] == arm and report['nonzero_aux_samples'] == 0
            assert report['samples_sha256'] == sha(rnd / 'prepared/samples.jsonl')
            log = load(rnd / 'adapter/train_log.json')
            assert log['complete'] and log['optimizer_resumed'] == (i > 0)
            assert log['n_trained_rows'] == log['n_source_steps'] == report['total_rows']
            assert log['total_updates'] - prior_updates == log['updates_this_round'] == math.ceil(log['n_trained_rows'] / 64)
            assert log['weights_sha'] == sha(rnd / 'adapter/adapter_model.safetensors')
            assert log['optimizer_sha'] == sha(rnd / 'adapter/optimizer.pt')
            assert load(rnd / 'adapter/live_checkpoint_verification.json')['passed']
            for row in [*log['logs'], load(rnd / 'adapter/post_update.json')['stability']]:
                assert math.isfinite(row['sampled_behavior_k3']) and row['sampled_behavior_k3'] <= .02
                assert math.isfinite(row['clip_fraction']) and row['clip_fraction'] <= .30
            assert log['parity']['passed']
            rounds.append(dict(round=i, success=summary['success_rate'], rows=log['n_trained_rows'], updates=log['updates_this_round'], weights_sha256=log['weights_sha'], optimizer_sha256=log['optimizer_sha']))
            prior_updates, prior_sha = log['total_updates'], log['weights_sha']
        final = np.asarray(measurements(PILOT / arm / 'final_train_eval', prior_sha))
        delta = final - base
        rng = np.random.default_rng(20261113)
        draws = delta[rng.integers(0, 16, size=(10000, 16))].mean(axis=1)
        ci = np.quantile(draws, [.025, .975]).tolist()
        results[arm] = dict(rounds=rounds, baseline_success=float(base.mean()), final_success=float(final.mean()),
            paired_delta=float(delta.mean()), ci95=ci, per_game_delta=delta.tolist(),
            Learning='DEMONSTRATED' if delta.mean() > 0 and ci[0] > 0 else 'NOT_DEMONSTRATED',
            scope='Fixed sixteen TRAIN games, independent eval seeds, one training seed; no generalization claim.')
    output = dict(Execution='COMPLETE', Engineering='PASS (new full-response protocol)', results=results,
        Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED', CTHA='NOT_STARTED',
        limitations='No multi-seed replication or algorithm superiority claim. Four rounds only; no outcome-driven extension.')
    save(PILOT / 'verification.json', output, immutable=True)
    return output


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))

