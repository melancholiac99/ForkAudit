"""Original cold-start calculation on verified imported data, using local paths."""
import json
import os
import subprocess
import sys
from pathlib import Path

from a100_common import DATA, FAST, HERE, MODEL, PARENT, ROOT, load, save, sha


def main():
    imported = ROOT / '.runtime/imports/import_verification.json'
    assert load(imported)['verified'], 'Original archive must be verified first'
    sys.path.insert(0, str(FAST))
    import util
    util.MODEL = MODEL
    util.DATA = DATA
    from check_cpu import checks
    from habits import load_episodes
    from habits_v2 import make_batch, oof_audit
    p0 = Path(os.environ['CTHA_P0_ROOT'])
    legacy = Path(os.environ['CTHA_LEGACY_ROUND'])
    pool = []
    for i, directory in enumerate(sorted(p0.glob('round_*'))):
        pool += load_episodes(directory / 'rollouts', -1000 + i)
    train_games = {g['game_file'] for g in load(ROOT / 'exploratory_E4/games.json')['train']}
    assert len(pool) == 768
    assert all(e['game_file'] in train_games and e['provenance']['adapter_sha256'] is None
               and e['provenance']['split'] == 'train' for e in pool)
    options = dict(family='habit', min_mixed=20, min_pairs=200,
                   permutations=12, bootstraps=300)
    gate = oof_audit(pool, **options)
    current = load_episodes(legacy, 0)
    rows, report = make_batch(current, pool, mode='ctha', base='gigpo',
                              feature_window=8, train_horizon=0, aux_horizon=8,
                              sign_fraction=.75, **options)
    result = dict(checks=checks(), cold_start_gate=gate,
                  offline_nonzero_aux_samples=report['nonzero_aux_samples'],
                  offline_combined_pool_gate=report['gate'], p0_episodes=len(pool),
                  legacy_round_episodes=len(current), import_verification_sha256=sha(imported),
                  original_checker_sha256=sha(FAST / 'check_cpu.py'),
                  a100_wrapper_sha256=sha(Path(__file__)),
                  source_sha=util.source_hashes(),
                  interpretation='Original P0 data reused only for the original offline feasibility check. Not A100 on-policy training data or independent efficacy evidence.')
    save(DATA / 'cpu_validation.json', result, immutable=True)
    print(json.dumps(result, indent=2), flush=True)
    if not (gate['passed'] and report['nonzero_aux_samples'] > 0 and all(result['checks'].values())):
        raise RuntimeError('Original offline feasibility gate failed; no threshold relaxation')


if __name__ == '__main__':
    main()
