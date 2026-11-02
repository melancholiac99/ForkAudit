"""Operational supervisor and independent final audit; scientific scripts stay frozen."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
import numpy as np

H = Path(__file__).resolve().parent
ROOT = H.parent
RUN = H/'run'
ARMS = ['Azero', 'Agap', 'Dzero', 'Dgap', 'Bfacgap']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def status(stage, **extra):
    data = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), **extra)
    save(RUN/'status.json', data); print(json.dumps(data), flush=True)


def protected():
    for path, expected in read(RUN/'protected_inputs.json').items():
        assert sha(path) == expected, f'Protected input changed: {path}'
    assert read(RUN/'environment_preflight.json')['passed']
    expected = read(H/'e4b_gate.json')['expected_pairs_sha256']
    actual = {a: sha(H/'data/pairs'/f'{a}.jsonl') for a in ARMS}
    assert actual == expected == read(H/'data/pairs/manifest.json')['sha256']
    return actual


def verify():
    pair_hashes = protected()
    games = read(ROOT/'exploratory_E4/games.json'); result = read(H/'e4b.json')
    assert set(result['splits']) == {'valid_seen', 'valid_unseen'} and result['delta_pp'] == 4
    assert result['pairs_manifest_sha256'] == sha(H/'data/pairs/manifest.json')
    sys.path.insert(0, str(ROOT/'exploratory_E4'))
    from e4_training_implementation import verify_training_log
    adapter_hashes = {}
    for arm in ARMS:
        for seed in range(3):
            tag = f'{arm}_s{seed}'; path = H/'data/adapters'/tag
            log = read(path/'train_log.json')
            expected = dict(arm=arm, seed=seed, lr=5e-6, beta=.1, r=16, alpha=32, dropout=.05,
                            eff_batch=16, epochs=2, n_examples=960, steps=120, smoke=False, pairs_sha256=pair_hashes[arm])
            for k, v in expected.items(): assert log[k] == v, (tag, k)
            assert len(log['log']) == 120 and [r['step'] for r in log['log']] == list(range(1, 121))
            assert all(math.isfinite(r['loss']) and math.isfinite(r['lr']) for r in log['log'])
            verify_training_log(ROOT, tag, log)
            assert (path/'adapter_model.safetensors').stat().st_size > 0
            adapter_hashes[tag] = sha(path/'adapter_model.safetensors')
    checked = {}; episode_count = 0; max_error = 0.
    for split in ('valid_unseen', 'valid_seen'):
        gs = games[split]; matrices = {}; expected_keys = {str(g['seed']) for g in gs}
        for arm in ['base']+ARMS:
            cols = []
            for tag in (['base'] if arm == 'base' else [f'{arm}_s{s}' for s in range(3)]):
                path = H/'data/rollout'/split/tag; sm = read(path/'summary.json')
                assert set(sm['per_game']) == expected_keys and sm['n'] == len(gs)*3
                assert sm['provenance']['games_sha256'] == sha(ROOT/'exploratory_E4/games.json')
                assert sm['provenance']['adapter_sha256'] == adapter_hashes.get(tag)
                matrix = []
                assert len(list(path.glob('s*_r*.json'))) == len(gs)*3
                for game in gs:
                    vals = []
                    for rep in range(3):
                        row = read(path/f's{game["seed"]}_r{rep}.json')
                        assert row['seed'] == game['seed'] and row['rep'] == rep and row['game_file'] == game['game_file']
                        assert row['n_turns'] == len(row['steps'])
                        vals.append(int(row['won'])); episode_count += 1
                    assert vals == sm['per_game'][str(game['seed'])]
                    matrix.append(vals)
                cols.append(np.array(matrix, float))
                assert abs(sm['success']-cols[-1].mean()) < 1e-12
            matrices[arm] = np.concatenate(cols, axis=1)
        r = result['splits'][split]; assert r['missing'] == [] and set(r['success']) == set(matrices)
        for arm, matrix in matrices.items():
            assert abs(r['success'][arm]-100*matrix.mean()) < 1e-10
        means = {k: v.mean(1) for k, v in matrices.items()}
        means['Z'] = (means['Azero']+means['Dzero'])/2
        means['G'] = (means['Agap']+means['Dgap'])/2
        indexes = np.random.default_rng(31102026).integers(0, len(gs), size=(10000, len(gs)))

        def contrast(a, b, reported):
            nonlocal max_error
            d = a-b; sample = d[indexes].mean(1)*100
            values = [float(d.mean()*100), *np.percentile(sample, [2.5, 97.5])]
            wanted = [reported['diff'], *reported['ci95']]
            error = float(max(abs(np.array(values)-wanted))); max_error = max(max_error, error)
            assert error < 1e-10 and reported['n_games'] == len(gs)

        names = ['Azero-base', 'Dzero-base', 'Agap-base', 'Dgap-base', 'Bfacgap-base', 'Z-G',
                 'Azero-Agap', 'Dzero-Dgap', 'Bfacgap-Agap', 'Bfacgap-Dgap', 'Azero-Dzero']
        assert set(r['diff']) == set(names)
        for name in names:
            a, b = name.split('-'); contrast(means[a], means[b], r['diff'][name])
        old = read(ROOT/'exploratory_E4/data/rollout'/split/'base/summary.json')
        old_m = np.array([old['per_game'][str(g['seed'])] for g in gs], float)
        bd = r['base_drift_vs_E4']; assert bd is not None
        assert bd['episode_agreement'] == float((old_m == matrices['base']).mean())
        assert abs(bd['success_E4']-old_m.mean()*100) < 1e-10
        assert abs(bd['success_E4b']-matrices['base'].mean()*100) < 1e-10
        contrast(means['base'], old_m.mean(1), bd['diff'])
        checked[split] = dict(games=len(gs), models=16, episodes=len(gs)*3*16, all_contrasts_verified=True)
    d = result['splits']['valid_unseen']['diff']; ci = d['Z-G']['ci95']
    p1 = d['Azero-base']['ci95'][0] > 0 and d['Dzero-base']['ci95'][0] > 0
    reading = 'E4b-NULL' if not p1 else 'E4b-SUPPORT' if ci[0] > -4 else 'E4b-COUNTER' if ci[1] < 0 else 'E4b-PARTIAL'
    assert result['reading'] == reading
    assert result['flags'] == dict(gap_significantly_better=ci[1] < 0, zero_significantly_better=ci[0] > 0)
    save(RUN/'final_verification.json', dict(passed=True, training_runs=15, episodes=episode_count,
         reading=reading, P1=p1, P2=ci[0] > -4, splits=checked, max_numeric_error=max_error,
         original_inputs_unchanged=True))
    paths = [p for p in (H/'data').rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.safetensors')]
    paths += [H/'e4b.json', H/'E4b_REPORT.txt', RUN/'final_verification.json']
    save(RUN/'final_artifact_manifest.json', {str(p.relative_to(H)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths})
    print('E4b independent final audit passed', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--verify-only', action='store_true'); args = ap.parse_args()
    if args.verify_only: verify(); return
    RUN.mkdir(exist_ok=True)
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('prelaunch_verification'); protected()
            busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            assert not busy, f'GPU is busy: {busy}'
            status('running', command=['bash', 'run_e4b.sh'], prereg_commit='82ebaa3801a6972a405b172ff1678997f73ff1c4')
            with (H/'data/run_e4b.log').open('a') as log:
                subprocess.run(['bash', str(H/'run_e4b.sh')], cwd=H, check=True, stdout=log, stderr=subprocess.STDOUT)
            status('final_verification'); verify()
            status('completed_verified', report=str(H/'E4b_REPORT.txt'), reading=read(H/'e4b.json')['reading'])
        except BaseException as exc:
            status('failed', error=str(exc)); traceback.print_exc(); raise


if __name__ == '__main__':
    main()
