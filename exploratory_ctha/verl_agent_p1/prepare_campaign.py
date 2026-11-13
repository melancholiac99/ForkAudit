"""Freeze full P1 inputs, environment dependencies and source hashes before learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from run import select_games
from native_p1 import load_p0
from habits_v2 import oof_audit


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_paths():
    paths = list(HERE.glob('*.py')) + [HERE/'PLAN.json']
    paths += list((HERE.parent/'verl_agent_migration').glob('*.py'))
    paths += [HERE.parent/name for name in ['habits.py','habits_v2.py','features_v2.py','prepare.py','run.py']]
    paths += [ROOT/'env/verl_agent.sh', ROOT/'env/requirements-verl-agent.txt']
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((HERE/'PLAN.json').read_text())
    shutil.copyfile(HERE/'PLAN.json', args.out/'PLAN.json')
    games = json.loads((ROOT/'exploratory_E4/games.json').read_text())
    assert len(games['train']) == 400 and len(games['tune']) == 40
    assert not ({g['game_file'] for g in games['train']} & {g['game_file'] for g in games['tune']})
    def checked(game):
        path = ROOT/'.runtime/alfworld'/game['game_file']
        assert sha(path) == game['game_sha256'], path
        assert json.loads(path.read_text())['solvable']
        return dict(game, native_path=str(path))
    schedule = dict(train=[[checked(g) for g in select_games(games['train'],0,i,16)] for i in range(12)],
        tune=[checked(g) for g in games['tune']])
    (args.out/'tasks.json').write_text(json.dumps(schedule,indent=2)+'\n')
    p0 = load_p0()
    gate = oof_audit(p0, family='habit', min_mixed=20,min_pairs=200,permutations=12,bootstraps=300)
    assert gate['passed'], 'Original P0 gate failed; do not lower thresholds'
    (args.out/'p0_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
    for relative in hashes:
        target = args.out/'registered_source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    registration = dict(protocol=plan['version'], source_sha256=hashes,
        tasks_sha256=sha(args.out/'tasks.json'), plan_sha256=sha(args.out/'PLAN.json'),
        model={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'.runtime/models/Qwen2.5-7B-Instruct').glob('*.safetensors'))},
        p0={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'.runtime/imports/p0_base').glob('round_*/rollouts/s*_r*.json'))},
        import_verification_sha256=sha(ROOT/'.runtime/imports/import_verification.json'),
        upstream='20bd331bdbc9026a5668e11362178e10ab7400c8',
        old_protocol='exploratory_ctha/a100_p1/PLAN.json',
        publication_required_before_launch=True, free_disk_bytes=shutil.disk_usage(ROOT).free)
    registration['configuration_sha256']=hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest()
    (args.out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),fingerprint=registration['configuration_sha256'],p0_gate=gate['passed'],training_episodes_per_arm=768,tune_episodes_per_arm=560)))


if __name__ == '__main__':
    main()
