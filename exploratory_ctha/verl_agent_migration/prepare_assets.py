"""Create a small read-only-input task view for the upstream ALFWorld loader."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / '.runtime/alfworld'
    dest = ROOT / '.runtime/verl-agent-assets'
    games = json.loads((ROOT / '.runtime/runs/ctha_a100_small_baseline_r1/train_games.json').read_text())
    assert {g['seed'] for g in games} == {1004, 1028, 1150, 1065}
    manifest = []
    for game in games:
        src = source / game['game_file']
        dst = dest / game['game_file']
        if not dst.parent.exists():
            shutil.copytree(src.parent, dst.parent)
        assert hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dst.read_bytes()).digest()
        assert json.loads(dst.read_text())['solvable']
        manifest.append(dict(game, native_path=str(dst.resolve()), source_path=str(src.resolve()),
            sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    path = ROOT / '.runtime/verl-agent-setup/fixed_tasks.json'
    value = dict(tasks=manifest, selection='Previously approved four TRAIN tasks, selected on old base outcomes',
        eval_role='Fixed training-task evaluation only; set native env.alfworld.eval_dataset=train')
    if path.exists():
        assert json.loads(path.read_text()) == value
    else:
        path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(task_count=4, manifest=str(path))))


if __name__ == '__main__':
    main()
