"""Shared provenance and immutable output helpers for the 2026-11-17 experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exploratory_shop_audit.runtime import COMMUNITY, SOURCE, CPU_PYTHON, GPU_PYTHON, MODEL, append, read_rows, utc
from exploratory_shop_audit.data import digest_value, digest_file, stable_seed, write_once
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / '.runtime/runs/shop_grpo_forced_20261117_01'
PARENT = ROOT / 'exploratory_shop_audit/runs/community_measurement_20261115_01'
SEED = 20261117
INTERFACE = 'shop-exact-actions-invalid-feedback-v1'


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def setup(run):
    run.mkdir(parents=True, exist_ok=True)
    write_once(run / 'registration.json', {'source_commit': read(run/'registration.json')['source_commit'] if (run/'registration.json').exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'protocol_sha256':digest_file(HERE/'PROTOCOL.md'), 'master_seed':SEED, 'interface':INTERFACE,
        'gpu_hours_limit':24,'historical_parent':str(PARENT.resolve()), 'run_id':run.name})
    if not (run/'PROTOCOL.md').exists(): (run/'PROTOCOL.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
    (run/'snapshots').mkdir(exist_ok=True)
    for name in ['product_prices_frozen.json','goals_frozen.json']:
        src=PARENT/'snapshots'/name; dst=run/'snapshots'/name
        if not dst.exists(): dst.write_bytes(src.read_bytes())
        assert digest_file(src)==digest_file(dst)


def identity(index, product, goal):
    return {'task_id':f'goal:{index}','source_task_id':index,'goal_index':index,'namespace':'ShopSimulator:13d2b636',
        'content_hash':digest_value({'instruction':goal['instruction_text'],'asin':goal['asin'],'options':goal['goal_options']}),
        'instruction_hash':digest_value(goal['instruction_text']), 'target_asin':goal['asin'],
        'tag':product['tag'], 'category':(goal.get('category') or 'unknown').split('›')[0]}
