import json
import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/ForkAudit/math-opsd')
RAW = ROOT/'run/cond-kl/forcing'
RAW.mkdir(exist_ok=True)
SEED = 20261124

def read(p):
    return json.loads(p.read_text())

def save(p, data):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(p)

def archived(model):
    folder=ROOT/'run/math500-long-seed1001-n2'/('nonthinking-'+model)
    data=(folder/'answers.json').read_bytes()
    assert hashlib.sha256(data).hexdigest()==read(folder/'receipt.json')['answers_sha256']
    run=json.loads(data)
    assert not run['enable_thinking'] and run['max_new_tokens']==16384 and run['seed']==1001 and run['val_n']==2
    return run['results']
