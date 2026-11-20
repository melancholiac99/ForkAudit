"""Freeze CPU inputs. Raw manifests are private to the run."""
import argparse
from collections import Counter
from itertools import permutations
import os
from pathlib import Path
import random
import shutil
import subprocess
import unicodedata
from transformers import AutoTokenizer
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, MODEL, SOURCE, digest_file, digest_value, read, save
from .packets import FIELDS, MATCH

HERE = Path(__file__).resolve().parent
PARENT = ROOT/'.runtime/runs/shop_grpo_forced_20261117_01'
SEED = 20261120

def freeze(run):
    os.umask(0o077)
    run.mkdir(parents=True, exist_ok=False)
    env = read(PARENT/'environment_manifest.json')
    for name, expected in env['model_files_sha256'].items():
        assert digest_file(MODEL/name) == expected, name
    for name, key in [('community_products.json','data_sha256'),('products.sqlite3','search_index_sha256')]:
        assert digest_file(SOURCE/name) == env[key], name
    (run/'snapshots').mkdir()
    for name, expected in env['snapshot_hashes'].items():
        src = PARENT/'snapshots'/name
        assert digest_file(src) == expected
        shutil.copyfile(src, run/'snapshots'/name)
    splits = read(PARENT/'splits.json')
    goals = read(run/'snapshots/goals_frozen.json')
    def record(row):
        i = row['goal_index']; goal = goals[i]
        content = ' '.join(unicodedata.normalize('NFKC', goal['instruction_text']).casefold().split())
        assert goal['asin'] == row['target_asin'] and row['tag']=='train'
        return dict(goal_index=i, asin=goal['asin'], content=digest_value(content))
    val = list(map(record,splits['V']))
    pool = list(map(record,splits['T']))
    rng = random.Random(SEED); rng.shuffle(pool)
    ids = {r['goal_index'] for r in val}; hashes = {r['content'] for r in val}
    excluded = {r['asin'] for r in val}
    clean = []
    for r in pool:
        if r['goal_index'] in ids or r['content'] in hashes or r['asin'] in excluded: continue
        ids.add(r['goal_index']); hashes.add(r['content']); clean.append(r)
    def take_unique(n):
        chosen = []; used = set()
        for r in clean:
            if r['asin'] not in used:
                chosen.append(r); used.add(r['asin'])
                if len(chosen)==n: break
        assert len(chosen)==n
        clean[:] = [r for r in clean if r['asin'] not in used]
        return chosen
    pre = take_unique(100); cal = take_unique(512)
    pools = dict(T_pre=pre,T_cal=cal,T_opd=clean,V=val)
    overlap = {}
    for a,b in permutations(pools,2):
        x = pools[a]; y = {r['asin'] for r in pools[b]}; z = {r['asin'] for r in x}
        overlap[a+' in '+b] = {'tasks':[sum(r['asin'] in y for r in x),len(x)],'asins':[len(z&y),len(z)]}
    assert all(v['tasks'][0]==0 for v in overlap.values())
    save(run/'asin_overlap_report.json',overlap)
    folds = {r['asin']:j%4 for j,r in enumerate(cal)}
    save(run/'asin_split_manifest.json',{**pools,'T_opd_eval':clean[:200],'cal_folds':folds,'V_unseen':[r['goal_index'] for r in val]})
    save(run/'splits_hash.json',{k:digest_value(v) for k,v in pools.items()})
    tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    manifest = dict(fields=FIELDS,match=MATCH,template_source_sha=digest_file(HERE/'packets.py'),tokenizer=env['model_files_sha256']['tokenizer.json'],chat_template_sha=digest_value(tok.chat_template),position='user message index 2',token_cap={'evidence':1024,'others':192},field_counts='per exposure recorded in steps and score_inputs')
    save(run/'teacher_packet_manifest.json',manifest)
    save(run/'token_parser_manifest.json',dict(source_sha=digest_file(HERE/'parser.py'),categories=['thought','search','click','option','format','unknown']))
    shutil.copyfile(HERE/'PROTOCOL.md',run/'PROTOCOL.md')
    save(run/'env_manifest.json',dict(model_files=env['model_files_sha256'],adapter=None,data=env['data_sha256'],index=env['search_index_sha256'],snapshots=env['snapshot_hashes'],environment=read(SOURCE/'community_commit.json')['sha'],gpu=subprocess.check_output(['nvidia-smi','-L'],text=True),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),prior_splits=digest_file(PARENT/'splits.json')))
    save(run/'preflight_report.json',dict(seed=SEED,counts={k:len(v) for k,v in pools.items()},unseen=len(val),protocol_sha=digest_file(run/'PROTOCOL.md'),stage='CPU registered',smoke_tasks=8))
    print({k:len(v) for k,v in pools.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);freeze(p.parse_args().run)
