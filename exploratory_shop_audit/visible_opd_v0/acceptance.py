"""Independent CPU acceptance of the completed A.5 run from the saved evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from exploratory_shop_audit.data import stable_seed
from .packets import MATCH, POLICIES

ALL_REQUIREMENTS = {'gold_purchase', 'valid_alternative_purchase'}


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return row['goal'], row['rep'], row['step']


def audit(run):
    dest = run/'formal_report'
    load = lambda name: json.loads((run/name).read_text())
    split = load('asin_split_manifest.json')
    pre = split['T_pre']
    goals = [t['goal_index'] for t in pre]
    require(len(set(goals)) == len({t['asin'] for t in pre}) == 100, 'task/ASIN cluster count')
    for name in ('T_cal','T_opd','V'):
        for col in ('goal_index', 'content', 'asin'):
            require({t[col] for t in pre}.isdisjoint(t[col] for t in split[name]), 'split overlap')

    decision = load('formal_report/packet_selection.json')
    repair = {key(r):r['classes'] for r in rows(run/'parser_repair.jsonl')}
    require(digest(run/'parser_repair.jsonl') == load('parser_repair_manifest.json')['sidecar_sha'], 'repair hash')
    completed = {}
    student = {}
    tapes = {}
    packet_status = Counter()
    manifest = {}
    n_steps = 0
    seen_steps = set()
    for stage in ('smoke', 'formal'):
        for folder in sorted(run.glob(stage+'_batch_*')):
            for path in folder.glob('*.jsonl'):
                manifest[str(path.relative_to(run))] = digest(path)
            for r in rows(folder/'episodes.jsonl'):
                k = (r['mode'], r['goal'], r['rep'])
                require(k not in completed and 1 <= r['length'] <= 35, 'invalid episode')
                require(r['goal'] in (goals[:8] if stage=='smoke' else goals[8:]), 'wrong collection partition')
                completed[k] = r
            for r in rows(folder/'steps.jsonl'):
                k = (r['mode'], *key(r))
                require(k not in seen_steps and k[:-1] in completed, 'invalid step')
                require(0 <= r['step'] < completed[k[:-1]]['length'], 'episode length mismatch')
                require(r['seed'] == stable_seed(20261120,r['goal'],r['rep'],r['step']), 'unaligned seed')
                seen_steps.add(k)
                n_steps += 1
                packet_status[(r['mode'],r['packet_status'] or 'available')] += 1
                if r['mode']=='none':
                    student[key(r)] = r['prompt_ids'],r['response_ids']
            for r in rows(folder/'score_inputs.jsonl'):
                k = key(r)
                require(k in student and k not in tapes, 'unpaired/duplicate scorer input')
                require(student[k] == (r['views']['none']['ids'],r['response']), 'student prefix/response mismatch')
                available = {m for m,v in r['views'].items() if not v['reason']}
                tapes[k] = r['response'],repair.get(k,r['classes']),available
    expected = {(m,g,r) for m in POLICIES for g in goals for r in range(3)}
    require(set(completed)==expected and set(tapes)==set(student), 'missing cells')
    require(n_steps==sum(e['length'] for e in completed.values()), 'incomplete steps')

    hits = defaultdict(list)
    seen = set()
    classes = Counter()
    max_center = 0.
    max_kl_error = 0.
    for pattern in ('scores_*.jsonl','formal_scores_*.jsonl'):
        for path in sorted(run.glob(pattern)):
            manifest[path.name]=digest(path)
            for row in rows(path):
                k = key(row)
                require(k not in seen and k in tapes, 'unpaired/duplicate score')
                seen.add(k)
                actual,labels,available = tapes[k]
                require(row['actual']==actual and set(row['scores'])==available and row['empty_max']<=1e-5, 'scorer coverage or empty control')
                classes.update(labels)
                mask=np.asarray(row['candidate_mask'],bool)
                ids=np.asarray(row['ids'])
                included=(ids==np.asarray(actual)[:,None]) & mask
                require(np.all(included.sum(1)==1) and np.all(mask.sum(1)>=8), 'sampled candidate missing')
                require(all(len(set(tokens[valid]))==valid.sum() for tokens,valid in zip(ids,mask)), 'duplicate candidates')
                qs=np.asarray(row['qs'])
                require(np.allclose(qs.sum(1),1,atol=1e-5) and np.all(qs[~mask]==0), 'invalid conditional support')
                for m,ref in MATCH.items():
                    if not {m,ref}<=available:continue
                    real,base = row['scores'][m],row['scores'][ref]
                    d=np.asarray(real['d']); dp=np.asarray(base['d'])
                    max_center=max(max_center,float(np.abs((d*qs).sum(1)).max()))
                    def normalized(delta):
                        z=qs*np.exp(delta-delta.max(1,keepdims=True))
                        return z/z.sum(1,keepdims=True)
                    q,p=normalized(d),normalized(dp)
                    content_kl=(q*(np.log(q.clip(1e-250))-np.log(p.clip(1e-250)))).sum(1)
                    max_kl_error=max(max_kl_error,float(np.abs(content_kl-real['content_kl']).max()))
                    dy=(d-dp)[included]
                    require(np.allclose(dy,real['content_d_y'],atol=1e-5), 'sampled content mismatch')
                    keep=np.isin(labels,['click','option']) & (np.minimum.reduce([row['scores']['none']['mass'],real['mass'],base['mass']])>=.25)
                    crosses=(np.abs(dy)>.02) & (np.asarray(real['content_kl'])>1e-5)
                    hits[m].extend((k[0],bool(v)) for v in crosses[keep])
    require(seen==set(tapes) and max_center<1e-5 and max_kl_error<1e-5,'distribution reconstruction')
    for m in MATCH:
        saved=decision['decisions'][m]
        required=hits[m]
        require(len(required)==saved['positions'] and sum(v for _,v in required)==saved['hits'] and len({g for g,v in required if v})==saved['shifted_tasks'],'content gate mismatch')
        gain=np.array([np.mean([int(completed[m,g,r]['reason'] in ALL_REQUIREMENTS)-int(completed['none',g,r]['reason'] in ALL_REQUIREMENTS) for r in [0,1]]) for g in goals])
        require(np.isclose(gain.mean(),saved['lift']['estimate']), 'paired estimate mismatch')
        boot=np.random.default_rng(20261120).choice(gain,size=(10000,len(goals)),replace=True).mean(1)
        require(np.allclose(np.quantile(boot,[.025,.975]),saved['lift']['ci95']), 'bootstrap CI mismatch')
    ledger=list(rows(run/'preflight_cost_ledger.jsonl'))
    used=sum(x['gpu_hours'] for x in ledger if x['stage'].startswith('formal_'))
    require(0<used<=load('formal_authorization.json')['additional_gpu_hours'], 'budget exceeded')
    (dest/'input_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(passed=True,episodes=len(completed),steps=n_steps,student_steps=len(seen),response_classes=classes,
                max_center_error=max_center,max_content_kl_error=max_kl_error,formal_gpu_hours=used,
                data_hash_manifest_sha256=digest(dest/'input_hashes.json'))
    (dest/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    print(audit(parser.parse_args().run))
