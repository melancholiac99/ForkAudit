"""Think localization after completed E1c. Does NOT modify E1/E1c requests or scores.

Conditions relative to factual L: DROP_T (remove only decision t think),
KEEP_T (remove all but decision t think), SWAP_T (replace decision t think
with a different trajectory's think). SWAP_T is exploratory, not evidence
that a replacement is factually relevant.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import exp1c
from exp2_common import ROOT, HERE, decode_rows, encode_rows, read_json, save_json, sha_file

CONDITIONS = ('DROP_T', 'KEEP_T', 'SWAP_T')
# Review 2026-11-03: the frozen E1c L text is re-scored in THIS run as the baseline. Subtracting
# scores from the separate E1c run is a cross-run comparison (new cards, new load); agreement with
# E1c L is reported as a reproducibility check instead.
ANCHOR = 'L_RE'


def edit_episode(ep, turn, condition, donor=None):
    assert condition in CONDITIONS and 1 <= turn <= len(ep['steps'])
    out = copy.deepcopy(ep)
    changes = 0
    for j, step in enumerate(out['steps'], 1):
        if 'response' not in step:
            continue
        if condition == 'DROP_T' and j == turn or condition == 'KEEP_T' and j != turn:
            step['response'], n = exp1c.THINK_BLOCK.subn('', step['response'])
            changes += n
        elif condition == 'SWAP_T' and j == turn and donor:
            step['response'], n = exp1c.THINK_BLOCK.subn(lambda _: donor, step['response'])
            changes += n
    return out, changes


def build(meta, e2, template):
    eps = {}
    donors = {}
    for z in meta:
        pid = z['point_id']
        ep = exp1c.extract_episode(e2[pid]['S'], e2[pid]['L'], template)
        t = int(e2[pid]['S']['turn'])
        eps[pid] = (ep, t)
        spans = exp1c.THINK_BLOCK.findall(ep['steps'][t-1].get('response', ''))
        if spans:
            donors[pid] = spans[0]
    ids = [x['point_id'] for x in meta]
    rows, audit = [], []
    for z in meta:
        pid, traj = z['point_id'], z['traj']
        ep, turn = eps[pid]
        # Deterministic, different-trajectory donor; do not select based on outcome.
        pool = [p for p in sorted(donors) if p != pid and next(x['traj'] for x in meta if x['point_id'] == p) != traj]
        assert pool, 'No independent rationale donor found'
        donor_id = pool[int(exp1c.sha(pid.encode())[:10], 16) % len(pool)]
        anchor = exp1c.make_request(e2[pid]['S'], ANCHOR, e2[pid]['L']['text'])
        rows.append(anchor)
        audit.append(dict(point_id=pid, traj=traj, turn=turn, condition=ANCHOR, changed_blocks=0, donor_id=None))
        for cond in CONDITIONS:
            adjusted, n = edit_episode(ep, turn, cond, donors[donor_id])
            rendered = e2[pid]['S']['text'] + '\n' + template + json.dumps(adjusted, ensure_ascii=False)
            req = exp1c.make_request(e2[pid]['S'], cond, rendered)
            rows.append(req)
            audit.append(dict(point_id=pid, traj=traj, turn=turn, condition=cond,
                              changed_blocks=n, donor_id=donor_id if cond == 'SWAP_T' else None))
        assert all(r['point_id'] == pid for r in rows[-4:])
    assert len(rows) == len(meta)*4 and len({r['request_id'] for r in rows}) == len(rows)
    return rows, audit


def prepare(args):
    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    frozen_rows, _, frozen = exp1c.prepared(Path(args.exp1c))
    meta, e2, _, template, src = exp1c.original_sources(repo)
    assert len(meta) == exp1c.N_POINTS and frozen['points'] == len(meta)
    base = {r['request_id']: r for r in frozen_rows}
    for z in meta:
        pid = z['point_id']
        assert base[pid+'__L']['text'] == e2[pid]['L']['text'], 'frozen L request mismatch'
    rows, audit = build(meta, e2, template)
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'requests.jsonl.gz'
    if p.exists():
        old, _ = decode_rows(p)
        assert old == rows, 'Existing preparation differs; use a new output dir'
    else:
        encode_rows(p, rows)
    save_json(out / 'audit.json', audit)
    save_json(out / 'manifest.json', dict(status='prepared_not_scored', n=len(rows), conditions=[ANCHOR]+list(CONDITIONS),
        requests_sha256=decode_rows(p)[1], source_sha256=src,
        exp1c_requests_sha256=frozen['requests_sha256'],
        script_sha256=sha_file(Path(__file__))))
    print(f'PREPARED {len(rows)} localization requests; changed blocks by condition:',
          {c: sum(x['changed_blocks'] for x in audit if x['condition'] == c) for c in CONDITIONS})


def prepared(out):
    out = Path(out)
    m = read_json(out/'manifest.json')
    rows, h = decode_rows(out/'requests.jsonl.gz')
    assert h == m['requests_sha256'] and len(rows) == m['n']
    assert all(x['request_sha256'] == exp1c.sha(json.dumps(
        {k:v for k,v in x.items() if k != 'request_sha256'},
        sort_keys=True, ensure_ascii=False).encode()) for x in rows)
    return rows, m


def score(args):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from exp2_common import load_32b
    from transformers import AutoTokenizer
    out = Path(args.out).resolve()
    rows, m = prepared(out)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    jobs = [exp1c.tokenize(tok, r) for r in rows]
    lens = [j['max_tokens'] for j in jobs]
    assert max(lens) <= exp1c.MAX_TOKENS, 'No context truncation is permitted'
    target = out / '32b'; target.mkdir(exist_ok=True)
    score_file = target / 'scores.jsonl'
    run_file = target / 'run.json'
    if run_file.exists():
        previous = read_json(run_file)
        assert previous['requests_sha256'] == m['requests_sha256']
        assert previous['script_sha256'] == sha_file(Path(__file__))
    else:
        assert not score_file.exists()
    seen = {}
    if score_file.exists():
        for line in score_file.read_text().splitlines():
            r = json.loads(line)
            assert r['request_id'] not in seen
            seen[r['request_id']] = r
    assert set(seen) <= {j['request_id'] for j in jobs}
    if run_file.exists() and previous.get('complete'):
        assert len(seen) == len(jobs) and sha_file(score_file) == previous['scores_sha256']
        print('ALREADY COMPLETE'); return
    with sdpa_kernel(SDPBackend.MATH):
        scorer, meta = load_32b(Path(args.model), Path(args.repo))
        if run_file.exists():
            assert previous['model_sha256'] == meta['weights']
        def probs(seq):
            lp = np.array([sum(x) for x in seq], dtype=float)
            p = np.exp(lp-lp.max()); return p/p.sum()
        # Numerical check: shared-prefix KV scoring against independently prefilling prefix+candidate.
        checks = []
        for j in [jobs[int(np.argmax(lens))], jobs[int(np.argmin(lens))]]:
            a = probs(scorer.score(j))
            b = probs([scorer.direct(j, arm) for arm in range(5)])
            tv = float(np.abs(a-b).sum()/2)
            checks.append(dict(request_id=j['request_id'], tokens=j['max_tokens'], tv=tv, passed=tv<0.005))
        save_json(target/'numerical_gate.json', checks)
        assert all(x['passed'] for x in checks), 'Numerical gate failed, no new scores written'
        run = dict(complete=False, requests_sha256=m['requests_sha256'],
                   script_sha256=sha_file(Path(__file__)), model_sha256=meta['weights'],
                   scorer_backend=meta['frozen_scorer_sha256'], precision=meta['precision'],
                   numerical_gate=checks)
        save_json(run_file, run)
        with score_file.open('a', buffering=1) as f:
            for j in jobs:
                if j['request_id'] in seen:
                    assert seen[j['request_id']]['request_sha256'] == j['request_sha256']
                    continue
                lp = np.array([sum(z) for z in scorer.score(j)], float)
                assert np.isfinite(lp).all()
                p = np.exp(lp-lp.max()); p /= p.sum()
                result = dict(request_id=j['request_id'], point_id=j['point_id'],
                     condition=j['condition'], request_sha256=j['request_sha256'],
                     candidate_prob=p.tolist(), sequence_logp=lp.tolist(), argmax=int(p.argmax()))
                f.write(json.dumps(result, allow_nan=False)+'\n')
                seen[j['request_id']] = result
    assert len(seen) == len(jobs)
    save_json(run_file, {**run, 'complete':True, 'scores_sha256':sha_file(score_file)})
    print('SCORED', len(seen))


def analyze(args):
    out = Path(args.out)
    rows, man = prepared(out)
    new_run = read_json(out/'32b/run.json')
    frozen_run = read_json(Path(args.exp1c)/'32b/run.json')
    assert new_run['complete'] and frozen_run['complete']
    assert sha_file(out/'32b/scores.jsonl') == new_run['scores_sha256']
    assert frozen_run['requests_sha256'] == man['exp1c_requests_sha256']
    scores = {}
    for p in [Path(args.exp1c)/'32b/scores.jsonl', out/'32b/scores.jsonl']:
        for line in p.read_text().splitlines():
            r=json.loads(line); scores[(r['point_id'],r['condition'])] = r
    audit = read_json(out/'audit.json')
    meta = {x['point_id']:x for x in read_json(Path(args.repo)/'hindsight_judges/results/exp1_p4/pointmeta.json')}
    arms_sha = read_json(Path(args.repo)/'exploratory_P4/e2.json')['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    arms = exp1c.read_arms(Path(args.repo),Path(args.arms),arms_sha)
    conditions = ('M','L','LN',ANCHOR) + CONDITIONS
    points = sorted(meta)
    assert all((p,c) in scores for p in points for c in conditions)
    metrics=defaultdict(list); trajs=defaultdict(list)
    for pid in points:
        z=meta[pid]; tr=z['traj']; b=z['swap_arm']
        q=np.array([arms[pid][i]['q3456'] for i in range(5)],float)
        assert np.isfinite(q).all()
        for cond in conditions:
            r=scores[(pid,cond)]; p=np.array(r['candidate_prob']); lp=np.array(r['sequence_logp'])
            for key,v in [('value',float(p@q)),('factual_logodds',float(lp[0]-lp[b])),('choose_f',float(np.argmax(p)==0))]:
                metrics[cond+'_'+key].append(v);trajs[cond+'_'+key].append(tr)
        for c in CONDITIONS:
            for name in ('value','factual_logodds','choose_f'):
                # primary: within-run anchor; the E1c-run baseline is kept only for comparison
                key=c+'_minus_LRE_'+name
                metrics[key].append(metrics[c+'_'+name][-1]-metrics[ANCHOR+'_'+name][-1]);trajs[key].append(tr)
                key=c+'_minus_L_'+name
                metrics[key].append(metrics[c+'_'+name][-1]-metrics['L_'+name][-1]);trajs[key].append(tr)
        a=np.array(scores[(pid,ANCHOR)]['candidate_prob']);o=np.array(scores[(pid,'L')]['candidate_prob'])
        metrics['LRE_vs_E1c_L_tv'].append(float(np.abs(a-o).sum()/2));trajs['LRE_vs_E1c_L_tv'].append(tr)
        metrics['LRE_vs_E1c_L_argmax_agree'].append(float(a.argmax()==o.argmax()));trajs['LRE_vs_E1c_L_argmax_agree'].append(tr)
    result={k:exp1c.mean_ci(v,trajs[k]) for k,v in metrics.items()}
    result['status']='EXPLORATORY: intervention effects, not a causal decomposition of rationalization'
    result['n_points']=len(points)
    save_json(out/'analysis.json',result)
    for k in sorted(result):
        if k.endswith(('minus_LRE_value','minus_LRE_factual_logodds','minus_L_value')) or k.startswith('LRE_vs'):
            print(k,result[k])


def selftest(_):
    ep={'steps':[{'response':'<think>A</think><action>a</action>'},
                 {'response':'<think>B</think><action>b</action>'}]}
    assert edit_episode(ep,1,'DROP_T')[0]['steps'][0]['response']=='<action>a</action>'
    assert '<think>B' in edit_episode(ep,1,'DROP_T')[0]['steps'][1]['response']
    assert '<think>B' not in edit_episode(ep,1,'KEEP_T')[0]['steps'][1]['response']
    assert '<think>X' in edit_episode(ep,1,'SWAP_T','<think>X</think>')[0]['steps'][0]['response']
    assert '<think>A' in ep['steps'][0]['response']
    print('SELFTEST PASS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','score','analyze','selftest'])
    p.add_argument('--repo',default=str(ROOT))
    p.add_argument('--exp1c',default=str(HERE/'results/exp1c_p4'))
    p.add_argument('--out',default=str(HERE/'results/think_localization'))
    p.add_argument('--arms',default=str(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet'))
    p.add_argument('--model',default=None)
    args=p.parse_args()
    if args.command=='score' and not args.model:p.error('score requires --model')
    globals()[args.command](args)


if __name__=='__main__':main()
