"""P8 frozen blind-sample S/M/L/F preparation, exact reuse, scoring and analysis."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import p8_audit as A
from p4_e2m_prepare import build as build_memory

H, SOURCE = A.HERE, A.SOURCE
D = H / 'data/judge'
CONDITIONS = ('S', 'M', 'L', 'F')
MODELS = {'7b': ('/root/autodl-tmp/models/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
          '32b': ('/root/autodl-tmp/models/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')}


def read(p):
    return json.loads(Path(p).read_text())


def requests(p):
    return [json.loads(line) for line in gzip.decompress(Path(p).read_bytes()).splitlines()]


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def prepare():
    assert read(H / 'consistency_gate.json')['passed']
    sample = read(H / 'blind_sample.json')
    rd = A.Reader()
    templates = {k: rd.raw(f'exploratory_P4/prompts/{k}.txt').decode() for k in ('base', 'log', 'feedback', 'memory')}
    old = {}
    for file in ('e2_requests.jsonl.gz', 'e2m_requests.jsonl.gz'):
        raw = gzip.decompress(rd.raw('exploratory_P4/' + file))
        prep = rd.js('exploratory_P4/' + file.replace('_requests.jsonl.gz', '_preparation.json'))
        assert A.sha(raw) == prep['requests_sha256']
        for r in map(json.loads, raw.splitlines()):
            old[r['request_id']] = r
    jobs, trs, coverage = [], {}, []
    for s in sorted(sample['points'], key=lambda r: (r['traj_id'], r['turn'])):
        tid, t, pid = s['traj_id'], s['turn'], s['point_id']
        if tid not in trs:
            trs[tid] = rd.js(f'data/rollout/{tid}.json')
        tr = trs[tid]; step = tr['steps'][t-1]
        prompt = rd.raw(step['prompt_path']).decode()
        assert A.sha(prompt.encode()) == s['prompt_sha256']
        path = f'data/peek_alts/{tid}/t{t:02d}.json'
        if not (SOURCE / path).exists():
            coverage.append(dict(point_id=pid, missing=path))
            continue
        pk = rd.js(path)
        assert pk['checks']['factual_obs_match'] and pk['checks']['repeat_all_match'] and pk['checks']['refeed_div_total'] == 0
        peeks = sorted(pk['peeks'], key=lambda r: (r['role'] != 'factual', r['alt_index'] or 0))
        actions = [r['action'] for r in peeks]
        assert actions == s['actions'] and actions[0] == step['action'] and peeks[0]['obs'] == step['obs_after']
        seed = int(A.sha(f'P4-v1:30102026:{pid}'.encode())[:16], 16)
        order = np.random.default_rng(seed).permutation(5).tolist()
        base = templates['base'] + f'\nMarked decision: turn {t}.\n<decision_time_prompt>\n' + prompt + '\n</decision_time_prompt>\nCandidates:\n' + json.dumps([actions[j] for j in order])
        ep = dict(task=tr['task'], initial_observation=tr['initial_obs'],
                  steps=[dict(turn=a['turn'], response=a['response'], executed_action=a['action'], observation_after=a['obs_after']) for a in tr['steps']],
                  final_reward=int(tr['won']))
        texts = dict(S=base, L=base+'\n'+templates['log']+json.dumps(ep, ensure_ascii=False),
                     F=base+'\n'+templates['feedback']+json.dumps([dict(action=actions[i], response=peeks[i]['obs']) for i in order], ensure_ascii=False))
        rows = {}
        for cond, text in texts.items():
            row = dict(request_id=pid+'__'+cond, point_id=pid, traj_id=tid, turn=t, condition=cond, text=text, actions=actions)
            row['request_sha256'] = A.sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            rows[cond] = row
        memory, _ = build_memory(rows, templates['memory'])
        rows['M'] = memory[0]
        for c in CONDITIONS:
            r = rows[c]
            if r['request_id'] in old:
                assert r == old[r['request_id']], ('old request differs', r['request_id'])
            jobs.append(r)
    A.save(H / 'phase_c_coverage.json', dict(sampled=len(sample['points']), missing_peek=coverage))
    assert not coverage, 'Missing peeks: see phase_c_coverage.json; do not replace or outcome-sample'
    D.mkdir(parents=True, exist_ok=True)
    body = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in jobs)+'\n').encode()
    (D / 'requests.jsonl.gz').write_bytes(gzip.compress(body, mtime=0))
    summary = dict(points=sample['n'], contexts_per_model=len(jobs), candidate_sequences_per_model=len(jobs)*5,
                   requests_sha256=A.sha(body), blind_sample_sha256=file_sha(H / 'blind_sample.json'), reuse={})
    for name, (_, revision) in MODELS.items():
        available = {}
        originals = []
        for prefix in ('e2', 'e2m'):
            run = rd.js(f'exploratory_P4/{prefix}_{name}/run.json')
            assert run['complete'] and run['revision'] == revision
            assert run['scorer_sha256'] == file_sha(SOURCE / 'exploratory_P4/p4_e2_backend.py')
            originals.append(run)
            checks = rd.js(f'exploratory_P4/{prefix}_{name}/numerical_check.json')
            assert all(c['passed'] and c['tv'] < .005 for c in checks)
            raw = rd.raw(f'exploratory_P4/{prefix}_{name}/scores.jsonl')
            for r in map(json.loads, raw.splitlines()):
                assert r['request_id'] not in available
                available[r['request_id']] = r
        assert originals[0]['model_sha256'] == originals[1]['model_sha256']
        reused = []
        for job in jobs:
            r = available.get(job['request_id'])
            if r is None:
                continue
            assert job == old[job['request_id']] and job['request_sha256'] == r['request_sha256']
            prob = np.array(r['candidate_prob']); lp = np.array(r['sequence_logp'])
            assert prob.shape == (5,) and np.isfinite(prob).all() and abs(prob.sum()-1) < 1e-12
            expected = np.exp(lp-lp.max()); expected /= expected.sum()
            assert np.max(abs(prob-expected)) < 1e-12
            reused.append(dict(r, p8_reused=True, original_revision=revision))
        target = D / name; target.mkdir(exist_ok=True)
        path = target / 'scores.jsonl'
        assert not path.exists(), 'Already prepared; do not overwrite scores'
        path.write_text(''.join(json.dumps(r)+'\n' for r in reused))
        summary['reuse'][name] = dict(contexts=len(reused), remaining=len(jobs)-len(reused),
                                     per_condition={c: sum(r['condition'] == c for r in reused) for c in CONDITIONS})
    A.save(H / 'phase_c_preparation.json', summary)
    A.save(H / 'phase_c_inputs.json', rd.hashes)
    print(json.dumps(summary, indent=2))


def tokenize(tok, job):
    prefix = tok.apply_chat_template([dict(role='user', content=job['text'])], tokenize=False, add_generation_prompt=True) + '<action>'
    pre = tok.encode(prefix, add_special_tokens=False)
    full = [tok.encode(prefix+a+'</action>'+tok.eos_token, add_special_tokens=False) for a in job['actions']]
    start = len(pre)
    for ids in full:
        i = 0
        while i < min(len(pre), len(ids)) and pre[i] == ids[i]:
            i += 1
        start = min(start, i)
    assert len(pre)-start <= 3 and max(map(len, full)) <= 32768
    assert all(ids[:start] == full[0][:start] and ids[-1] == tok.eos_token_id for ids in full)
    return dict(**job, prefix_ids=full[0][:start], candidates=[dict(suffix_ids=ids[start:]) for ids in full])


def probability(lp):
    sums = np.array([sum(x) for x in lp]); assert np.isfinite(sums).all()
    prob = np.exp(sums-sums.max()); return sums, prob/prob.sum()


def score(name):
    import torch
    from transformers import AutoTokenizer
    path, revision = MODELS[name]
    prep = read(H / 'phase_c_preparation.json')
    raw = gzip.decompress((D / 'requests.jsonl.gz').read_bytes())
    assert A.sha(raw) == prep['requests_sha256']
    jobs = [json.loads(line) for line in raw.splitlines()]
    original = read(SOURCE / f'exploratory_P4/e2_{name}/run.json')
    assert original['revision'] == revision
    model_hash = {p.name: file_sha(p) for p in sorted(Path(path).glob('*.safetensors'))}
    assert model_hash == original['model_sha256']
    source = SOURCE / 'exploratory_P4/p4_e2_backend.py'
    assert file_sha(source) == original['scorer_sha256']
    target = D / name; target.mkdir(exist_ok=True)
    done = {}
    for line in (target / 'scores.jsonl').read_text().splitlines():
        r = json.loads(line)
        assert r['request_id'] not in done
        done[r['request_id']] = r
    for job in jobs:
        if job['request_id'] in done:
            assert done[job['request_id']]['request_sha256'] == job['request_sha256']
    assert set(done) <= {j['request_id'] for j in jobs}
    if (target / 'run.json').exists() and read(target / 'run.json')['complete']:
        assert len(done) == len(jobs)
        return
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    spec = importlib.util.spec_from_file_location('p8_existing_backend', source)
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    backend.MODEL = path
    start = time.time(); scorer = backend.Scorer(); load_seconds = time.time()-start
    checks = []; tcheck = time.time()
    fixed = [next(j for j in jobs if j['condition'] == c) for c in CONDITIONS] + [max(jobs, key=lambda j: len(j['text']))]
    for rawjob in fixed:
        j = tokenize(tok, rawjob)
        _, shared = probability(scorer.score(j))
        _, direct = probability([scorer.direct(j, k) for k in range(5)])
        tv = float(abs(shared-direct).sum()/2)
        checks.append(dict(kind='shared_vs_direct', request_id=j['request_id'], tv=tv, passed=tv < .005))
    # Fixed first existing point per condition, independent of gains/labels.
    for c in CONDITIONS:
        candidates = [j for j in jobs if j['condition'] == c and done.get(j['request_id'], {}).get('p8_reused')]
        assert candidates, f'No old context for reproduction gate: {c}'
        rawjob = candidates[0]; _, prob = probability(scorer.score(tokenize(tok, rawjob)))
        tv = float(abs(prob-np.array(done[rawjob['request_id']]['candidate_prob'])).sum()/2)
        checks.append(dict(kind='original_score_reproduction', request_id=rawjob['request_id'], tv=tv, passed=tv < .005))
    A.save(target / 'numerical_check.json', checks)
    assert all(c['passed'] for c in checks), 'Numerical/reproduction gate failed'
    check_seconds = time.time()-tcheck
    costs = {c: dict(contexts=0, candidate_sequences=0, prefix_tokens=0, suffix_tokens=0, scoring_seconds=0.) for c in CONDITIONS}
    with (target / 'scores.jsonl').open('a', buffering=1) as f:
        for rawjob in jobs:
            if rawjob['request_id'] in done:
                continue
            j = tokenize(tok, rawjob); ts = time.time()
            sums, prob = probability(scorer.score(j))
            elapsed = time.time()-ts
            r = {k: j[k] for k in ('request_id', 'request_sha256', 'point_id', 'traj_id', 'turn', 'condition')}
            r.update(sequence_logp=sums.tolist(), candidate_prob=prob.tolist(), argmax=int(prob.argmax()), p8_reused=False,
                     scoring_seconds=elapsed, prefix_tokens=len(j['prefix_ids']), suffix_tokens=sum(len(x['suffix_ids']) for x in j['candidates']))
            f.write(json.dumps(r)+'\n'); done[r['request_id']] = r
            if len(done) % 20 == 0:
                print(json.dumps(dict(completed=len(done), total=len(jobs), elapsed=time.time()-start)), flush=True)
    for r in done.values():
        if r.get('p8_reused'):
            continue
        c = costs[r['condition']]
        c['contexts'] += 1; c['candidate_sequences'] += 5
        for k in ('prefix_tokens', 'suffix_tokens', 'scoring_seconds'):
            c[k] += r[k]
    A.save(target / 'run.json', dict(complete=len(done) == len(jobs), contexts=len(done), revision=revision,
           model_sha256=model_hash, backend_sha256=file_sha(source), requests_sha256=prep['requests_sha256'],
           cost_by_condition=costs, load_seconds=load_seconds, numerical_gate_seconds=check_seconds,
           process_seconds=time.time()-start, gpu=torch.cuda.get_device_name(0), commit=A.commit()))


def analyze():
    rows = read(H / 'data/points.json'); array = np.load(H / 'data/arrays.npz')
    q = array['q']; sample = read(H / 'blind_sample.json')
    idx = {r['point_id']: i for i, r in enumerate(rows)}
    wanted = {r['point_id']: r for r in sample['points']}
    mask = np.array([r['point_id'] in wanted for r in rows])
    weights = np.array([wanted[r['point_id']]['inverse_probability'] if r['point_id'] in wanted else 0 for r in rows])
    boot = A.Bootstrap([r['traj_id'] for r in rows])
    probs = {}; expected = {r['request_id']: r for r in requests(D / 'requests.jsonl.gz')}
    for name in MODELS:
        run = read(D / name / 'run.json'); assert run['complete']
        assert all(c['passed'] and c['tv'] < .005 for c in read(D / name / 'numerical_check.json'))
        scored = [json.loads(l) for l in (D / name / 'scores.jsonl').read_text().splitlines()]
        assert len(scored) == len(expected) and {r['request_id'] for r in scored} == set(expected)
        for c in CONDITIONS:
            p = np.full_like(q, np.nan)
            for r in scored:
                if r['condition'] != c:
                    continue
                assert r['request_sha256'] == expected[r['request_id']]['request_sha256']
                i = idx[r['point_id']]; prob = np.array(r['candidate_prob'])
                assert prob.shape == (5,) and np.isfinite(prob).all() and (prob >= 0).all() and abs(prob.sum()-1) < 1e-12
                p[i] = prob
            assert np.isfinite(p[mask]).all()
            probs[name+'/'+c] = p
    probs.update(policy=np.array([r['policy'] for r in rows]), R1_window=np.array([r['R1_window'] for r in rows]),
                 R1_full=np.array([r['R1_full'] for r in rows]), R0=np.full_like(q, .2))
    gains = {k: A.gu(p, q) for k, p in probs.items()}
    live = np.array([r['live'] for r in rows]); fork = np.array([r['fork'] for r in rows])
    groups = dict(H_prompt=mask, H_full=mask & np.array([r['H_full'] for r in rows]), H_Live=mask & live,
                  H_nonFork_Live=mask & live & ~fork, H_Fork=mask & fork)
    table = {name: {group: boot.stat(g, m, weights)[0] for group, m in groups.items()} for name, g in gains.items()}
    pairs = [('R1_window', 'policy'), ('R1_window', '7b/S'), ('R1_window', '32b/S'),
             ('R1_full', '7b/M'), ('R1_full', '32b/M'), ('32b/F', '32b/M'), ('32b/F', 'R1_full'),
             ('32b/S', '7b/S'), ('32b/M', '7b/M'), ('32b/L', '32b/M')]
    contrasts = {a+' - '+b: {group: boot.stat(gains[a]-gains[b], m, weights)[0] for group, m in groups.items()} for a, b in pairs}
    s5 = all(contrasts[k]['H_prompt']['ci95'] is not None and contrasts[k]['H_prompt']['ci95'][0] > 0
             for k in ('32b/F - 32b/M', '32b/F - R1_full'))
    out = dict(commit=A.commit(), estimator='inverse-inclusion weighted ratio point mean; shared trajectory cluster bootstrap',
               table=table, contrasts=contrasts, S5=s5, units='fraction', sample_sha256=file_sha(H / 'blind_sample.json'),
               costs={name: read(D / name / 'run.json') for name in MODELS})
    A.save(H / 'p8_phase_c.json', out)
    cpu = read(H / 'p8_cpu.json')
    full = dict(cpu=cpu, phase_c=out, reading=cpu['reading'], P1=cpu['P1'], P2=cpu['P2'], S4=cpu['S4'], S5=s5)
    A.save(H / 'p8_complete.json', full)
    lines = [(H / 'P8_CPU_REPORT.txt').read_text().replace('S5 and same-population model comparisons: pending phase C', 'Phase C complete; see below.'),
             '\nPhase C (blind history-only sample; IPW; same points):', 'S5: '+str(s5)]
    lines += [name+': '+A.fmt(t['H_prompt']) for name, t in table.items()]
    lines += ['\nPaired contrasts:'] + [name+': '+A.fmt(t['H_prompt']) for name, t in contrasts.items()]
    lines += ['\nThis is a post-hoc audit. F/L use extra information. A2 is descriptive; no hidden-world surprise or global irrationality is identified.']
    (H / 'P8_REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(reading=cpu['reading'], S5=s5)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['prepare', 'score', 'analyze'])
    ap.add_argument('--name', choices=list(MODELS))
    args = ap.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'score':
        assert args.name; score(args.name)
    else: analyze()
