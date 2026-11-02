"""P4 hindsight identity-label intervention: prepare (CPU), score (GPU), analyze (CPU).

Source of truth: P4 E2 S/F contexts, audit permutation, deep-discovery arm q12/q3456/qgold.
Never compare newly scored contexts to original P4 scores across GPUs.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDS = ('S0', 'SE', 'SX', 'SPE', 'SPX', 'F0', 'FE', 'FX', 'FPE', 'FPX')
NPOINTS = 386
SEED = 20261102
NBOOT = 4000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_parquet(repo, path):
    sys.path.insert(0, str(repo / 'exploratory_P4' / 'post_e2'))
    import minipq
    return minipq.read(str(path))


def source_info(repo, arms_path):
    p4 = repo / 'exploratory_P4'
    prep = load_json(p4 / 'e2_preparation.json')
    raw = gzip.decompress((p4 / 'e2_requests.jsonl.gz').read_bytes())
    assert sha(raw) == prep['requests_sha256'], 'E2 input request SHA mismatch'
    e2 = load_json(p4 / 'e2.json')
    expected = e2['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    assert digest(arms_path) == expected, 'deep-discovery arms SHA mismatch against P4 E2'
    files = {
        'exploratory_P4/e2_requests.jsonl.gz': p4 / 'e2_requests.jsonl.gz',
        'exploratory_P4/e2_preparation.json': p4 / 'e2_preparation.json',
        'exploratory_P4/e2_point_audit.json': p4 / 'e2_point_audit.json',
        'exploratory_P4/strata.parquet': p4 / 'strata.parquet',
        'exploratory_P4/e2.json': p4 / 'e2.json',
        'exploratory_deep_discovery/exploratory_arms.parquet': arms_path,
    }
    for key, value in e2['inputs'].items():
        if key in files:
            assert digest(files[key]) == value, 'original source SHA mismatch: ' + key
    hashes = {k: digest(v) for k, v in files.items()}
    return raw, hashes


def build_dataset(repo, arms_path):
    raw, input_hashes = source_info(repo, arms_path)
    p4 = repo / 'exploratory_P4'
    s = read_parquet(repo, p4 / 'strata.parquet')
    target = {}
    for i in range(len(s['traj_id'])):
        if bool(s['live'][i]) and bool(s['het'][i]) and s['stratum'][i] in ('I', 'U'):
            pid = f"{s['traj_id'][i]}_t{int(s['turn'][i]):02d}"
            assert pid not in target
            target[pid] = dict(traj=str(s['traj_id'][i]), layer=str(s['stratum'][i]))
    assert len(target) == NPOINTS, f'expected {NPOINTS} I/U Forks; got {len(target)}'

    a = read_parquet(repo, arms_path)
    q12 = {}
    for i in range(len(a['traj_id'])):
        pid = f"{a['traj_id'][i]}_t{int(a['turn'][i]):02d}"
        if pid in target:
            q12.setdefault(pid, {})[int(a['j'][i])] = float(a['q_12'][i])
    for pid in target:
        assert set(q12.get(pid, {})) == set(range(5)), f'missing q12 arms: {pid}'
        assert all(math.isfinite(v) for v in q12[pid].values()), f'nonfinite q12 {pid}'

    audit_records = load_json(p4 / 'e2_point_audit.json')
    audits = {r['point_id']: r for r in audit_records}
    assert len(audits) == len(audit_records), 'duplicate audit point'
    context = {}
    for line in raw.decode().splitlines():
        r = json.loads(line)
        if r['point_id'] in target and r['condition'] in ('S', 'F'):
            assert r['condition'] not in context.setdefault(r['point_id'], {})
            context[r['point_id']][r['condition']] = r

    records, pointmeta = [], []
    for pid in sorted(target):
        assert set(context.get(pid, {})) == {'S', 'F'}, f'missing S/F {pid}'
        S, F = context[pid]['S'], context[pid]['F']
        assert S['actions'] == F['actions'] and F['text'].startswith(S['text'])
        assert S['point_id'] == F['point_id'] == pid
        order = list(audits[pid]['display_order'])
        assert sorted(order) == list(range(5))
        suffix = S['text'].split('Candidates:\n')[-1].strip()
        displayed = json.loads(suffix)
        assert displayed == [S['actions'][j] for j in order], 'candidate display order mismatch'
        # q12 is used ONLY to choose the contrast arm. q3456/Gold are held out for evaluation.
        swap = min(range(1, 5), key=lambda j: (-q12[pid][j], j))
        assert swap != 0
        real_slot, swap_slot = order.index(0) + 1, order.index(swap) + 1
        pointmeta.append(dict(point_id=pid, traj=target[pid]['traj'], layer=target[pid]['layer'],
                              factual_arm=0, swap_arm=swap, real_display_number=real_slot,
                              swap_display_number=swap_slot, q12_gap=q12[pid][swap] - q12[pid][0]))
        for orig in ('S', 'F'):
            row = context[pid][orig]
            tail = '' if orig == 'S' else F['text'][len(S['text']):]
            for c, slot, placebo in [('0', None, False), ('E', real_slot, False),
                                      ('X', swap_slot, False), ('PE', real_slot, True),
                                      ('PX', swap_slot, True)]:
                if c == '0':
                    content = row['text']
                else:
                    if placebo:
                        label = (f"Placebo metadata: candidate number {slot} in the displayed Candidates list "
                                 "carries an artificial selected marker introduced by this evaluation, "
                                 "not a record of the original agent's decision. Do not treat this marker as evidence of quality.")
                    else:
                        label = (f"Execution metadata at the marked decision: candidate number {slot} in the "
                                 "displayed Candidates list was selected by the original agent. "
                                 "This historical choice is not proof of action quality.")
                    content = S['text'] + '\n' + label + tail
                cond = orig + c
                r = dict(request_id=pid+'__'+cond, point_id=pid, traj_id=S['traj_id'],
                         turn=int(S['turn']), condition=cond, text=content, actions=S['actions'])
                r['request_sha256'] = sha(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
                records.append(r)
    assert len(records) == NPOINTS * len(CONDS)
    assert sorted(set(r['condition'] for r in records)) == sorted(CONDS)
    return records, pointmeta, input_hashes


def prepare(args):
    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    arms = Path(args.arms).resolve()
    records, meta, hashes = build_dataset(repo, arms)
    out.mkdir(parents=True, exist_ok=True)
    body = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n').encode()
    (out / 'requests.jsonl.gz').write_bytes(gzip.compress(body, mtime=0))
    write_json(out / 'pointmeta.json', meta)
    write_json(out / 'manifest.json', dict(status='PRE-SCORING', source='P4-E2',
                points=len(meta), contexts=len(records), conditions=list(CONDS),
                input_sha256=hashes, requests_sha256=sha(body),
                pointmeta_sha256=digest(out / 'pointmeta.json'),
                prepare_script_sha256=digest(__file__), selection='argmax_q12_alternative; tie lowest arm',
                value_eval='q3456 independent, Gold independent; q12 never as held-out outcome',
                git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip()))
    print('Prepared', len(meta), 'points x',len(CONDS),'conditions =',len(records),'contexts at',out)


def rows_and_meta(out):
    man = load_json(out / 'manifest.json')
    raw = gzip.decompress((out / 'requests.jsonl.gz').read_bytes())
    assert sha(raw) == man['requests_sha256'], 'prepared request SHA mismatch'
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert len(rows) == man['contexts'] == NPOINTS * len(CONDS)
    assert len(set(x['request_id'] for x in rows)) == len(rows)
    for r in rows:
        assert r['request_sha256'] == sha(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,ensure_ascii=False).encode())
    assert digest(out / 'pointmeta.json') == man['pointmeta_sha256']
    return rows, load_json(out / 'pointmeta.json'), man


def tokenize(tok, job):
    prefix = tok.apply_chat_template([dict(role='user', content=job['text'])],
                                     tokenize=False, add_generation_prompt=True) + '<action>'
    pre = tok.encode(prefix, add_special_tokens=False)
    full = [tok.encode(prefix + a + '</action>' + tok.eos_token, add_special_tokens=False) for a in job['actions']]
    start = len(pre)
    for ids in full:
        x = 0
        while x < min(len(pre),len(ids)) and pre[x]==ids[x]: x += 1
        start = min(start,x)
    assert len(pre)-start <= 3 and max(map(len,full)) <= 32768
    assert all(ids[:start]==full[0][:start] and ids[-1]==tok.eos_token_id for ids in full)
    return dict(**job, prefix_ids=full[0][:start],
                candidates=[dict(suffix_ids=ids[start:]) for ids in full])


def score(args):
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no GPU scores were written')
    repo=Path(args.repo).resolve(); p4=repo/'exploratory_P4'; out=Path(args.out).resolve()
    rawjobs, _, prep=rows_and_meta(out)
    original=load_json(p4/f'e2_{args.name}'/'run.json')
    model=Path(args.model).resolve()
    assert original['complete'] and args.revision==original['revision']
    assert model.is_dir() and list(model.glob('*.safetensors'))
    actual={f.name: digest(f) for f in sorted(model.glob('*.safetensors'))}
    assert actual==original['model_sha256'], 'model weights differ from frozen P4 E2'
    src=p4/'p4_e2_backend.py'
    assert digest(src)==original['scorer_sha256'], 'scorer source differs from original'
    if (p4/'e2_environment.json').exists():
        hashes=load_json(p4/'e2_environment.json')['model_metadata_hashes'][args.name.upper()]
        for name, value in hashes.items():
            assert digest(model/name)==value, 'model metadata changed: '+name

    target=out/args.name; target.mkdir(parents=True,exist_ok=True)
    runpath=target/'run.json'; scorespath=target/'scores.jsonl'
    env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
             revision=args.revision, model_sha256=actual, backend_sha256=digest(src),
             scorer_script_sha256=digest(__file__), requests_sha256=prep['requests_sha256'])
    if runpath.exists():
        old=load_json(runpath)
        assert all(old.get(k)==v for k,v in env.items()), 'resume env/checkpoint mismatch; use another output dir'
    write_json(runpath,dict(**env,complete=False,contexts=0))

    spec=importlib.util.spec_from_file_location('original_p4_e2_backend',src)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.MODEL=str(model)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    scorer=module.Scorer()
    def prob(lp):
        x=np.array([sum(v) for v in lp],float)
        x=np.exp(x-x.max()); return x/x.sum()
    checks=[]
    for job in [next(r for r in rawjobs if r['condition']==c) for c in CONDS]+[max(rawjobs,key=lambda r:len(r['text']))]:
        j=tokenize(tok,job)
        p1=prob(scorer.score(j)); p2=prob([scorer.direct(j,k) for k in range(5)])
        tv=float(np.abs(p1-p2).sum()/2)
        checks.append(dict(request_id=job['request_id'],tv=tv,passed=tv<0.005))
    write_json(target/'numerical_check.json',checks)
    assert all(x['passed'] for x in checks), 'shared prefix vs direct score numerical gate FAILED'

    done={}
    if scorespath.exists():
        for line in scorespath.read_text().splitlines():
            row=json.loads(line)
            assert row['request_id'] not in done
            done[row['request_id']]=row
    by_id={j['request_id']:j for j in rawjobs}
    assert set(done).issubset(by_id)
    assert all(done[k]['request_sha256']==by_id[k]['request_sha256'] for k in done)
    t0=time.time()
    with scorespath.open('a',buffering=1) as f:
        for job in rawjobs:
            if job['request_id'] in done:continue
            j=tokenize(tok,job)
            lp=scorer.score(j)
            logits=np.array([sum(v) for v in lp]); assert np.isfinite(logits).all()
            pp=np.exp(logits-logits.max());pp/=pp.sum()
            row=dict(request_id=job['request_id'],request_sha256=job['request_sha256'],
                     point_id=job['point_id'],condition=job['condition'],
                     candidate_prob=pp.tolist(),sequence_logp=logits.tolist(),
                     argmax=int(pp.argmax()))
            f.write(json.dumps(row)+'\n')
            done[row['request_id']]=row
            if len(done)%50==0: print('scored',len(done),'/',len(rawjobs),flush=True)
    write_json(runpath,dict(**env,complete=len(done)==len(rawjobs),contexts=len(done),
                            total_contexts=len(rawjobs),last_segment_seconds=time.time()-t0,
                            peak_gb=float(torch.cuda.max_memory_allocated()/1e9)))
    print('DONE',len(done),'/',len(rawjobs))


def mean_ci(values, trajs, nboot=NBOOT, seed=SEED):
    v=np.asarray(values,dtype=float); tt=np.array(trajs)
    ok=np.isfinite(v);v=v[ok];tt=tt[ok]
    if not len(v):return dict(n=0,estimate=None,ci95=None)
    tids=sorted(set(tt)); idx={t:i for i,t in enumerate(tids)}
    g=np.array([idx[t] for t in tt])
    sv=np.bincount(g,weights=v,minlength=len(tids)); sn=np.bincount(g,minlength=len(tids))
    rng=np.random.default_rng(seed)
    W=rng.multinomial(len(tids),np.full(len(tids),1/len(tids)),size=nboot)
    with np.errstate(invalid='ignore',divide='ignore'):
        b=(W@sv)/(W@sn)
    b=b[np.isfinite(b)]
    assert len(b)>int(nboot*.95), 'bootstrap too many empty draws'
    return dict(n=len(v),n_trajectories=len(tids),estimate=float(v.mean()),
                ci95=[float(x) for x in np.percentile(b,[2.5,97.5])])


def analyze(args):
    repo=Path(args.repo).resolve();out=Path(args.out).resolve()
    jobs, meta, man=rows_and_meta(out)
    assert len(jobs)==NPOINTS*len(CONDS)
    runs=load_json(out/args.name/'run.json')
    assert runs['complete'] and runs['requests_sha256']==man['requests_sha256']
    sc={}
    for line in (out/args.name/'scores.jsonl').read_text().splitlines():
        r=json.loads(line)
        assert r['request_id'] not in sc
        sc[r['request_id']]=r
    assert len(sc)==len(jobs)
    for j in jobs:
        r=sc[j['request_id']]
        assert r['request_sha256']==j['request_sha256']
        p=np.array(r['candidate_prob'],float)
        assert len(p)==5 and np.isfinite(p).all() and np.all(p>=0) and abs(p.sum()-1)<1e-6
    arms=Path(args.arms).resolve()
    assert digest(arms)==man['input_sha256']['exploratory_deep_discovery/exploratory_arms.parquet']
    a=read_parquet(repo,arms)
    vals={}
    for i in range(len(a['traj_id'])):
        pid=f"{a['traj_id'][i]}_t{int(a['turn'][i]):02d}"
        if pid not in vals:vals[pid]={}
        vals[pid][int(a['j'][i])]=dict(q12=float(a['q_12'][i]),
            q3456=float(a['q_3456'][i]) if a['q_3456'][i] is not None else float('nan'),
            qgold=float(a['q_gold'][i]) if a['q_gold'][i] is not None else float('nan'))
    result=dict(status='EXPERIMENTAL; NOT YET PEER REVIEWED',model=args.name,points=NPOINTS,
                input_sha256=man['input_sha256'], score_run=runs, metrics={})
    for ctx in ['S','F']:
        for group in ['all','I_all','U_all','q12_factual_worse',
                      'q12_factual_worse_ge_1_3','I_q12_factual_worse_ge_1_3',
                      'U_q12_factual_worse_ge_1_3','q12_factual_not_worse',
                      'gold_q12_factual_worse']:
            measures={k:[] for k in ['p_exec_true_minus_swap','p_alt_swap_minus_true',
                       'identity_switch_effect','placebo_switch_effect','identity_minus_placebo',
                       'value_true_minus_swap','value_true_minus_no_label',
                       'value_swap_minus_no_label','bad_exec_chosen_true','bad_exec_chosen_no_label']}
            tids=[]
            for r in meta:
                pid=r['point_id'];tr=r['traj'];alt=r['swap_arm']; gap=r['q12_gap']
                if group.startswith('I_') and r['layer']!='I':continue
                if group.startswith('U_') and r['layer']!='U':continue
                if group=='q12_factual_worse' and gap<=0:continue
                if group.endswith('q12_factual_worse_ge_1_3') and gap < 1/3-1e-8:continue
                if group=='q12_factual_not_worse' and gap>0:continue
                if group=='gold_q12_factual_worse' and gap<=0:continue
                qkey='qgold' if group.startswith('gold') else 'q3456'
                qq=np.array([vals[pid][i][qkey] for i in range(5)],float)
                if not np.isfinite(qq).all():continue
                p={c:np.array(sc[pid+'__'+ctx+c]['candidate_prob'],float) for c in ('0','E','X','PE','PX')}
                e=p['E'][0]-p['X'][0]; x=p['X'][alt]-p['E'][alt]
                pe=p['PE'][0]-p['PX'][0]; px=p['PX'][alt]-p['PE'][alt]
                values=dict(p_exec_true_minus_swap=e,p_alt_swap_minus_true=x,
                    identity_switch_effect=(e+x)/2,
                    placebo_switch_effect=(pe+px)/2,
                    identity_minus_placebo=(e+x-pe-px)/2,
                    value_true_minus_swap=float((p['E']-p['X'])@qq),
                    value_true_minus_no_label=float((p['E']-p['0'])@qq),
                    value_swap_minus_no_label=float((p['X']-p['0'])@qq),
                    bad_exec_chosen_true=float(p['E'].argmax()==0),
                    bad_exec_chosen_no_label=float(p['0'].argmax()==0))
                for k,v in values.items():measures[k].append(v)
                tids.append(tr)
            result['metrics'][ctx+'/'+group]={k:mean_ci(v,tids,seed=SEED+sum(map(ord,ctx+group+k))) for k,v in measures.items()}
    write_json(out/args.name/'analysis.json',result)
    lines=[f"Experiment 1: {args.name}, {NPOINTS} I/U Forks. Percentages below are pp; 95% trajectory-cluster bootstrap.\n"]
    for key,metrics in result['metrics'].items():
        lines.append('\n'+key)
        for nm in ['identity_switch_effect','placebo_switch_effect','identity_minus_placebo',
                   'p_exec_true_minus_swap','value_true_minus_swap','value_true_minus_no_label']:
            v=metrics[nm]
            if v['n']:lines.append(f"  {nm:30s}: {v['estimate']*100:+.2f} [{v['ci95'][0]*100:+.2f},{v['ci95'][1]*100:+.2f}] n={v['n']} traj={v['n_trajectories']}")
    (out/args.name/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def selftest(_args):
    S=dict(point_id='s000_t02',traj_id='s000',turn=2,actions=['a0','a1','a2','a3','a4'],
           text='Q\nCandidates:\n["a4", "a1", "a0", "a3", "a2"]')
    F=dict(S,text=S['text']+'\nFive feedback records here')
    order=[4,1,0,3,2]
    assert json.loads(S['text'].split('Candidates:\n')[-1])==[S['actions'][i] for i in order]
    assert order.index(0)+1==3 and order.index(2)+1==5
    # No evidence changes when adding a provenance statement to S/F.
    label='Execution metadata at the marked decision: candidate number 3 was selected.'
    assert F['text'].startswith(S['text'])
    changed=S['text']+'\n'+label+F['text'][len(S['text']):]
    assert changed.replace('\n'+label,'')==F['text']
    # Bootstrap preserves a trajectory-cluster estimate rather than pretending points independent.
    ci=mean_ci([0,1,1],["t0","t0","t1"],nboot=500)
    assert ci['n']==3 and abs(ci['estimate']-2/3)<1e-8
    print('SELFTEST OK (candidate display index; F evidence unchanged; bootstrap).')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare','score','analyze','selftest'])
    ap.add_argument('--repo',default=str(ROOT))
    ap.add_argument('--arms',default=str(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet'))
    ap.add_argument('--out',default=str(HERE/'results/exp1_p4'))
    ap.add_argument('--model')
    ap.add_argument('--name',choices=['32b','7b'],default='32b')
    ap.add_argument('--revision',default='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
    args=ap.parse_args()
    if args.command=='score' and not args.model:ap.error('--model is required for score')
    globals()[args.command](args)

if __name__=='__main__': main()
