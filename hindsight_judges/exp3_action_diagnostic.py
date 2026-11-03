"""Hindsight action correction diagnostic (386 original P4 forks).

Four privilege conditions M/L/LN/F x current-turn Student Think present/removed.
Evaluate *action continuation* scores, not Judge ratings or full OPD KL.
Frozen action alternatives and independently replayed q3456 are held fixed.
No training; no data from E4 or main is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import exp1c
from exp2_common import ROOT, HERE, decode_rows, encode_rows, read_json, save_json, sha_file, load_32b

CONDS = ('M', 'L', 'LN', 'F')
THINKS = ('keep', 'drop')
OPEN_ACTION = re.compile(r'<\s*action\s*>|\[\s*action\s*[>\]]', re.I)
CLOSE_THINK = exp1c.THINK_BLOCK


def student_prefix(response: str, mode: str) -> str:
    """Preserve exact written prefix before the *last* action tag, with optional CoT ablation."""
    assert mode in THINKS
    matches = list(OPEN_ACTION.finditer(response))
    if not matches:
        raise ValueError('No tagged action in original Student response')
    # parse_action uses the last action after the last completed Think.
    last_think_end = response.lower().rfind('</think>')
    after = [m for m in matches if m.start() >= last_think_end + len('</think>')]
    action = after[-1] if after else matches[-1]
    before = response[:action.start()]
    if mode == 'drop':
        before = CLOSE_THINK.sub('', before)
        # The experiment changes only CoT, not any non-CoT preamble.
    return before + action.group(0)


def redact_episode(ep: dict, turn: int, strip_other_thinks: bool) -> dict:
    """Mask current action and its duplicated entire response, retain environment observation.

    This is deliberately different from E1c's unguarded L/LN. Prevents the
    privileged log from directly giving away the action being token-scored.
    """
    x = json.loads(json.dumps(ep, ensure_ascii=False))
    for i, step in enumerate(x['steps'], 1):
        if i == turn:
            step['response'] = '[TARGET RESPONSE REDACTED]'
            step['executed_action'] = '[TARGET ACTION REDACTED]'
        elif strip_other_thinks:
            step['response'] = CLOSE_THINK.sub('', step.get('response', ''))
    return x


def extract_decision_prompt(S: dict) -> str:
    text = S['text']
    anchor = '<decision_time_prompt>\n'
    end = '\n</decision_time_prompt>'
    assert text.count(anchor) == 1 and text.count(end) == 1
    return text.split(anchor, 1)[1].split(end, 1)[0]


def make_cases(S, L, F, template, feedback_template):
    ep = exp1c.extract_episode(S, L, template)
    turn = int(S['turn'])
    response = ep['steps'][turn - 1]['response']
    pre = {m:student_prefix(response, m) for m in THINKS}
    assert pre['keep'] != pre['drop'], 'Missing operative <think> span at marked turn'
    feedback_prefix = S['text'] + '\n' + feedback_template
    assert F['text'].startswith(feedback_prefix), 'Original P4 F evidence mismatch'
    feedback = json.loads(F['text'][len(feedback_prefix):])
    assert len(feedback) == 5
    privilege = {
        'M': None,
        'L': redact_episode(ep, turn, False),
        'LN': redact_episode(ep, turn, True),
        'F': feedback,
    }
    prompt = extract_decision_prompt(S)
    rows = []
    for condition in CONDS:
        user = prompt
        if condition != 'M':
            user += ('\n\n[PRIVILEGED ENVIRONMENT EVIDENCE — not part of the Agent history; '
                     'use it to choose the best action, do not copy it]\n'
                     + json.dumps(privilege[condition], ensure_ascii=False)
                     + '\n[/PRIVILEGED ENVIRONMENT EVIDENCE]')
        for mode in THINKS:
            item = dict(id=S['point_id']+'__'+condition+'__'+mode,
                point_id=S['point_id'], traj=S['traj_id'], turn=turn,
                condition=condition, think=mode, prompt=user,
                reply_prefix=pre[mode], actions=S['actions'])
            item['sha256'] = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            rows.append(item)
    return rows


def prepare(args):
    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    meta, e2, _, template, hashes = exp1c.original_sources(repo)
    feedback_template = (repo/'exploratory_P4/prompts/feedback.txt').read_text()
    rows=[]; skipped=[]
    for z in meta:
        S,L,F=[e2[z['point_id']][k] for k in ('S','L','F')]
        try: rows += make_cases(S,L,F,template,feedback_template)
        except (ValueError,AssertionError) as exc:
            skipped.append(dict(point_id=z['point_id'],reason=str(exc)))
    if skipped:
        print('NONOPERATIVE_OR_INVALID:',len(skipped),'of',len(meta),file=sys.stderr)
    if args.require_full and skipped:
        raise RuntimeError('Some of the 386 points lack an operative tagged Think; inspect the audit first')
    assert len(rows) == (len(meta)-len(skipped))*8 and rows
    out.mkdir(parents=True,exist_ok=True)
    file=out/'requests.jsonl.gz'
    if file.exists():
        prev,_ = decode_rows(file)
        assert prev == rows,'Refusing to overwrite a different diagnostic dataset'
    else:encode_rows(file,rows)
    save_json(out/'prepare.json',dict(points=len(rows)//8,total_points=len(meta),skipped=skipped,
        request_hash=decode_rows(file)[1],source_hashes=hashes,
        prompt_protocol='same decision prompt + privileged evidence, current action redacted'))
    print('PREPARED:',len(rows)//8,'of',len(meta),'forks;',len(rows),'Teacher contexts')


def tokenize(tok, row):
    prefix=tok.apply_chat_template([dict(role='user',content=row['prompt'])],
                    tokenize=False,add_generation_prompt=True)+row['reply_prefix']
    pre=tok.encode(prefix,add_special_tokens=False)
    full=[tok.encode(prefix+a+'</action>'+tok.eos_token,add_special_tokens=False) for a in row['actions']]
    start=len(pre)
    for ids in full:
        i=0
        while i<min(len(pre),len(ids)) and pre[i]==ids[i]:i+=1
        start=min(start,i)
    assert len(pre)-start <= 3, 'Tokenizer merge crosses prefix boundary unexpectedly'
    suffix=[ids[start:] for ids in full]
    assert all(len(s)>1 for s in suffix), 'Scorer needs >1 suffix token'
    return dict(**row,prefix_ids=full[0][:start],candidates=[dict(suffix_ids=s) for s in suffix],
                max_tokens=max(map(len,full)))


def score(args):
    import torch
    from transformers import AutoTokenizer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from exp1c_backend import ChunkedScorer
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=Path(args.out).resolve()
    prep=read_json(out/'prepare.json');rows,h=decode_rows(out/'requests.jsonl.gz')
    assert h==prep['request_hash']
    rows=[r for r in rows if r['condition'] in args.conditions]
    pids=sorted({r['point_id'] for r in rows})
    if args.limit>0:
        chosen=set(pids[:args.limit]);rows=[r for r in rows if r['point_id'] in chosen]
    assert rows
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    jobs=[tokenize(tok,r) for r in rows]
    assert max(j['max_tokens'] for j in jobs)<=32768,'No silent truncation'
    # A partial-run folder is *distinct* from a full cohort, preventing silent mixing.
    suffix='full' if args.limit<=0 else f'first{args.limit}'
    dest=out/args.name/suffix; dest.mkdir(parents=True,exist_ok=True)
    file=dest/'scores.jsonl';runfile=dest/'run.json'
    fingerprint=dict(name=args.name,limit=args.limit,conditions=args.conditions,
        dataset_sha256=prep['request_hash'],model=str(Path(args.model).resolve()),
        script_sha256=sha_file(Path(__file__)))
    if runfile.exists():
        old=read_json(runfile)
        assert all(old[k]==v for k,v in fingerprint.items()), 'Resume fingerprint mismatch'
    elif file.exists():raise RuntimeError('Scores without run metadata')
    completed={}
    if file.exists():
        for l in file.read_text().splitlines():
            x=json.loads(l); assert x['id'] not in completed
            completed[x['id']]=x
    assert set(completed)<={j['id'] for j in jobs}
    if runfile.exists() and old.get('complete'):
        assert sha_file(file)==old['scores_sha256'];print('Already complete');return
    with sdpa_kernel(SDPBackend.MATH):
        if args.name=='32b':
            scorer,model_meta=load_32b(Path(args.model),Path(args.repo))
        else:
            # Standard frozen P4 BF16 7B checkpoint; same original scoring stack.
            from exp1c_backend import ChunkedScorer
            original=read_json(Path(args.repo)/'exploratory_P4/e2_7b/run.json')
            weights={p.name:sha_file(p) for p in sorted(Path(args.model).glob('*.safetensors'))}
            assert weights and weights==original['model_sha256'], '7B weights differ from P4'
            script=Path(args.repo)/'exploratory_P4/p4_e2_backend.py'
            assert sha_file(script)==original['scorer_sha256']
            spec=importlib.util.spec_from_file_location('p4_7b_diagnostic',script)
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.MODEL=str(args.model)
            frozen=m.Scorer();scorer=ChunkedScorer(frozen)
            model_meta=dict(weights=weights,precision='frozen_p4_bf16_transformer_fp32_head')
        def probs(v):
            ll=np.array([sum(x) for x in v],float);p=np.exp(ll-ll.max());return p/p.sum()
        gate=[]
        for j in [min(jobs,key=lambda a:a['max_tokens']),max(jobs,key=lambda a:a['max_tokens'])]:
            p=probs(scorer.score(j));q=probs([scorer.direct(j,k) for k in range(5)])
            tv=float(np.abs(p-q).sum()/2)
            gate.append(dict(id=j['id'],tv=tv,passed=tv<.005))
        save_json(dest/'numerical_gate.json',gate)
        assert all(x['passed'] for x in gate),'Numerical gate failed'
        save_json(runfile,{**fingerprint,**model_meta,"complete":False,"gate":gate})
        with file.open('a',buffering=1) as f:
            for i,j in enumerate(jobs):
                if j['id'] in completed:continue
                ll=np.array([sum(x) for x in scorer.score(j)],float)
                assert np.isfinite(ll).all()
                p=np.exp(ll-ll.max());p/=p.sum()
                item=dict(id=j['id'],point_id=j['point_id'],condition=j['condition'],think=j['think'],
                          sha256=j['sha256'],sequence_logp=ll.tolist(),prob=p.tolist())
                f.write(json.dumps(item,allow_nan=False)+'\n');completed[item['id']]=item
                if (i+1)%20==0:print('SCORED',args.name,len(completed),'/',len(jobs),flush=True)
    assert len(completed)==len(jobs)
    save_json(runfile,{**fingerprint,**model_meta,"complete":True,"gate":gate,"scores_sha256":sha_file(file),
                "n":len(completed)})
    print('DONE',args.name,len(completed))


def group_ci(v,clusters,nboot=2000):
    return exp1c.mean_ci(v,clusters,nboot=nboot)


def analyze(args):
    out=Path(args.out);pre=read_json(out/'prepare.json');rows,h=decode_rows(out/'requests.jsonl.gz')
    assert h==pre['request_hash']
    cohort='full' if args.limit<=0 else f'first{args.limit}'
    scores={}
    for name in args.models:
        folder=out/name/cohort;run=read_json(folder/'run.json')
        assert run['complete'] and sha_file(folder/'scores.jsonl')==run['scores_sha256']
        for l in (folder/'scores.jsonl').read_text().splitlines():
            z=json.loads(l);scores[(name,z['point_id'],z['condition'],z['think'])]=z
    repo=Path(args.repo)
    meta=read_json(repo/'hindsight_judges/results/exp1_p4/pointmeta.json')
    meta={x['point_id']:x for x in meta}
    arms_hash=read_json(repo/'exploratory_P4/e2.json')['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    arms=exp1c.read_arms(repo,Path(args.arms),arms_hash)
    mids=sorted({z['point_id'] for z in rows})
    if args.limit>0:mids=mids[:args.limit]
    measured=defaultdict(list);clusters=defaultdict(list);paired={}
    def put(name,v,cl):
        measured[name].append(float(v));clusters[name].append(cl)
    for name in args.models:
        for pid in mids:
            if not all((name,pid,c,m) in scores for c in args.conditions for m in THINKS):continue
            zz=meta[pid];q=np.array([arms[pid][j]['q3456'] for j in range(5)],float)
            if not np.isfinite(q).all():continue
            b=zz['swap_arm'];gap=float(q[0]-q[b]);tr=zz['traj']
            for mode in THINKS:
                # Student reference is the 7B no-privilege model at this SAME prefix.
                if ('7b',pid,'M',mode) not in scores:continue
                ref=scores[('7b',pid,'M',mode)];pr=np.array(ref['prob']);llr=np.array(ref['sequence_logp'])
                for cond in args.conditions:
                    z=scores[(name,pid,cond,mode)]
                    p=np.array(z['prob']);ll=np.array(z['sequence_logp'])
                    tag=f'{name}_{cond}_{mode}'
                    correction=float((ll[0]-ll[b])-(llr[0]-llr[b]))
                    put(tag+'_correction_logodds',correction,tr)
                    put(tag+'_value_gain',float((p-pr)@q),tr)
                    put(tag+'_factual_logp_delta',float(ll[0]-llr[0]),tr)
                    put(tag+'_factual_probability',float(p[0]),tr)
                    put(tag+'_bad_factual_correction',correction if gap < -1e-6 else np.nan,tr)
                    paired[(name,pid,cond,mode)]=dict(correction=correction,value_gain=float((p-pr)@q),gap=gap,traj=tr)
            for cond in args.conditions:
                a=paired.get((name,pid,cond,'keep'));b2=paired.get((name,pid,cond,'drop'))
                if a and b2:
                    put(f'{name}_{cond}_drop_minus_keep_value_gain',b2['value_gain']-a['value_gain'],tr)
                    put(f'{name}_{cond}_drop_minus_keep_correction',b2['correction']-a['correction'],tr)
    stats={}
    for key,v in measured.items():
        v=np.array(v,float);keep=np.isfinite(v)
        if keep.any():stats[key]=group_ci(v[keep].tolist(),np.array(clusters[key])[keep].tolist())
    # Clustered slope: correct direction implies positive change of factual log odds with factual Q advantage.
    for name in args.models:
        for cond in args.conditions:
            for mode in THINKS:
                cases=[paired[(name,pid,cond,mode)] for pid in mids if (name,pid,cond,mode) in paired]
                if len(cases)>3:
                    key=f'{name}_{cond}_{mode}_slope_vs_qgap'
                    stats[key]=exp1c.slope_ci([x['gap'] for x in cases],[x['correction'] for x in cases],
                                              [x['traj'] for x in cases],nboot=2000)
    save_json(out/f'analysis_{cohort}.json',dict(n_points=len(mids),status='diagnosis, NOT proof of OPD gradient quality',metrics=stats))
    for k,v in stats.items():
        if k.endswith(('slope_vs_qgap','drop_minus_keep_value_gain','bad_factual_correction')):
            print(k,v)


def selftest(_):
    s={'text':'PREF\n<decision_time_prompt>\nTask\n</decision_time_prompt>\nOther',
       'actions':['open fridge','look','take apple','go west','examine desk'],'point_id':'s000_t01','traj_id':'s000','turn':1}
    ep={'steps':[dict(turn=1,response='<think>open fridge, looks promising</think>\n<action>open fridge</action>',
         executed_action='open fridge',observation_after='nothing'),
         dict(turn=2,response='<think>try again</think><action>look</action>',executed_action='look',observation_after='nothing')]}
    assert '<think>' in student_prefix(ep['steps'][0]['response'],'keep')
    assert '<think>' not in student_prefix(ep['steps'][0]['response'],'drop')
    L=redact_episode(ep,1,False);LN=redact_episode(ep,1,True)
    assert L['steps'][0]==LN['steps'][0] and L['steps'][0]['observation_after']=='nothing'
    assert '<think>' in L['steps'][1]['response'] and '<think>' not in LN['steps'][1]['response']
    assert extract_decision_prompt(s)=='Task'
    print('SELFTEST PASS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('prepare','score','analyze','selftest'))
    p.add_argument('--repo',default=str(ROOT));p.add_argument('--out',default=str(HERE/'results/action_diagnostic'))
    p.add_argument('--arms',default=str(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet'))
    p.add_argument('--name',choices=('7b','32b'),default='7b')
    p.add_argument('--models',nargs='+',choices=('7b','32b'),default=['7b','32b'])
    p.add_argument('--model');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--conditions',nargs='+',choices=CONDS,default=list(CONDS))
    p.add_argument('--require-full',action='store_true')
    args=p.parse_args()
    if args.command=='score' and not args.model:p.error('score requires --model')
    globals()[args.command](args)

if __name__=='__main__':main()
