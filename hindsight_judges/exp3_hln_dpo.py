"""E4 H-vs-H-LN direct learning bridge, using the FROZEN E4 DPO trainer.

Both H_re and H_LN are newly scored with identical E1c 32B FP16 backend.
This is E4 DPO, NOT OPD. Never subtract the old E4 BF16 H raw scores.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np

import exp1c
from exp2_common import ROOT,HERE,decode_rows,encode_rows,read_json,save_json,sha_file,load_32b

HERE4=ROOT/'exploratory_E4'
CONDS=('L','LN')


def make_ln(l_row,template):
    prefix=l_row['text'].split('\n'+template,1)
    assert len(prefix)==2, 'Could not locate original E4 log template'
    ep=json.loads(prefix[1]); assert ep['steps']
    removed=0
    for s in ep['steps']:
        s['response'],n=exp1c.THINK_BLOCK.subn('',s['response'])
        removed+=n
    text=prefix[0]+'\n'+template+json.dumps(ep,ensure_ascii=False)
    assert text != l_row['text'] or removed==0
    row=dict(l_row,condition='LN',request_id=l_row['point_id']+'__LN',text=text)
    row.pop('request_sha256',None)
    row['request_sha256']=exp1c.sha(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())
    return row,removed


def prepare(args):
    root=Path(args.repo).resolve()
    src=Path(args.e4_data).resolve()/'judge'
    prep=read_json(src/'preparation.json')
    raw=gzip.decompress((src/'requests.jsonl.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==prep['requests_sha256'],'E4 original requests changed'
    lrows=[json.loads(s) for s in raw.decode().splitlines() if s and json.loads(s)['condition']=='L']
    assert len(lrows)==prep['points'], 'Not all E4 H points found'
    template=(root/'exploratory_P4/prompts/log.txt').read_text()
    rows=[];nthink=0
    for l in sorted(lrows,key=lambda r:r['point_id']):
        ln,n=make_ln(l,template); nthink+=n
        rows += [l,ln]
    assert len({r['request_id'] for r in rows})==len(rows)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    file=out/'requests.jsonl.gz'
    if file.exists():
        old,_=decode_rows(file);assert old==rows,'Existing H-LN requests differ'
    else:encode_rows(file,rows)
    save_json(out/'prepare.json',dict(points=len(rows)//2,contexts=len(rows),think_blocks_removed=nthink,
        source_e4_request_sha256=prep['requests_sha256'],request_sha256=decode_rows(file)[1],
        producer_sha256=sha_file(Path(__file__))))
    print('E4 H-LN PREPARED:',len(rows)//2,'points;',nthink,'removed think blocks')


def tokenize(tok,job):
    return exp1c.tokenize(tok,job)


def score(args):
    import torch
    from transformers import AutoTokenizer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    out=Path(args.out).resolve();man=read_json(out/'prepare.json');rows,h=decode_rows(out/'requests.jsonl.gz')
    assert h==man['request_sha256']
    if args.limit>0:
        pids=sorted({x['point_id'] for x in rows})[:args.limit]
        rows=[r for r in rows if r['point_id'] in set(pids)]
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    jobs=[tokenize(tok,x) for x in rows]
    assert max(j['max_tokens'] for j in jobs)<=exp1c.MAX_TOKENS,'No silent truncation'
    folder=out/('full' if args.limit<=0 else f'first{args.limit}')
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/'scores.jsonl';run=folder/'run.json'
    fingerprint=dict(request_sha256=man['request_sha256'],limit=args.limit,
        script_sha256=sha_file(Path(__file__)),model=str(Path(args.model).resolve()))
    if run.exists():
        old=read_json(run);assert all(old[k]==v for k,v in fingerprint.items()),'Resume mismatch'
    elif file.exists():raise RuntimeError('Scores exist without matching run.json')
    done={}
    if file.exists():
        for l in file.read_text().splitlines():
            r=json.loads(l);assert r['request_id'] not in done
            done[r['request_id']]=r
    assert set(done)<={j['request_id'] for j in jobs}
    if run.exists() and old.get('complete'):
        assert sha_file(file)==old['scores_sha256'];print('ALREADY COMPLETE');return
    with sdpa_kernel(SDPBackend.MATH):
        scorer,modelmeta=load_32b(Path(args.model),Path(args.repo))
        if run.exists():assert old['weights']==modelmeta['weights']
        def probs(rows):
            ll=np.array([sum(x) for x in rows]);p=np.exp(ll-ll.max());return p/p.sum()
        checks=[]
        for item in [min(jobs,key=lambda r:r['max_tokens']),max(jobs,key=lambda r:r['max_tokens'])]:
            a=probs(scorer.score(item))
            b=probs([scorer.direct(item,j) for j in range(5)])
            tv=float(np.abs(a-b).sum()/2)
            checks.append(dict(id=item['request_id'],tv=tv,passed=tv<.005))
        save_json(folder/'numerical_gate.json',checks)
        assert all(c['passed'] for c in checks),'Numerical gate failed; no scores written'
        save_json(run,{**fingerprint,**modelmeta,'complete':False,'numerical_gate':checks})
        with file.open('a',buffering=1) as f:
            for k,j in enumerate(jobs):
                if j['request_id'] in done:
                    assert done[j['request_id']]['request_sha256']==j['request_sha256']
                    continue
                ll=np.array([sum(x) for x in scorer.score(j)],float)
                assert np.isfinite(ll).all()
                p=np.exp(ll-ll.max());p/=p.sum()
                item=dict(request_id=j['request_id'],point_id=j['point_id'],condition=j['condition'],
                          request_sha256=j['request_sha256'],candidate_prob=p.tolist())
                f.write(json.dumps(item,allow_nan=False)+'\n');done[item['request_id']]=item
                if (k+1)%20==0:print('H-LN score',len(done),'/',len(jobs),flush=True)
        assert len(done)==len(jobs)
        save_json(run,{**fingerprint,**modelmeta,'complete':True,'numerical_gate':checks,
                       'scores_sha256':sha_file(file),'contexts':len(done)})
    print('H-LN SCORED',len(done))


def make_pairs(points,scores,cap):
    """Make E4-format DPO pairs; matched sample budget for both arms.

    An additional common-point check (H_re_common/H_LN_common) is emitted at
    no extra judging cost, so different selection sets can be distinguished.
    """
    decisions={}
    for cond,arm in [('L','H_re'),('LN','H_LN')]:
        rows=[]
        for r in points:
            pid=r['point_id'];p=np.array(scores[(pid,cond)]['candidate_prob'])
            j=int(p.argmax())
            if j==0:continue
            assert 0<j<5
            rows.append(dict(point_id=pid,prompt=r['prompt'],chosen=r['arm_responses'][j],
                 rejected=r['arm_responses'][0],preferred=j,rejected_arm=0))
        decisions[arm]=rows
    n=min(len(decisions['H_re']),len(decisions['H_LN']),cap)
    assert n>0,'No non-factual preference pairs'
    # Same *number* of examples; E4's cap rule is preserved but source sets may differ.
    def sample(rows,n):
        rows=sorted(rows,key=lambda x:hashlib.sha256(('Hindsight-HLN:'+x['point_id']).encode()).hexdigest())
        return sorted(rows[:n],key=lambda x:x['point_id'])
    paired={name:sample(rows,n) for name,rows in decisions.items()}
    both=set(x['point_id'] for x in decisions['H_re'])&set(x['point_id'] for x in decisions['H_LN'])
    both=sorted(both,key=lambda x:hashlib.sha256(('Hindsight-HLN:'+x).encode()).hexdigest())[:cap]
    for name,rows in decisions.items():
        paired[name+'_common']=sorted([r for r in rows if r['point_id'] in set(both)],key=lambda r:r['point_id'])
    return paired,dict(raw_count={n:len(v) for n,v in decisions.items()},
                       matched_n=n,common_point_n=len(both))


def pairs(args):
    from pathlib import Path
    root=Path(args.repo);out=Path(args.out)
    folder=out/('full' if args.limit<=0 else f'first{args.limit}')
    run=read_json(folder/'run.json'); assert run['complete']
    assert sha_file(folder/'scores.jsonl')==run['scores_sha256']
    rows,man=decode_rows(out/'requests.jsonl.gz');assert man==read_json(out/'prepare.json')['request_sha256']
    if args.limit>0:
        selected=set(sorted({r['point_id'] for r in rows})[:args.limit]);rows=[r for r in rows if r['point_id'] in selected]
    expected={r['request_id']:r for r in rows}
    scores={}
    for line in (folder/'scores.jsonl').read_text().splitlines():
        z=json.loads(line);assert z['request_id'] in expected and z['request_sha256']==expected[z['request_id']]['request_sha256']
        scores[(z['point_id'],z['condition'])]=z
    assert len(scores)==len(rows)
    original=sorted((json.loads(p.read_text()) for p in (Path(args.e4_data).resolve()/'points').glob('*.json')
                     if json.loads(p.read_text())['cf_defined']),key=lambda x:x['point_id'])
    selected={r['point_id'] for r in rows}
    points=[p for p in original if p['point_id'] in selected]
    assert len(points)==len(selected),'Missing original E4 point JSON'
    pairmap,info=make_pairs(points,scores,args.cap)
    target=Path(args.e4_data).resolve()/'pairs';target.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,rs in pairmap.items():
        p=target/(name+'.jsonl')
        body=''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in rs)
        if p.exists(): assert p.read_text()==body,'Existing E4 pairs differ; no overwrite'
        else:p.write_text(body)
        hashes[name]=sha_file(p)
    save_json(folder/'pairs_manifest.json',dict(**info,source_score_sha256=run['scores_sha256'],
            pairs_sha256=hashes,counts={k:len(v) for k,v in pairmap.items()}))
    print('PAIRS READY',info,'(H_re and H_LN share identical number of examples)')
    print('Train using: python hindsight_judges/exp3_hln_dpo.py train --arm H_re --seed 0 --lr 5e-6')


def train(args):
    """Zero-change reuse of original frozen E4 e4_06_train.py (DPO implementation)."""
    assert args.arm in ('H_re','H_LN','H_re_common','H_LN_common')
    sys.path.insert(0,str(Path(args.repo)/'exploratory_E4'))
    import e4_common as E
    import e4_06_train as T
    E.DATA=str(Path(args.e4_data).resolve())
    if args.arm not in E.ARMS_TRAINED:E.ARMS_TRAINED.append(args.arm)
    # Keep original E4 CLI and all hyperparameter semantics untouched.
    sys.argv=['e4_06_train.py','--arm',args.arm,'--seed',str(args.seed),'--lr',str(args.lr)]
    if args.model:sys.argv+=['--model',str(args.model)]
    if args.smoke:sys.argv+=['--smoke']
    T.main()



def evaluate(args):
    """Reuse E4 held-out rollout harness without modifying its frozen source."""
    assert args.split in ('valid_unseen','valid_seen')
    sys.path.insert(0,str(Path(args.repo)/'exploratory_E4'))
    import e4_common as E
    import e4_07_eval as original
    E.DATA=str(Path(args.e4_data).resolve())
    tags=[f'{arm}_s{seed}' for arm in ('H_re','H_LN') for seed in args.seeds]
    sys.argv=['e4_07_eval.py','--split',args.split,'--no_base','--adapters']+tags
    original.main()

def analyze(args):
    sys.path.insert(0,str(Path(args.repo)/'exploratory_E4'))
    import e4_common as E
    split=args.split;games=read_json(Path(args.repo)/'exploratory_E4/games.json')[split]
    d={}
    for arm in ('H_re','H_LN'):
        mats=[]
        for seed in args.seeds:
            path=Path(args.e4_data).resolve()/'rollout'/split/f'{arm}_s{seed}'/'summary.json'
            summary=read_json(path)
            mats.append(np.array([summary['per_game'][str(g['seed'])] for g in games],float))
        d[arm]=np.concatenate(mats,axis=1)
    diff=E.game_bootstrap_diff(d['H_LN'],d['H_re'])
    result=dict(split=split,seeds=args.seeds,success={x:float(100*d[x].mean()) for x in d},
                difference_H_LN_minus_H_re_pp=diff,method='DPO matched count, same 32B FP16 scorer')
    save_json(Path(args.out)/'hln_dpo_result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def selftest(_):
    templ='LOG JSON:\n'
    ep=dict(steps=[dict(response='<think>Reason</think><action>look</action>',executed_action='look',observation_after='fine')])
    row=dict(text='BASE\n'+templ+json.dumps(ep),point_id='p',request_id='p__L',condition='L',actions=['look','open','go','take','put'])
    ln,n=make_ln(row,templ)
    assert n==1 and '<think>' not in ln['text']
    assert json.loads(ln['text'].split(templ)[1])['steps'][0]['executed_action']=='look'
    pts=[dict(point_id='p'+str(i),prompt='x',arm_responses=['f','a','b','c','d']) for i in range(8)]
    scores={(p['point_id'],c):dict(candidate_prob=([0.05,.6,.15,.1,.1] if (i+c_i)%2==0 else [0.6,.05,.15,.1,.1]))
            for i,p in enumerate(pts) for c_i,c in enumerate(CONDS)}
    paired,info=make_pairs(pts,scores,10)
    assert len(paired['H_re'])==len(paired['H_LN'])==info['matched_n']
    print('SELFTEST PASS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('prepare','score','pairs','train','evaluate','analyze','selftest'))
    p.add_argument('--repo',default=str(ROOT));p.add_argument('--out',default=str(HERE/'results/hln_dpo'))
    p.add_argument('--e4-data',default=str(ROOT/'exploratory_E4/data'),help='Original E4 data/, or restored nonweight archive data/')
    p.add_argument('--model',default=None);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--cap',type=int,default=1000)
    p.add_argument('--arm',default='H_LN');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    # E4 selected 5e-6 on tune (every E4 adapter train_log.json records lr=5e-06); 2e-5 is NOT the E4 setting.
    p.add_argument('--lr',type=float,default=5e-6)
    p.add_argument('--split',choices=('valid_unseen','valid_seen'),default='valid_unseen')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.command=='score' and not args.model:p.error('--model (32B) is required')
    globals()[args.command](args)

if __name__=='__main__':main()
