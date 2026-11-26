"""First-answer timing and exact accuracy decomposition on archived generations."""
import bisect
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import analyze as org

ARMS=('B','R@100','T0@100','R-noclip@100','R-cond@100','P@100','Rres@100')
RULES=('explicit','boxed')
KEYS=('length','before','after','covered','first','wrong_right','right_wrong','final','missing_to_correct')
RAW=org.ROOT/'run/first-answer-decomposition'
RAW.mkdir(exist_ok=True)

def receipt(ds,arm):
    if ds in org.SETS[:2]:
        return org.ROOT/'run/math500-long-seed1001-n2'/f'{ds}-{arm}'/'receipt.json'
    return org.ROOT/'run/evaluation'/arm/ds/'1001.receipt.json'

def measure(task):
    ds,arm=task
    data,h=org.source(ds,arm)
    assert data['seed']==1001 and data['enable_thinking']==(ds!='nonthinking')
    qs={};timing={};n=2 if ds in org.SETS[:2] else 12
    for q in data['results']:
        key=hashlib.sha256(q['problem'].encode()).hexdigest()
        assert key not in qs and len(q['generations'])==n
        values=[]
        for g in q['generations']:
            text=g['full_generation'];b=org.box(text)
            ends=[e for _,e in org.tok.encode(text,add_special_tokens=False).offsets]
            row=[]
            for item in (org.first(text,b),b):
                end=len(ends) if not item else min(len(ends),bisect.bisect_left(ends,item[0])+1)
                final=int(g['correct'])
                correct=int(bool(item) and (final if item[1]==g['predicted_answer'] else org.grade(item[1],q['ground_truth'])))
                wr=(1-correct)*final;rw=correct*(1-final)
                assert final==correct+wr-rw
                row.append([len(ends),end,len(ends)-end,int(bool(item)),correct,wr,rw,final,int(not item)*final])
            values.append(row)
        qs[key]=np.mean(values,axis=0).tolist()
        timing[key]=np.array(values)[:,:,[1,2,3]].tolist()
    (RAW/f'{ds}-{arm}.json').write_text(json.dumps(dict(sha=h,qs=qs,timing=timing)))
def ci(v):
    a,b=np.nanquantile(v[1:],[.025,.975],axis=0)
    return [float(v[0]),float(a),float(b)]
def summarize(x):
    return {rule:{k:ci(x[:,i,j]) for j,k in enumerate((*KEYS,'first_given_covered'))} for i,rule in enumerate(RULES)}
def fmt(v,f=100):
    return '%.2f [%.2f, %.2f]' % tuple(f*n for n in v)
def main():
    tasks=[(d,a) for d in org.SETS for a in ARMS if receipt(d,a).is_file()]
    with ProcessPoolExecutor(max_workers=6,initializer=org.init) as pool:list(pool.map(measure,tasks))
    out={};contest={}
    for index,ds in enumerate(org.SETS):
        by={};meta={};qs=None;common={}
        for d,arm in tasks:
            if d!=ds:continue
            data=json.loads((RAW/f'{ds}-{arm}.json').read_text())
            ks=sorted(data['qs']); meta[arm]=data['sha']
            if qs is None:
                qs=ks;n=len(qs);rng=np.random.default_rng([org.SEED,index])
                w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),10000)])/n
            assert qs==ks
            timings=np.array([data['timing'][k] for k in qs])
            if arm=='B':base_timings=timings
            mask=timings[:,:,:,2]*base_timings[:,:,:,2]
            den=w@mask.mean(1)
            changes=(timings[:,:,:,:2]-base_timings[:,:,:,:2])*mask[:,:,:,None]
            vals=(w@changes.mean(1).reshape(n,-1)).reshape(-1,2,2)/den[:,:,None]
            common[arm]={r:{k:ci(vals[:,i,j]) for j,k in enumerate(('before','after'))} for i,r in enumerate(RULES)}
            a=np.array([data['qs'][k] for k in qs])
            x=(w@a.reshape(n,-1)).reshape(-1,2,len(KEYS))
            np.testing.assert_allclose(x[:,:,7],x[:,:,4]+x[:,:,5]-x[:,:,6],atol=1e-12)
            by[arm]=np.concatenate((x,(x[:,:,4]/x[:,:,3])[:,:,None]),axis=2)
        if index>=2:contest[ds]=by
        out[ds]=dict(n=n,sha256=meta,both_detected_timing_difference=common,models={a:summarize(v) for a,v in by.items()},paired={a:summarize(v-by['B']) for a,v in by.items() if a!='B'})
    arms=set.intersection(*(set(d) for d in contest.values()))
    by={a:np.mean([d[a] for d in contest.values()],axis=0) for a in sorted(arms)}
    out['contest_equal_weight']=dict(n=90,models={a:summarize(v) for a,v in by.items()},paired={a:summarize(v-by['B']) for a,v in by.items() if a!='B'})
    (org.HERE/'first_scores.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# 首答案时机、正确率与最终正确率分解','','首答案前含提出答案的 token。未检出者在全体分母中计首答案错，整条归入首答案前。另报检出者正确率和覆盖率；缺失首答案而最终正确的数量单独核对。区间按题配对重抽 10,000 次；三竞赛等权只纳入三个基准都有归档的模型。']
    for ds,r in out.items():
        lines+=['','## '+ds,'','|模型|口径|首答案前 token|之后 token|覆盖率 %|首答案正确率 %（全体）|首答案正确率 %（检出）|最终正确率 %|','|---|---|---:|---:|---:|---:|---:|---:|']
        for arm,rules in r['models'].items():
            for rule,v in rules.items():
                lines.append('|'+arm+'|'+rule+'|'+'|'.join(fmt(v[k],f) for k,f in [('before',1),('after',1),('covered',100),('first',100),('first_given_covered',100),('final',100)])+'|')
        lines+=['','|相对 B（百分点）|口径|首答案差|先错后对差|先对后错差（公式中减去）|最终差|首答案前长度差|首答案后长度差|','|---|---|---:|---:|---:|---:|---:|---:|']
        for arm,rules in r['paired'].items():
            for rule,v in rules.items():
                lines.append('|'+arm+'−B|'+rule+'|'+'|'.join(fmt(v[k],1 if k in ('before','after') else 100) for k in ('first','wrong_right','right_wrong','final','before','after'))+'|')
    (org.HERE/'FIRST_ANSWERS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
