"""Regrade complete AIME24 archives and localize the observed accuracy loss."""
import bisect,gzip,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[1]/'cap_trajectories'))
import cap_audit as audit
from cap_audit import org
root=org.ROOT
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def official_box(t):
    # Reuse the exact archived grader/extractor definitions without importing an inference stack.
    return extract(t)
from ast import parse,Module
tree=parse(audit.GRADER.read_text())
ns={}
exec(compile(Module(body=[n for n in tree.body if getattr(n,'name','')=='extract_boxed_answer'],type_ignores=[]),str(audit.GRADER),'exec'),ns)
extract=ns['extract_boxed_answer']
arms=('B','R-noclip@100','R@100','R-contrast@100')
def measure(arm):
    raw,digest=org.source('aime24',arm)
    assert raw['seed']==1001 and raw['temperature']==1 and raw['max_new_tokens']==38912 and raw['enable_thinking']
    tok=org.Tokenizer.from_file(str(audit.TOKENIZER))
    rows=[];cases=[];sources={'answers':digest}
    if arm!='B':
        model,step=arm.split('@')
        p=root/'run'/model/f'checkpoint-{step}'/'adapter_model.safetensors'
        ack=read(root/'run/evaluation'/arm/'aime24/1001.receipt.json')
        assert sha(p)==ack['adapter_sha256']
        sources['adapter']=ack['adapter_sha256']
    for qi,q in enumerate(raw['results']):
        for j,g in enumerate(q['generations']):
            t=g['full_generation'];gold=q['ground_truth'];n=g['generated_tokens']
            pred=official_box(t)
            assert (pred is not None)==g['formatted'] and bool(audit.grade(pred,gold))==g['correct']
            enc=tok.encode(t,add_special_tokens=False)
            ends=[b for a,b in enc.offsets];ids=enc.ids
            pos=t.find('</think>');close=(bisect.bisect_left(ends,pos+8)+1) if pos>=0 else None
            items=list(audit.candidates(t));hit=next(((k,v) for k,v in items if audit.grade(v,gold)),None)
            first=bisect.bisect_left(ends,hit[0])+1 if hit else None
            seq=ids[-1024:];grams=[tuple(seq[i:i+8]) for i in range(len(seq)-7)]
            repeat=1-len(set(grams))/max(1,len(grams))
            row=dict(q=qi,j=j,problem_id=q['problem_id'],gold=gold,pred=pred,cap=g['finish_reason']=='length',correct=g['correct'],
                     n=n,missing=pred is None,close=close,first_right=first,tail_repeat8=repeat,
                     prior_right=first is not None,has_close=close is not None,
                     after_close=n-close if close else None)
            rows.append(row)
            if arm=='R-contrast@100' and row['cap']:
                cases.append(dict(q=qi,j=j,problem=q['problem'],gold=gold,ending=t[-250:],close=close,
                                  first_right=first,explicit_snippet=t[max(0,hit[0]-180):hit[0]+85] if hit else None))
    return rows,cases,sources
def main():
    allrows={};out={};hashes={}
    for arm in arms:
        rows,cases,hashes[arm]=measure(arm);allrows[arm]=rows
        if cases:(D/'capped_endings.json').write_text(json.dumps(cases,ensure_ascii=False,indent=1))
        counts=Counter()
        def median(r,key):return float(np.median([v[key] for v in r if v[key] is not None]))
        for r in rows:
            state=('cap' if r['cap'] else 'stop')+('/closed' if r['has_close'] else '/open')+('/right' if r['correct'] else '/missing' if r['missing'] else '/wrong')
            counts[state]+=1
        out[arm]=dict(counts=counts,cap=sum(v['cap'] for v in rows),missing=sum(v['missing'] for v in rows),
                      close=sum(v['has_close'] for v in rows),ever_right=sum(v['prior_right'] for v in rows),
                      right_then_missing=sum(v['prior_right'] and v['missing'] for v in rows),
                      correct=sum(v['correct'] for v in rows),median_close_conditional=median(rows,'close'),
                      mean_thinking_tokens=np.mean([v['close'] or v['n'] for v in rows]),
                      mean_after_close_unconditional=np.mean([v['after_close'] or 0 for v in rows]),mean_length=np.mean([v['n'] for v in rows]),
                      median_cap_tail_repeat8=median([v for v in rows if v['cap']],'tail_repeat8'),
                      median_stop_tail_repeat8=median([v for v in rows if not v['cap']],'tail_repeat8'))
    rng=np.random.default_rng(20261124);w=rng.multinomial(30,np.full(30,1/30),10000)/30
    paired={}
    for key in ['cap','missing','has_close','correct','prior_right','n']:
        z=np.array([[v[key] for v in allrows[a]] for a in ('R-contrast@100','R@100')],float)
        delta=(z[0]-z[1]).reshape(30,12).mean(1);paired[key]=[delta.mean(),*np.quantile(w@delta,[.025,.975])]
    with gzip.open(D/'responses.jsonl.gz','wt') as f:
        for arm,rows in allrows.items():
            for r in rows:f.write(json.dumps(dict(arm=arm,**r))+'\n')
    (D/'behavior.json').write_text(json.dumps(dict(by_arm=out,paired_diff=paired,sha256=hashes),indent=2)+'\n')
if __name__=='__main__':main()
