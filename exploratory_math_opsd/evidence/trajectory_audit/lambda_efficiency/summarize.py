import csv,gzip,json
from itertools import combinations
import numpy as np
from measure import ARMS,BUDGET,HERE

def load(ds,m):
    with gzip.open(HERE/f'{ds}_{m}.jsonl.gz') as f:return [json.loads(l) for l in f]
def mean(x):return float(np.mean(x)) if len(x) else None
def dist(x):
    if not x:return {}
    qs=(10,25,50,75,90,95,99)
    return dict(n=len(x),mean=mean(x),std=float(np.std(x)),**{f'p{p}':float(np.percentile(x,p)) for p in qs})
def strata(r):return ('cap' if r['cap'] else 'stop')+'_'+('right' if r['correct'] else 'missing' if r['missing'] else 'wrong')
def group(rs,ds):
    if not rs:return {'n':0}
    n=len(rs);tokens=sum(r['tokens'] for r in rs);correct=sum(r['correct'] for r in rs);c=[r for r in rs if r['cap']];u=[r for r in rs if not r['cap']]
    out=dict(n=n,tokens=tokens,length=dist([r['tokens'] for r in rs]),accuracy=correct/n,
        tokens_per_correct=tokens/correct if correct else None,cap=len(c)/n,missing=mean([r['missing'] for r in rs]),
        missing_given_cap=mean([r['missing'] for r in c]),missing_given_stop=mean([r['missing'] for r in u]),
        cap_and_missing=sum(r['missing'] for r in c)/n,cap_tokens=sum(r['tokens'] for r in c)/tokens,
        stopped_length=dist([r['tokens'] for r in u]),end_think=mean([r['end_think'] for r in rs]),formatted=mean([r['formatted'] for r in rs]),
        budget=np.mean([r['official_budget'] for r in rs],axis=0).tolist())
    for rule in ('explicit','boxed'):
        hit=[r for r in rs if r[rule]['first_right'] is not None]
        pre=sum(r[rule]['first_right'] for r in hit)
        post=sum(r['reencoded']-r[rule]['first_right'] for r in hit)
        transitions={}
        for r in rs:
            first=r[rule]['first'];k=('missing' if not first else 'right' if first[2] else 'wrong')+'_'+('right' if r['correct'] else 'missing' if r['missing'] else 'wrong')
            transitions[k]=transitions.get(k,0)+1
        out[rule]=dict(ever=len(hit)/n,first_right_mean=mean([r[rule]['first_right'] for r in hit]),pre=pre,post=post,
            no_hit=sum(r['reencoded'] for r in rs if r[rule]['first_right'] is None),transitions=transitions,
            hit_then_cap_or_missing=sum(r['cap'] or r['missing'] for r in hit),hits=len(hit),
            by_budget=[sum(r[rule]['first_right']<=b for r in hit)/n for b in BUDGET[ds]],
            **{k:sum(r[rule][k] for r in rs) for k in ('wr','rw','changes','repeated_wrong','final_matches_prior')})
    out['segments']=[]
    for i in range(5):
        ss=[r['segments'][i] for r in rs];v={k:sum(s[k] for s in ss) for k in ss[0]}
        v['longest_mean']=mean([s['longest'] for s in ss]);v['longest_max']=max(s['longest'] for s in ss)
        out['segments'].append(v)
    return out

def per_question(r):
    s=r['segments'][0];hit=r['explicit']['first_right'];h=int(hit is not None);rep=s['rep_equation']
    return [r['tokens'],r['cap'],r['correct'],h,(hit or 0),(r['reencoded']-hit if h else 0),s['cov8']/s['n'],rep/max(s['equation'],1),int(s['role']>0),int(s['offtopic_cue']>0),int(s.get('script_switch',0)>0)]
NAMES=['tokens','cap','accuracy','hit','first_right_given_hit','after_correct_per_answer','repeated8_fraction','repeated_equation_fraction','role_present','offtopic_cue_present','script_switch_present']
def results():
    allout={};pairs={}
    for ds in BUDGET:
        arm_rows={m:load(ds,m) for m in ARMS}
        N=30 if ds=='aime24' else 500;K=12 if ds=='aime24' else 2
        order=[(r['q'],r['j'],r['hash']) for r in arm_rows['B']]
        assert len(order)==N*K and len(set(order))==N*K
        right=np.reshape([r['correct'] for r in arm_rows['B']],(N,K)).sum(1)
        band=np.where(right<=(3 if K==12 else 0),'hard',np.where(right<=(8 if K==12 else 1),'mid','easy'))
        out={};mat={}
        for m,rs in arm_rows.items():
            assert [(r['q'],r['j'],r['hash']) for r in rs]==order
            assert all(r['official_budget'][-1]==bool(r['correct']) for r in rs)
            v=group(rs,ds)
            v['strata']={z+'_'+e:group([r for r in rs if strata(r)==z+'_'+e],ds) for z in ('cap','stop') for e in ('right','wrong','missing')}
            v['bands']={z:group([r for r in rs if band[r['q']]==z],ds) for z in ('hard','mid','easy')}
            v['sha256']=rs[0]['archive'];out[m]=v
            mat[m]=np.array([per_question(r) for r in rs]).reshape(N,K,-1).mean(1)
        pairs[ds]={}
        for z in ('all','hard','mid','easy'):
            keep=np.ones(N,bool) if z=='all' else band==z
            nq=int(keep.sum());rng=np.random.default_rng(20261124)
            w=np.vstack([np.ones(nq),rng.multinomial(nq,np.ones(nq)/nq,10000)])/nq
            boot={m:w@mat[m][keep] for m in ARMS}
            for a in boot.values():a[:,4]=np.divide(a[:,4],a[:,3],out=np.full(len(a),np.nan),where=a[:,3]>0)
            result={}
            for a,b in combinations(ARMS,2):
                d=boot[b]-boot[a];ci=np.nanquantile(d[1:],[.025,.975],axis=0)
                result[b+' minus '+a]={k:[float(v),float(l),float(u)] for k,v,l,u in zip(NAMES,d[0],*ci)}
            pairs[ds][z]=dict(questions=nq,contrasts=result)
        allout[ds]=out
    for name,obj in [('summary',allout),('paired',pairs)]:
        (HERE/(name+'.json')).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False))
if __name__=='__main__':results()
