"""Descriptive cross-arm comparisons on the same four fixed probe records."""
from itertools import combinations
import math
import torch
from .d_data import ARMS,folder
from .s21_data import SmokeBatch


def pair_readout(batch,initial,left,right):
    totals=torch.zeros(4,dtype=torch.float64)
    mass=0.
    for i in batch.probes:
        mask=batch.rows[i]['mask']
        def logp(logits):
            return logits[i].double().masked_fill(~mask,-torch.inf).log_softmax(-1)
        base,a,b=map(logp,(initial,left,right))
        mix=torch.logaddexp(a,b)-math.log(2.)
        js=.5*((a.exp()*(a-mix)).masked_fill(~mask,0)+(b.exp()*(b-mix)).masked_fill(~mask,0)).sum(-1)
        p0=base.exp()
        da=(a-base).masked_fill(~mask,0)
        db=(b-base).masked_fill(~mask,0)
        da-= (p0*da).sum(-1,keepdim=True)
        db-= (p0*db).sum(-1,keepdim=True)
        w=batch.weights[i]
        totals+=w*torch.stack((js.sum(),(p0*da*db).sum(),(p0*da.square()).sum(),(p0*db.square()).sum()))
        mass+=w*len(js)
    js,dot,aa,bb=(totals/mass).tolist()
    cosine=dot/math.sqrt(aa*bb) if aa>0 and bb>0 else None
    assert -1e-12<=js<=math.log(2)+1e-12
    assert cosine is None or abs(cosine)<=1+1e-12
    return dict(js_nats=max(0.,js),baseline_weighted_direction_cosine=cosine)


def pairs_through(last):
    batch=SmokeBatch()
    baselines={a:torch.load(folder(1,a)/'probes.pt',weights_only=True)['before'] for a in ARMS}
    initial=baselines['R']
    assert all(torch.equal(base[i],initial[i]) for base in baselines.values() for i in batch.probes)
    result=[]
    for k in range(last+1):
        states=baselines if k==0 else {a:torch.load(folder(k,a)/'probes.pt',weights_only=True)['after'] for a in ARMS}
        for a,b in combinations(ARMS,2):
            result.append(dict(update=k,left=a,right=b,**pair_readout(batch,initial,states[a],states[b])))
    return result
