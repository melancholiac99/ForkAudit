"""Shared fixed-support replay schedule and KL decomposition."""
from collections import Counter
from math import ceil

import numpy as np
import torch
from .c_targets import RUN as ORIGINAL, SEED
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, stable_seed


def balanced_batches(n,update):
    count=ceil(n/64)
    seed=stable_seed(SEED,'section21',update)
    order=np.random.default_rng(seed).permutation(n)
    return [part.tolist() for part in np.array_split(order,count)]


class SmokeBatch:
    def __init__(self):
        self.signatures=read(ORIGINAL/'smoke_acceptance.json')['staged_tensor_sha256']
        rows=[]
        for rank in range(4):
            for i,r in enumerate(self.shard(rank,'student')):
                rows.append((r,rank,i))
        self.items=sorted(rows,key=lambda x:(x[0]['goal'],x[0]['rep'],x[0]['step']))
        self.rows=[r for r,rank,i in self.items]
        self.keys=[(r['goal'],r['rep'],r['step']) for r in self.rows]
        assert len(set(self.keys))==len(self.keys)==282
        lengths=Counter()
        for r in self.rows:lengths[r['goal'],r['rep']]+=len(r['answer'])
        trajectories=Counter(g for g,rep in lengths)
        self.weights=[1/(len(trajectories)*trajectories[r['goal']]*lengths[r['goal'],r['rep']]) for r in self.rows]
        assert abs(sum(w*len(r['answer']) for w,r in zip(self.weights,self.rows))-1)<1e-12
        self.batches=balanced_batches(len(rows),1)
        self.probes=[next(j for j,(_,shard,i) in enumerate(self.items) if shard==k and i==0) for k in range(4)]

    def shard(self,rank,kind):
        key=f'batch_{rank}/{kind}.pt'
        p=ORIGINAL/key
        assert digest_file(p)==self.signatures[key]
        return torch.load(p,weights_only=True,map_location='cpu')

    def targets(self,arm):
        shards=[self.shard(k,arm) for k in range(4)]
        return [shards[k][i] for _,k,i in self.items]

    def local(self,subset,rank):
        # Assign whole records within a frozen minibatch, keeping both membership and weights.
        loads=[0]*4
        assigned=[[] for _ in range(4)]
        size=lambda j:len(self.rows[j]['prompt'])+len(self.rows[j]['answer'])
        for j in sorted(subset,key=lambda j:(-size(j),j)):
            k=min(range(4),key=lambda k:(loads[k],k))
            assigned[k].append(j);loads[k]+=size(j)
        return sorted(assigned[rank])


def components(a,b,target,mask):
    """Per-position natural-log quantities on the full valid candidate support."""
    l0=a.double().masked_fill(~mask,-torch.inf).log_softmax(-1).masked_fill(~mask,0)
    l1=b.double().masked_fill(~mask,-torch.inf).log_softmax(-1).masked_fill(~mask,0)
    p=l0.exp().masked_fill(~mask,0)
    q=target.double()
    q=q/q.sum(-1,keepdim=True)
    h=torch.special.xlogy(q,q).sum(-1)
    k0=h-(q*l0).sum(-1)
    k1=h-(q*l1).sum(-1)
    align=((q-p)*(l1-l0)).sum(-1)
    move=(p*(l0-l1)).sum(-1)
    return torch.stack((k0,k1,k0-k1,align,move),-1)


def summarize(batch,old,new,targets,indices):
    values=torch.zeros(5,dtype=torch.float64)
    mass=0.
    for i in indices:
        r=batch.rows[i];w=batch.weights[i]
        values+=w*components(old[i],new[i],targets[i],r['mask']).sum(0)
        mass+=w*len(r['answer'])
    # For a strict subset, condition the original evaluation weights on that subset.
    k0,k1,delta,align,move=(values/mass if mass else values).tolist()
    residual=delta-align+move
    assert abs(residual)<1e-11
    return dict(kl_before=k0,kl_after=k1,delta_kl=delta,alignment=align,
                movement=move,identity_residual=residual,global_weight_mass=mass)


def select_multiplier(rows):
    order=(1.,.5,.25,.125)
    for scale in order:
        if all(row[str(scale)]['alignment']>row['0.0']['error_envelope'] for row in rows.values()):
            return scale
    return None
