"""Compare saved B probabilities with three adjacent cached response hidden states."""
from settings import *
import torch
import numpy as np
from transformers import AutoTokenizer
labels=read(RAW/'labels.json');tids=read(HERE/'tokens.json')['included_ids']
work=ROOT/'run/P-projector'
head=torch.load(work/'head.pt',map_location='cpu',weights_only=True)
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
ids=sorted(map(int,labels))
qids=np.random.default_rng(SEED).choice(ids,32,replace=False)
acc=np.zeros(3);n=0
for q in qids:
    row=labels[str(q)];bounds=[x[0] for x in row['boundaries']]
    which=[i for i,p in enumerate(bounds) if p+1<row['length']][:12]
    if not which:continue
    ix=np.array(bounds)[which]
    h=torch.load(work/f'features-{q}.pt',weights_only=True)['h']
    with np.load(RAW/f'score-{q}.npz') as z: ref=z['p'][which,0]
    for j,shift in enumerate((-1,0,1)):
        x=torch.nn.functional.linear(h[ix+shift],head).double()/1.1
        p=torch.softmax(x,dim=1)[:,tids].numpy()
        acc[j]+=np.abs(p-ref).sum()
    n+=len(ix)*len(tids)
print(n,acc/n)
save(HERE/'offset_check.json',dict(questions_sampled=len(qids),token_probability_comparisons=n,mean_absolute_error=dict(zip(('previous','registered','next'),(acc/n).tolist()))))
