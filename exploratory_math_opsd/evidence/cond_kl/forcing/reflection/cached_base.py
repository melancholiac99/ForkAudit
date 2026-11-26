"""Preview B from archived response-position hidden states; the six-model table uses HF GPU logits."""
import torch
import numpy as np
from transformers import AutoTokenizer
from settings import *

def main():
    work=ROOT/'run/P-projector'
    head=torch.load(work/'head.pt',weights_only=True,map_location='cpu')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    ids=[tok.encode(s,add_special_tokens=False) for word in ('Wait','But','Hmm','Alternatively') for s in (word,' '+word)]
    assert all(len(x)==1 for x in ids)
    ids=[x[0] for x in ids]
    labels=read(RAW/'labels.json')
    sums=[]
    for idx in sorted(map(int,labels)):
        meta=labels[str(idx)]
        v=np.zeros((2,2,2))
        f=RAW/f'base-{idx}.npy'
        if f.exists():sums.append(np.load(f));continue
        if meta['correct'] is not None and meta['boundaries']:
            h=torch.load(work/f'features-{idx}.pt',weights_only=True,map_location='cpu')['h']
            positions=[p for p,_ in meta['boundaries']]
            g=int(meta['correct']);probs=[]
            for sl in torch.tensor(positions).split(96):
                x=torch.nn.functional.linear(h[sl],head).double()/1.1
                probs.append((x[:,ids].logsumexp(1)-x.logsumexp(1)).exp().numpy())
            probs=np.concatenate(probs)
            v[0,g]=[len(probs),probs.sum()]
            if meta['ended'] and meta['has_box']:v[1,g]=v[0,g]
        np.save(f,v);sums.append(v)
    data=np.stack(sums)
    from report import bootstrap
    def calc(w):
        sums=(w@data.reshape(len(data),-1)).reshape(-1,2,2,2)
        p=sums[:,:,:,1]/sums[:,:,:,0]
        return np.concatenate([p,np.log(p[:,:,0:1]/p[:,:,1:2])],axis=-1)
    point=calc(np.ones((1,len(data))))[0]
    draws=np.concatenate([calc(w) for w in bootstrap(len(data))])
    from report import interval
    save(HERE/'cached_B.json',{s:{k:interval(point[i,j],draws[:,i,j]) for j,k in enumerate(('wrong','correct','log_ratio'))} for i,s in enumerate(('all','completed_boxed'))})
if __name__=='__main__':main()
