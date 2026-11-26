"""Reuse the four full-probe probability archives and score the missing adapters."""
import numpy as np
import torch
from peft import PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
from settings import *
sys.path.insert(0,str(HERE.parents[1]))
from measure import old
from launch import sha

@torch.inference_mode()
def run(lane):
    labels=read(RAW/'labels.json')
    info=read(HERE.parent/'reflection_ids.json')
    ids=info['included_ids'];wait=[info['encodings'][k][0] for k in ('Wait',' Wait')]
    assert len(ids)==8 and len(set(wait))==2
    actor=None
    for arm in ('T0','P','R'):
        path=ROOT/'run'/arm/'checkpoint-100'
        assert sha(path/'adapter_model.safetensors')==old.d.PRIOR['inputs']['adapters'][arm+'@100']
        if actor is None:actor=PeftModel.from_pretrained(old.probe.base_model(),str(path),adapter_name=arm)
        else:actor.load_adapter(str(path),arm,is_trainable=False)
        state=get_peft_model_state_dict(actor,adapter_name=arm,save_embedding_layers=False)
        src=load_file(str(path/'adapter_model.safetensors'))
        assert state.keys()==src.keys() and all(torch.equal(state[k].cpu(),v) for k,v in src.items())
    actor.eval()
    for idx,sp,_,y in old.archive(lane):
        dest=RAW/f'score-{idx}.npz'
        if dest.exists():continue
        row=labels[str(idx)];boundary=np.array([p for p,g in row['boundaries']],dtype=int)
        sites=row['sites'];probs=np.empty((len(boundary),len(MODELS),len(ids)))
        with np.load(FORCING_RAW/'secondary'/f'{idx}.npz') as z:
            saved=z['models'][boundary]
        for j,k in enumerate((0,2,3,4)):probs[:,k]=saved[:,j,3:]
        first=np.zeros((len(sites),2,4),dtype=int)
        if len(boundary):
            inp,mask=old.probe.inputs(sp,y)
            for arm in ('T0','P','R'):
                if arm=='R' and not sites:continue
                actor.set_adapter(arm);assert actor.active_adapter==arm
                x=old.probe.forward(actor,inp,mask,len(sp))
                if arm!='R':
                    for start in range(0,len(boundary),32):
                        ix=boundary[start:start+32]
                        probs[start:start+len(ix),MODELS.index(arm+'@100')]=(x[ix].double()/1.1).softmax(-1)[:,ids].cpu().numpy()
                else:
                    for j,pos in enumerate(sites):
                        p=(x[pos].double()/1.1).softmax(-1).cpu().numpy()
                        np.testing.assert_allclose(p[ids],probs[boundary==pos,2][0],rtol=2e-5,atol=1e-10)
                        pw=p[wait]/p[wait].sum();p[wait]=0;p/=p.sum()
                        rng=np.random.default_rng([SEED,idx,pos])
                        first[j,0]=rng.choice(wait,p=pw,size=4)
                        first[j,1]=rng.choice(len(p),p=p,size=4)
                del x
        old.atomic(dest,p=probs,first=first,sites=np.array(sites,dtype=int))
if __name__=='__main__':run(int(sys.argv[1]))
