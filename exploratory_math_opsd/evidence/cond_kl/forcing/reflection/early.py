"""Use AIME24 lanes only after their answer file is sealed. Yield before the next GPU cell."""
import fcntl
import os
import time
import numpy as np
import torch
from peft import PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer
from settings import *
sys.path.insert(0,str(HERE.parents[1]))
from measure import old,checkpoint
from launch import sha

@torch.inference_mode()
def run(lane):
    folder=ROOT/'run/evaluation/R-cond@100/aime24'
    done=folder/'1001.receipt.json'
    while not (folder/'1001-lanes'/f'lane{lane}.json').exists():
        if done.exists():return
        time.sleep(2)
    with (RAW/f'gpu-{lane}.lock').open('a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX)
        if done.exists():return
        labels=read(RAW/'labels.json')
        tok=AutoTokenizer.from_pretrained(old.MODEL)
        forms=[f for w in ('Wait','But','Alternatively','Hmm') for f in (w,' '+w)]
        ids=sorted({tok.encode(f,add_special_tokens=False)[0] for f in forms})
        wait=[tok.encode(f,add_special_tokens=False)[0] for f in forms[:2]]
        actor=None
        for name in MODELS[1:]:
            arm=name.split('@')[0]
            if arm in ('R-cond','R-noclip'):ck=checkpoint(arm,100)
            else:
                ck=ROOT/'run'/arm/'checkpoint-100'
                assert sha(ck/'adapter_model.safetensors')==old.d.PRIOR['inputs']['adapters'][name]
            if actor is None:actor=PeftModel.from_pretrained(old.probe.base_model(),str(ck),adapter_name=arm)
            else:actor.load_adapter(str(ck),arm,is_trainable=False)
            state=get_peft_model_state_dict(actor,adapter_name=arm,save_embedding_layers=False)
            src=load_file(str(ck/'adapter_model.safetensors'))
            assert state.keys()==src.keys() and all(torch.equal(state[k].cpu(),v) for k,v in src.items())
        actor.eval()
        records=[]
        for k in range(4):records+=old.archive((lane+k)%4)
        for idx,sp,_,y in records:
            if done.exists():return
            dest=RAW/f'score-{idx}.npz'
            if dest.exists():continue
            with (RAW/f'site-lock-{idx}').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if dest.exists():continue
                row=labels[str(idx)];sites=row['sites'];where=np.array([p for p,g in row['boundaries']],dtype=int)
                probs=np.empty((len(where),6,len(ids)))
                first=np.empty((len(sites),2,4),dtype=int)
                si,sm=old.probe.inputs(sp,y)
                for m,name in enumerate(MODELS):
                    if not len(where):break
                    if name=='B':
                        with actor.disable_adapter():x=old.probe.forward(actor,si,sm,len(sp))
                    else:
                        actor.set_adapter(name.split('@')[0]);x=old.probe.forward(actor,si,sm,len(sp))
                    for a in range(0,len(where),32):
                        sl=where[a:a+32];probs[a:a+len(sl),m]=(x[sl].double()/1.1).softmax(-1)[:,ids].cpu().numpy()
                    if name=='R@100':
                        for j,pos in enumerate(sites):
                            p=(x[pos].double()/1.1).softmax(-1).cpu().numpy()
                            pw=p[wait]/p[wait].sum();p[wait]=0;p/=p.sum()
                            rng=np.random.default_rng([SEED,idx,pos])
                            first[j,0]=rng.choice(wait,p=pw,size=4)
                            first[j,1]=rng.choice(len(p),p=p,size=4)
                    del x
                old.atomic(dest,p=probs,first=first,sites=np.array(sites,dtype=int))
if __name__=='__main__':run(int(sys.argv[1]))
