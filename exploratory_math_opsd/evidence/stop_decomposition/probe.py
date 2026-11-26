"""Score fixed archived prefixes; run four lanes with one visible GPU per lane."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
sys.path[:0]=[str(EXP/'pmath'),str(EXP)]
from common import ROOT, WORK, MODEL, projected_logits
from data import stream
from fit import split
from launch import sha
from rres_teacher import residual_logits
sys.path.insert(0,str(ROOT/'source'))
from data_collator import SelfDistillationDataCollator

OUT=ROOT/'run/stop-decomposition-v1'
TARGETS=('R','P','Rres','T0')
MODELS=('B','P','R','Rres','T0')
PART1=('delta','M','J_C','J_S','J','delta_U','outside','delta_on_U_positions')
PART2=('p_stop','log_continue_stop','top1_agreement','G','entropy')
SCALE=.96874645


def indices(n):
    ordered=np.array(sorted(split()))
    return ordered[np.arange(n)*len(ordered)//n].tolist()


def base_model():
    return AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2').cuda().eval()


def adapters(manifest):
    base=base_model()
    first=ROOT/'run/P/checkpoint-100'
    actor=get_peft_model(base,PeftConfig.from_pretrained(first),adapter_name='P')
    for name in MODELS[1:]:
        path=ROOT/f'run/{name}/checkpoint-100'
        file=path/'adapter_model.safetensors'
        assert sha(file)==manifest['adapters'][name+'@100']
        # Same PEFT load_adapter used by Transformers Trainer for checkpoint resume.
        actor.load_adapter(str(path),name,is_trainable=False)
        actor.set_adapter(name)
        original=load_file(str(file))
        actual=get_peft_model_state_dict(actor,adapter_name=name,save_embedding_layers=False)
        assert actual.keys()==original.keys()
        assert all(torch.equal(actual[k].cpu(),v) for k,v in original.items()),name
    return actor.eval()


def inputs(prompt,y):
    ids=torch.tensor([prompt+y],device='cuda')
    return ids,torch.ones_like(ids)


def forward(model,ids,mask,start):
    return model(input_ids=ids,attention_mask=mask,use_cache=False).logits[0,start-1:-1]


def odds(logits,eos):
    stop=logits[:,eos]
    cont=logits.clone()
    cont[:,eos]=-torch.inf
    return cont.logsumexp(1)-stop.logsumexp(1)


def decomposition(s,t,eos,union,valid):
    a=s.double()/1.1
    b=t.double()/1.1
    eps=b-a
    direct=odds(b.log_softmax(1),eos)-odds(a.log_softmax(1),eos)
    vals=[]
    for stop in (False,True):
        if stop:
            lp=a[:,eos].log_softmax(1)
            change=eps[:,eos]
        else:
            z=a.clone()
            z[:,eos]=-torch.inf
            lp=z.log_softmax(1)
            change=eps
        m=(lp.exp()*change).sum(1)
        l=(lp+change).logsumexp(1)
        vals.append((l,m,l-m))
    (lc,mc,jc),(ls,ms,js)=vals
    delta=lc-ls
    stopmask=(union[:,:,None]==eos).any(2)&valid
    cmask=valid&~stopmask
    def limited(logits):
        z=logits.gather(1,union)
        return z.masked_fill(~cmask,-torch.inf).logsumexp(1)-z.masked_fill(~stopmask,-torch.inf).logsumexp(1)
    has=stopmask.any(1)&cmask.any(1)
    du=torch.where(has,limited(b)-limited(a),torch.nan)
    full=delta.masked_fill(~has,torch.nan)
    return torch.stack([delta,mc-ms,jc,js,jc-js,du,full-du,full],1),(direct-delta).abs().max()


def behavior(x,t,eos):
    xl=x.float()
    ratio=odds(xl.double(),eos)
    lp=(xl/1.1).log_softmax(1)
    lt=(t.float()/1.1).log_softmax(1)
    return torch.stack([(-ratio).sigmoid(),ratio,(xl.argmax(1)==t.argmax(1)).double(),
                        (lt.exp()*(lt-lp)).sum(1,dtype=torch.float64),
                        -(lp.exp()*lp).sum(1,dtype=torch.float64)],1)


@torch.inference_mode()
def run(lane,n):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE.parent/'internalization/inputs.json').read_text())
    selected=indices(n)[lane::4]
    selected=[i for i in selected if not (OUT/f'{i}.npz').exists()]
    if not selected:return
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    eos=torch.tensor(json.loads((MODEL/'generation_config.json').read_text())['eos_token_id'],device='cuda')
    assert len(eos)==2
    collator=SelfDistillationDataCollator(tokenizer,max_length=20000,reason_first=False,
               student_thinking=False,teacher_thinking=True,question_only_teacher=True)
    questions=dict(stream())
    records={}
    for j in range(4):
        f=WORK/f'rollouts-{j}.pt'
        assert sha(f)==manifest['archives'][f.name]
        records.update({i:(s,t,y) for i,s,t,y in torch.load(f,weights_only=True)})
    base=base_model()
    actor=adapters(manifest)
    folds=split()
    states=[torch.load(WORK/f'fit-{j}.pt',map_location='cuda',weights_only=True) for j in range(4)]
    for st in states:st['scale']=SCALE
    for idx in selected:
        sp,tp,y=records[idx]
        batch=collator([questions[idx]])
        assert batch['student_prompts'][0].tolist()==sp
        qp=batch['teacher_prompts'][0].tolist()
        si,sm=inputs(sp,y);ti,tm=inputs(tp,y);qi,qm=inputs(qp,y)
        s=forward(base,si,sm,len(sp));t=forward(base,ti,tm,len(tp))
        p=projected_logits(base,si,sm,len(sp)-1,states[folds[idx]])[0]
        rr=residual_logits(base,dict(student_input_ids=si,student_attention_mask=sm,
                   teacher_input_ids=ti,teacher_attention_mask=tm),len(sp),len(tp),states[folds[idx]])[0]
        t0=forward(base,qi,qm,len(qp))
        first=[];mask=[];gu=[];errors=[]
        for j in range(0,len(y),32):
            sl=slice(j,j+32)
            ss=s[sl];tt=t[sl];pp=p[sl]
            U=torch.cat([ss.topk(32).indices,tt.topk(32).indices],1).sort(1).values
            valid=torch.ones_like(U,dtype=torch.bool);valid[:,1:]=U[:,1:]!=U[:,:-1]
            group=[]
            for target in (tt,pp,rr[sl],t0[sl]):
                v,e=decomposition(ss,target,eos,U,valid)
                group.append(v);errors.append(e)
            first.append(torch.stack(group,1).cpu())
            mask.append((-odds(ss.double()/1.1,eos)).sigmoid().cpu())
            lp=(pp.float()/1.1).log_softmax(1);lt=(tt.float()/1.1).log_softmax(1)
            gu.append((lt.exp()*(lt-lp)).sum(1,dtype=torch.float64).cpu())
        del p,rr,t0
        second=[]
        for name in MODELS:
            if name=='B': logits=s
            else:
                actor.set_adapter(name)
                assert actor.active_adapter==name
                logits=forward(actor,si,sm,len(sp))
            second.append(torch.cat([behavior(logits[j:j+32],t[j:j+32],eos).cpu()
                                     for j in range(0,len(y),32)]))
        stop=np.isin(y,eos.cpu().numpy())
        assert not stop[:-1].any()
        tmp=OUT/f'{idx}.pending.npz'
        np.savez(tmp,first=torch.cat(first).numpy(),second=torch.stack(second,1).numpy(),
                 p=torch.cat(mask).numpy(),u=torch.cat(gu).numpy(),eos=stop,
                 error=torch.stack(errors).reshape(-1,4).amax(0).cpu().numpy())
        tmp.replace(OUT/f'{idx}.npz')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('lane',type=int);parser.add_argument('n',type=int)
    a=parser.parse_args();run(a.lane,a.n)
