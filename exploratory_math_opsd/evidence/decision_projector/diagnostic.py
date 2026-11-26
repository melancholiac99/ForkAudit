"""OOF decision-site projector checks and the fixed content-subset R trajectory."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from peft import PeftConfig,get_peft_model,get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stop_decomposition'))
import probe
from common import ROOT, WORK, MODEL, save, projected_logits
from fit import split,fit
from launch import sha
from data import stream
from data_collator import SelfDistillationDataCollator

OUT=ROOT/'run/decision-projector-v1'
PRE=probe.OUT
PRIOR=json.loads((HERE.parent/'stop_decomposition/results.json').read_text())
CUT=PRIOR['content_u80']
STEPS=(25,50,75)
GROUPS=('candidates','eos_candidates','continue_log_odds')


def load(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=True)


def checkpoint_actor():
    source=ROOT/'run/R'
    actor=get_peft_model(probe.base_model(),PeftConfig.from_pretrained(source/'checkpoint-25'),adapter_name='step25')
    for k in STEPS:
        ck=source/f'checkpoint-{k}'
        assert sha(ck/'adapter_model.safetensors')==PRIOR['inputs']['adapters'][f'R@{k}']
        name=f'step{k}'
        actor.load_adapter(str(ck),name,is_trainable=False)
        actor.set_adapter(name)
        actual=get_peft_model_state_dict(actor,adapter_name=name,save_embedding_layers=False)
        expected=load_file(str(ck/'adapter_model.safetensors'))
        assert actual.keys()==expected.keys() and all(torch.equal(actual[key].cpu(),v) for key,v in expected.items())
    return actor.eval()


@torch.inference_mode()
def prepare(lane):
    base=probe.base_model()
    actor=checkpoint_actor()
    collator=SelfDistillationDataCollator(AutoTokenizer.from_pretrained(MODEL),max_length=20000,
        reason_first=False,student_thinking=False,teacher_thinking=True,question_only_teacher=True)
    records=dict(stream())
    archive=WORK/f'rollouts-{lane}.pt'
    assert sha(archive)==PRIOR['inputs']['archives'][archive.name]
    eos=torch.tensor(PRIOR['eos_ids'],device='cuda')
    for idx,sp,tp,y in load(archive):
        path=OUT/f'features-{idx}.pt'
        if path.exists():continue
        with np.load(PRE/f'{idx}.npz') as z:
            content=(z['p']<.01)&(z['u']>=CUT)
            archived=z['second'][content][:,[0,2]][:,:,[3,2,4]]
        coll=collator([records[idx]])
        assert coll['student_prompts'][0].tolist()==sp
        qp=coll['teacher_prompts'][0].tolist()
        qi,qm=probe.inputs(qp,y);si,sm=probe.inputs(sp,y)
        t0=probe.forward(base,qi,qm,len(qp))
        orig=load(WORK/f'features-{idx}.pt')
        hs=orig['h'].cuda()
        parts=[]
        for j in range(0,len(y),64):
            sl=slice(j,j+64)
            s=base.lm_head(hs[sl]).float()
            t=t0[sl].float()
            union=torch.cat((s.topk(32).indices,t.topk(32).indices),1).sort(1).values
            valid=torch.ones_like(union,dtype=torch.bool);valid[:,1:]=union[:,1:]!=union[:,:-1]
            a=s.gather(1,union)
            d=t.gather(1,union)-a
            d=d-(a.masked_fill(~valid,-torch.inf).softmax(1)*d).sum(1,keepdim=True)
            parts.append((union.int().cpu(),valid.cpu(),d.cpu()))
        totals=[archived[:,0].sum(0)]
        if content.any():
            ti,tm=probe.inputs(tp,y)
            mask=torch.as_tensor(content,device='cuda')
            teacher=probe.forward(base,ti,tm,len(tp))[mask]
            for step in STEPS:
                actor.set_adapter(f'step{step}')
                state=probe.forward(actor,si,sm,len(sp))[mask]
                values=torch.cat([probe.behavior(state[j:j+32],teacher[j:j+32],eos)[:,[3,2,4]]
                                  for j in range(0,len(teacher),32)])
                totals.append(values.sum(0).cpu().numpy())
        else:totals.extend(np.zeros((3,3)))
        totals.append(archived[:,1].sum(0))
        np.save(OUT/f'traj-{idx}.npy',np.array(totals))
        save(path,dict(h=orig['h'],ids=torch.cat([p[0] for p in parts]),
                       valid=torch.cat([p[1] for p in parts]),d=torch.cat([p[2] for p in parts])))


def train_fold(k):
    if (OUT/f'fit-{k}.pt').exists():return
    cache={i:load(OUT/f'features-{i}.pt','cuda') for i,f in split().items() if f!=k}
    fit(k,OUT,cache.__getitem__)


def factors(work,k,head):
    st=load(work/f'fit-{k}.pt','cuda')
    st['vocab']=F.linear(head.float(),st['B'])
    return st


def predict(x,st):
    return torch.einsum('pr,pcr->pc',F.linear(x['h'].float(),st['A']),st['vocab'][x['ids'].long()])


@torch.inference_mode()
def calibrate(k):
    st=factors(OUT,k,load(WORK/'head.pt','cuda'))
    xx=xy=0.
    for idx,fold in split().items():
        if fold!=k:continue
        x=load(OUT/f'features-{idx}.pt','cuda')
        p=predict(x,st);w=x['valid']/x['valid'].sum(1,keepdim=True)
        xx+=(w*p.square()).sum(dtype=torch.float64).item()
        xy+=(w*p*x['d']).sum(dtype=torch.float64).item()
    (OUT/f'cal-{k}.json').write_text(json.dumps([xx,xy]))


def moments(y,p,w):
    return torch.stack([w.sum(),(w*y).sum(),(w*y.square()).sum(),(w*(y-p).square()).sum(),
                        (w*(y.sign()==p.sign())).sum(),(w*(y.sign()*p.sign()<0)).sum(),
                        (w*(y==0)).sum(),(w*(p==0)).sum(),(w*(y>0)).sum(),(w*(p>0)).sum(),(w*p).sum()]).cpu().numpy()


@torch.inference_mode()
def measure(lane):
    base=probe.base_model()
    head=base.lm_head.weight
    xx,xy=np.sum([json.loads((OUT/f'cal-{k}.json').read_text()) for k in range(4)],0)
    scales=[probe.SCALE,float(np.clip(xy/xx,0,1))]
    fs=split()
    params=[[factors(work,k,head) for k in range(4)] for work in (WORK,OUT)]
    eos=torch.tensor(PRIOR['eos_ids'],device='cuda')
    for idx,sp,tp,y in load(WORK/f'rollouts-{lane}.pt'):
        dest=OUT/f'stats-{idx}.npy'
        if dest.exists():continue
        with np.load(PRE/f'{idx}.npz') as z:
            mask=z['p']>=.05
            actual=z['first'][mask,:,0]
        totals=np.zeros((2,3,11))
        if mask.any():
            mask=torch.as_tensor(mask,device='cuda')
            si,sm=probe.inputs(sp,y)
            for j,work in enumerate((WORK,OUT)):
                st=params[j][fs[idx]]
                st['scale']=scales[j]
                x={k:v[mask] for k,v in load(work/f'features-{idx}.pt','cuda').items()}
                d=x['d'].double(); p=(st['scale']*predict(x,st)).double()
                valid=x['valid']; ev=valid & (x['ids'][:,:,None]==eos).any(2)
                for i,w in enumerate((valid,ev)):
                    weights=w.double()/w.sum(1,keepdim=True).clamp_min(1)
                    totals[j,i]=moments(d,p,weights)
                if j==0:
                    dy,dp=actual[:,0],actual[:,1]
                else:
                    s=probe.forward(base,si,sm,len(sp))[mask]
                    q=projected_logits(base,si,sm,len(sp)-1,st)[0,mask]
                    dp=(probe.odds(q.double()/1.1,eos)-probe.odds(s.double()/1.1,eos)).cpu().numpy()
                    dy=actual[:,3]
                dy,dp=torch.from_numpy(dy),torch.from_numpy(dp)
                totals[j,2]=moments(dy,dp,torch.ones_like(dy))
        np.save(dest,totals)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'fit':train_fold,'calibrate':calibrate,'measure':measure}[sys.argv[1]](int(sys.argv[2]))
