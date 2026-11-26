"""Fixed and own-prefix measurements with the archived roll-in generator and masks."""
import sys
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file
from peft import get_peft_model_state_dict

HERE=Path(__file__).resolve().parent
import importlib.util
spec=importlib.util.spec_from_file_location('archive_rollins', HERE.parent/'projector_calibration_rollins/measure.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from preflight import controls
from common import projected_logits
from launch import read, sha

ROOT=old.ROOT
ARMS=('R-noclip','R-cond')
STEPS=old.STEPS
SETS=('all','content','stop05','actual_eos')
METRICS=('p_stop','log_continue_stop','top1','KL','entropy','top1_outside_S')
WORK=ROOT/'run/cond-kl'


def checkpoint(arm,step):
    path=ROOT/'run'/arm/f'checkpoint-{step}'
    assert sha(path/'adapter_model.safetensors')==read(ROOT/'run'/arm/'complete.json')['adapters'][str(step)]
    return path


def generate(arm,lane):
    old.OUT=WORK/arm
    old.OUT.mkdir(exist_ok=True)
    old.ckpt=lambda k:checkpoint(arm,k)
    old.generate(lane)


@torch.inference_mode()
def score(stage,arm,lane):
    out=WORK/arm
    out.mkdir(exist_ok=True)
    base=old.probe.base_model()
    actor=None
    for k in STEPS:
        ck=checkpoint(arm,k)
        key=f'step{k}'
        if actor is None:
            actor=PeftModel.from_pretrained(old.probe.base_model(),str(ck),adapter_name=key)
        else:
            actor.load_adapter(str(ck),key,is_trainable=False)
        expected=load_file(str(ck/'adapter_model.safetensors'))
        actual=get_peft_model_state_dict(actor,adapter_name=key,save_embedding_layers=False)
        assert expected.keys()==actual.keys() and all(torch.equal(actual[n].cpu(),v) for n,v in expected.items())
    actor.eval()
    ids,eos=controls()
    eos=torch.tensor(eos,device='cuda')
    fs=old.d.split()
    own=stage=='own'
    if own:
        states=[old.d.factors(old.WORK,k,base.lm_head.weight) for k in range(4)]
        for state in states: state['scale']=old.probe.SCALE
        answers={k:old.load(out/f'roll-{lane}-{k}.pt') for k in STEPS}
    for idx,sp,tp,archive_y in old.archive(lane):
        dest=out/f'{stage}-{idx}.npz'
        if dest.exists():continue
        sums=np.zeros((4,len(SETS),2,len(METRICS)))
        counts=np.zeros((4,len(SETS)),dtype=int)
        for source in (STEPS if own else [0]):
            y=answers[source][idx][0] if own else archive_y
            si,sm=old.probe.inputs(sp,y)
            s=old.probe.forward(base,si,sm,len(sp))
            t=old.probe.forward(base,*old.probe.inputs(tp,y),len(tp))
            if own:
                p=projected_logits(base,si,sm,len(sp)-1,states[fs[idx]])[0]
                parts=[]
                for start in range(0,len(y),64):
                    sl=slice(start,start+64)
                    lt=(t[sl].float()/1.1).log_softmax(-1)
                    lp=(p[sl].float()/1.1).log_softmax(-1)
                    parts.append((lt.exp()*(lt-lp)).sum(-1,dtype=torch.double))
                u=torch.cat(parts)
                stop=(-old.probe.odds(s.double()/1.1,eos)).sigmoid()
                ended=torch.isin(torch.as_tensor(y,device='cuda'),eos)
                del p
            else:
                with np.load(old.d.PRE/f'{idx}.npz') as z:
                    stop,u,ended=[torch.as_tensor(z[n],device='cuda') for n in ('p','u','eos')]
            masks=torch.stack([torch.ones_like(ended), (stop<.01)&(u>=old.d.CUT),stop>=.05,ended]).double()
            def aggregate(x):
                parts=[]
                for start in range(0,len(y),32):
                    sl=slice(start,start+32)
                    a,b=x[sl],t[sl]
                    v=old.probe.behavior(a,b,eos)
                    a,b=a.clone(),b.clone()
                    a[:,ids]=-torch.inf
                    b[:,ids]=-torch.inf
                    parts.append(torch.cat((v,(a.argmax(1)==b.argmax(1))[:,None]),1))
                return (masks @ torch.cat(parts)).cpu().numpy()
            b=aggregate(s)
            for k in ([source] if own else STEPS):
                actor.set_adapter(f'step{k}')
                assert actor.active_adapter==f'step{k}'
                i=STEPS.index(k)
                counts[i]=masks.sum(1).cpu().numpy()
                sums[i,:,0]=b
                sums[i,:,1]=aggregate(old.probe.forward(actor,si,sm,len(sp)))
        old.atomic(dest,sums=sums,counts=counts)


if __name__=='__main__':
    stage,arm,lane=sys.argv[1:]
    if stage=='generate':
        generate(arm,int(lane))
    else:
        score(stage,arm,int(lane))
