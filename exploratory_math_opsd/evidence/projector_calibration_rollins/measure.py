"""Archived OOF projector calibration and on-policy vs fixed-prefix measurements."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'decision_projector'))
import diagnostic as d
from diagnostic import probe,load,ROOT,WORK,MODEL,sha
from common import save, projected_logits
from peft import PeftConfig,get_peft_model,get_peft_model_state_dict
from safetensors.torch import load_file
OUT=ROOT/'run/projector-calibration-rollins-v1'
STEPS=(25,50,75,100)
ENT=HERE.parent/'entropy_probe/results.json'
CUTS=json.loads(ENT.read_text())['entropy_edges_nats']
SCALES=[probe.SCALE,json.loads((HERE.parent/'decision_projector/results.json').read_text())['T0_scale']]


def archive(lane):
    p=WORK/f'rollouts-{lane}.pt'
    assert sha(p)==d.PRIOR['inputs']['archives'][p.name]
    return load(p)


def ckpt(step):
    p=ROOT/f'run/R/checkpoint-{step}'
    assert sha(p/'adapter_model.safetensors')==d.PRIOR['inputs']['adapters'][f'R@{step}']
    return p


def atomic(path,**values):
    tmp=path.with_suffix('.pending.npz')
    np.savez(tmp,**values)
    tmp.replace(path)


def generate(lane):
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    records=archive(lane)
    llm=LLM(model=str(MODEL),dtype='bfloat16',enable_lora=True,max_lora_rank=64,max_model_len=20000,
            max_num_seqs=64,tensor_parallel_size=1,gpu_memory_utilization=.84,seed=lane)
    for step in STEPS:
        path=OUT/f'roll-{lane}-{step}.pt'
        if path.exists():continue
        req=LoRARequest(f'r{step}',step,str(ckpt(step)))
        params=[SamplingParams(n=1,temperature=1.1,top_k=20,top_p=.95,
                repetition_penalty=1.,presence_penalty=0.,max_tokens=1024,seed=42+i) for i,*_ in records]
        prompts=[dict(prompt_token_ids=s) for _,s,_,_ in records]
        gens=llm.generate(prompts,params,lora_request=req)
        assert step in llm.llm_engine.list_loras() and len(gens)==len(records)
        rows={}
        for (idx,s,_,_),g in zip(records,gens):
            assert list(g.prompt_token_ids)==s and len(g.outputs)==1
            ans=g.outputs[0]
            assert 0<len(ans.token_ids)<=1024
            rows[idx]=(list(ans.token_ids),ans.finish_reason)
        save(path,rows)


@torch.inference_mode()
def calibrate(lane):
    base=probe.base_model();head=base.lm_head.weight
    fs=d.split()
    factors=[[d.factors(path,k,head) for k in range(4)] for path in (WORK,d.OUT)]
    coll=d.SelfDistillationDataCollator(AutoTokenizer.from_pretrained(MODEL),max_length=20000,
                reason_first=False,student_thinking=False,teacher_thinking=True,question_only_teacher=True)
    records=dict(d.stream())
    for idx,sp,tp,y in archive(lane):
        dst=OUT/f'cal-{idx}.npz'
        if dst.exists():continue
        with_path=ROOT/f'run/internalization-v1/teacher/{idx}.pt'
        ht=load(with_path,'cuda')
        prompt=coll([records[idx]])
        assert prompt['student_prompts'][0].tolist()==sp
        qp=prompt['teacher_prompts'][0].tolist()
        ids,mask=probe.inputs(qp,y)
        t0=probe.forward(base,ids,mask,len(qp))
        ent=np.load(ROOT/f'run/entropy-probe-v1/q-{idx}.npy',mmap_mode='r')[:,0]
        bins=np.searchsorted(CUTS,ent,side='right')
        group_totals=[];cover=[]
        for j,wdir in enumerate((WORK,d.OUT)):
            x=load(wdir/f'features-{idx}.pt','cuda');assert len(x['h'])==len(ent)
            pred=SCALES[j]*d.predict(x,factors[j][fs[idx]])
            parts=[];mass=[]
            for a in range(0,len(y),64):
                sl=slice(a,a+64)
                s=base.lm_head(x['h'][sl]).float()
                t=base.lm_head(ht[sl]).float() if j==0 else t0[sl].float()
                change=((t/1.1).softmax(1)-(s/1.1).softmax(1)).abs()
                v=x['valid'][sl];ii=x['ids'][sl].long()
                wt=change.gather(1,ii).double()*v
                label=x['d'][sl].double();error=(label-pred[sl]).square()
                stats=[]
                for w in (v.double()/v.sum(1,keepdim=True),wt):
                    stats.append(torch.stack([w.sum(1),(w*label).sum(1),(w*label.square()).sum(1),(w*error).sum(1)],dim=1))
                parts.append(torch.stack(stats,1).cpu().numpy())
                mass.append(change.sum(1,dtype=torch.float64).cpu().numpy())
            raw=np.concatenate(parts);total=np.concatenate(mass)
            group_totals.append(np.stack([raw[bins==k].sum(0) for k in range(10)]))
            cover.append(np.bincount(bins,weights=total,minlength=10))
        atomic(dst,stats=np.stack(group_totals),all_tv=np.stack(cover),counts=np.bincount(bins,minlength=10))


def actor_all():
    actor=d.checkpoint_actor()
    path=ckpt(100)
    actor.load_adapter(str(path),'step100',is_trainable=False)
    actual=get_peft_model_state_dict(actor,adapter_name='step100',save_embedding_layers=False)
    expected=load_file(str(path/'adapter_model.safetensors'))
    assert actual.keys()==expected.keys() and all(torch.equal(actual[k].cpu(),v) for k,v in expected.items())
    return actor.eval()


@torch.inference_mode()
def score(lane):
    base=probe.base_model();actor=actor_all();folds=d.split()
    states=[d.factors(WORK,k,base.lm_head.weight) for k in range(4)]
    for st in states:st['scale']=probe.SCALE
    eos=torch.tensor(d.PRIOR['eos_ids'],device='cuda')
    answers={k:load(OUT/f'roll-{lane}-{k}.pt') for k in STEPS}
    for idx,sp,tp,archived in archive(lane):
        dst=OUT/f'score-{idx}.npz'
        if dst.exists():continue
        totals=np.zeros((4,2,2,2,3));counts=np.zeros((4,2,2),dtype=int)
        length=np.zeros((4,2),dtype=int)
        for source in (0,*STEPS):
            y=archived if not source else answers[source][idx][0]
            si,sm=probe.inputs(sp,y);ti,tm=probe.inputs(tp,y)
            t=probe.forward(base,ti,tm,len(tp));s=probe.forward(base,si,sm,len(sp))
            if source:
                p=projected_logits(base,si,sm,len(sp)-1,states[folds[idx]])[0]
                masks=[]
                for j in range(0,len(y),64):
                    sl=slice(j,j+64);lt=(t[sl].float()/1.1).log_softmax(1)
                    u=(lt.exp()*(lt-(p[sl].float()/1.1).log_softmax(1))).sum(1,dtype=torch.float64)
                    masks.append(((-probe.odds(s[sl].double()/1.1,eos)).sigmoid()<.01)&(u>=d.CUT))
                content=torch.cat(masks);del p
            else:
                with np.load(d.PRE/f'{idx}.npz') as z:
                    content=torch.tensor((z['p']<.01)&(z['u']>=d.CUT),device='cuda')
            def summed(logits):
                chunks=[probe.behavior(logits[j:j+64],t[j:j+64],eos)[:,[3,2,4]] for j in range(0,len(y),64)]
                g=torch.cat(chunks)
                return torch.stack([g.sum(0),g[content].sum(0)]).cpu().numpy()
            b=summed(s)
            for k in ((source,) if source else STEPS):
                i=STEPS.index(k);side=int(source!=0)
                actor.set_adapter(f'step{k}');assert actor.active_adapter==f'step{k}'
                totals[i,side,0]=b
                totals[i,side,1]=summed(probe.forward(actor,si,sm,len(sp)))
                counts[i,side]=[len(y),int(content.sum())]
                length[i,side]=len(y)==1024
        atomic(dst,sums=totals,counts=counts,limit=length)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    {'calibrate':calibrate,'generate':generate,'score':score}[sys.argv[1]](int(sys.argv[2]))
