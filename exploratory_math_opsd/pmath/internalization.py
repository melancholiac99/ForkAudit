"""Frozen-model, held-out internalization measurements on archived math prefixes."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from common import ROOT, MODEL, WORK, save, json_save
from fit import split, fit
from token_kl_report import classify, GROUPS

STATES = ('B','R@25','R@50','R@75','R@100','T0@100','P@100','Rres@100')
OUT = ROOT/'run/internalization-v1'
EVIDENCE = Path(__file__).resolve().parents[1]/'evidence/internalization'
TEMP = 1.1


def load(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)


def model_for(state):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2').cuda().eval()
    if state != 'B':
        arm, step = state.split('@')
        path = ROOT/'run'/arm/f'checkpoint-{step}'
        assert json.loads((path/'trainer_state.json').read_text())['global_step'] == int(step)
        model = PeftModel.from_pretrained(model, path, is_trainable=False).eval()
        assert model.active_adapter == 'default'
        model = model.get_base_model()
        assert any('lora_A.default' in n for n,_ in model.named_parameters())
    return model


def prefix_hidden(model, prompt, answer):
    ids = torch.tensor([prompt+answer], device='cuda')
    return model.model(input_ids=ids,use_cache=False).last_hidden_state[0,len(prompt)-1:-1].contiguous()


@torch.inference_mode()
def features(state,lane):
    folder = OUT/state
    teacher = OUT/'teacher'
    teacher.mkdir(parents=True,exist_ok=True)
    model = model_for(state)
    head = model.lm_head
    assert torch.equal(head.weight.cpu(),load(WORK/'head.pt'))
    for idx,prompt,priv,answer in load(WORK/f'rollouts-{lane}.pt'):
        path = folder/f'features-{idx}.pt'
        if path.exists(): continue
        tp = teacher/f'{idx}.pt'
        if not tp.exists():
            assert state == 'B'
            save(tp,prefix_hidden(model,priv,answer).cpu())
        ht = load(tp,'cuda')
        h = prefix_hidden(model,prompt,answer)
        assert ht.shape == h.shape and len(h)==len(answer)
        all_ids, masks, labels = [],[],[]
        for j in range(0,len(h),64):
            t,s = head(ht[j:j+64]).float(),head(h[j:j+64]).float()
            ids = torch.cat([t.topk(32).indices,s.topk(32).indices],1).sort().values
            valid = torch.ones_like(ids,dtype=torch.bool)
            valid[:,1:] = ids[:,1:] != ids[:,:-1]
            a = s.gather(1,ids)
            d = t.gather(1,ids)-a
            d -= (d*a.masked_fill(~valid,-torch.inf).softmax(1)).sum(1,keepdim=True)
            all_ids.append(ids.int());masks.append(valid);labels.append(d)
        x = dict(h=h.cpu(), ids=torch.cat(all_ids).cpu(), valid=torch.cat(masks).cpu(), d=torch.cat(labels).cpu())
        if state == 'B':
            ref = load(WORK/f'features-{idx}.pt')
            # Restore the independent baseline fold fits only for the exact original features.
            for key in x:
                if not torch.equal(x[key],ref[key]):
                    x['baseline_exact'] = False
                    break
        save(path,x)


def factors(state,fold):
    folder=OUT/state
    if (folder/f'fit-{fold}.pt').exists():return
    ids=[i for i,k in split().items() if k!=fold]
    cache={i:load(folder/f'features-{i}.pt','cuda') for i in ids}
    dest=folder/f'fit-{fold}.pt'
    if state=='B' and all(x.get('baseline_exact',True) for x in cache.values()):
        save(dest,load(WORK/f'fit-{fold}.pt'))
    else:
        fit(fold,folder,cache.__getitem__)


@torch.inference_mode()
def calibrate(state,fold):
    folder=OUT/state
    params=load(folder/f'fit-{fold}.pt','cuda')
    head=load(WORK/'head.pt','cuda')
    vocab=F.linear(head.float(),params['B'])
    records={}
    for idx,k in split().items():
        if k!=fold: continue
        x=load(folder/f'features-{idx}.pt','cuda')
        p=torch.einsum('pr,pcr->pc',F.linear(x['h'].float(),params['A']),vocab[x['ids'].long()])
        w=x['valid']/x['valid'].sum(1,keepdim=True)
        records[idx]=[(w*p.square()).sum(dtype=torch.float64).item(),(w*p*x['d']).sum(dtype=torch.float64).item()]
    json_save(folder/f'cal-{fold}.json',records)


def calibration(folder):
    stats={}
    for k in range(4):stats.update(json.loads((folder/f'cal-{k}.json').read_text()))
    assert len(stats)==1500
    xx,xy=np.sum(list(stats.values()),axis=0)
    return stats,xy/xx


@torch.inference_mode()
def measure(state,lane):
    folder=OUT/state
    head=load(WORK/'head.pt','cuda')
    _,slope=calibration(folder)
    scale=np.clip(slope,0,1).item()
    params=[load(folder/f'fit-{k}.pt','cuda') for k in range(4)]
    vocab=[F.linear(head.float(),p['B']) for p in params]
    group_file=OUT/f'vocabulary-groups-{lane}.pt'
    if not group_file.exists():
        tok=AutoTokenizer.from_pretrained(MODEL)
        size=len(tok)
        labels=[classify(tok.decode([i],clean_up_tokenization_spaces=False)) if i<size else 4
                for i in range(head.shape[0])]
        save(group_file,torch.tensor(labels))
    groups=load(group_file,'cuda')
    folds=split()
    for idx,_,_,answer in load(WORK/f'rollouts-{lane}.pt'):
        dest=folder/f'sums-{idx}.json'
        if dest.exists():continue
        h=load(folder/f'features-{idx}.pt','cuda')['h']
        ht=load(OUT/'teacher'/f'{idx}.pt','cuda')
        k=folds[idx]; by_id=torch.zeros((2,len(head)),dtype=torch.float64,device='cuda')
        for j in range(0,len(h),64):
            part=h[j:j+64]
            s=F.linear(part,head).float()
            t=F.log_softmax(F.linear(ht[j:j+64],head).float()/TEMP,dim=1)
            q=t.exp()
            top=s.topk(64).indices
            corr=torch.einsum('pr,pcr->pc',F.linear(part.float(),params[k]['A']),vocab[k][top])
            p=s.scatter_add(1,top,scale*corr)
            for a,student in enumerate((s,p)):
                by_id[a]+=(q*(t-F.log_softmax(student/TEMP,dim=1))).sum(0,dtype=torch.float64)
        values=torch.stack([by_id[:,groups==g].sum(1) for g in range(len(GROUPS))],dim=1)
        assert torch.isfinite(values).all() and (values.sum(1)>=-1e-6).all()
        json_save(dest,dict(n=len(answer),groups=values.cpu().tolist()))


def report(state):
    folder=OUT/state
    stats,_=calibration(folder)
    parts=[]
    for idx in sorted(split()):
        row=json.loads((folder/f'sums-{idx}.json').read_text())
        parts.append([row['n'],*stats[str(idx)],*np.ravel(row['groups'])])
    data=np.asarray(parts,dtype=np.float64)
    assert len(data)==1500 and data[:,0].sum()==1185192
    def metrics(total):
        n,xx,xy=total[...,:3].T
        groups=total[...,3:]/n[...,None]
        slope=xy/xx
        return np.stack([slope,np.clip(slope,0,1),groups[...,:5].sum(-1),groups[...,5:].sum(-1),
                         *groups.T],axis=-1)
    rng=np.random.default_rng(20261124)
    sample=np.concatenate([metrics(data[rng.integers(1500,size=(50,1500))].sum(1)) for _ in range(200)])
    point=metrics(data.sum(0));interval=np.percentile(sample,[2.5,97.5],axis=0)
    names=['raw_slope','lambda','G','U',*['G/'+g for g in GROUPS],*['U/'+g for g in GROUPS]]
    adapter_sha=None
    if state!='B':
        arm,step=state.split('@')
        adapter_sha=hashlib.sha256((ROOT/'run'/arm/f'checkpoint-{step}/adapter_model.safetensors').read_bytes()).hexdigest()
    EVIDENCE.mkdir(exist_ok=True,parents=True)
    json_save(EVIDENCE/f'{state}.json',dict(state=state,questions=1500,positions=1185192,
        bootstrap_seed=20261124,draws=10000,adapter_sha256=adapter_sha,
        metrics={k:dict(value=v,ci95=ci.tolist()) for k,v,ci in zip(names,point,interval.T)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['features','factors','calibrate','measure','report'])
    parser.add_argument('state',choices=STATES)
    parser.add_argument('lane',type=int,nargs='?')
    args=parser.parse_args()
    (OUT/args.state).mkdir(parents=True,exist_ok=True)
    if args.stage=='report':report(args.state)
    else:globals()[args.stage](args.state,args.lane)
