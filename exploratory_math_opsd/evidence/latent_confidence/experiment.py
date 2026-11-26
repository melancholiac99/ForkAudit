"""Measure confidence on archived prefixes. All positions count consumed response tokens."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
REF = HERE.parent / 'cond_kl/forcing/reflection'
sys.path.insert(0, str(REF))
from settings import ROOT, MODELS, SEED, read, save
from prepare import boundaries
from common_io import archived
RAW = ROOT / 'run/latent-confidence'
BASE = ROOT / 'models/Qwen3-1.7B'
SUFFIX = '\n\nMy confidence that the reasoning so far is correct, on a scale from 0 to 9, is'
ARMS = list(MODELS)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def uniform(items, n, rng):
    return [items[i] for i in sorted(rng.choice(len(items), min(n,len(items)), replace=False))]

def spread(items,n):
    return [items[i] for i in (np.arange(n)*len(items)//n)]

def manifest():
    labels=read(ROOT/'run/cond-kl/forcing/reflection/labels.json')
    ids=sorted(map(int,labels))
    extract, assess=ids[::2], ids[1::2]
    assert len(set(extract)|set(assess))==1500 and not set(extract)&set(assess)
    rng=np.random.default_rng(SEED)
    def sample(ids,n):
        return uniform([(i,p) for i in ids for p,_ in labels[str(i)]['boundaries']], n,rng)
    train=sample(extract,4000)
    val=set(uniform(range(4000),800,rng))
    fixed=sample(assess,3000)
    save(RAW/'manifest.json',dict(extract=extract,assess=assess,train=[[i,p,int(j in val)] for j,(i,p) in enumerate(train)],prompts=fixed,
         pilot_extract=spread(extract,34),pilot_assess=spread(assess,33),pilot_math=spread(list(range(500)),33)))

def actor_for(arm):
    base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    if arm!='B':
        name=arm.split('@')[0]; path=ROOT/'run'/name/'checkpoint-100'
        f=path/'adapter_model.safetensors'
        assert digest(f)==read(REF/'audit.json')['adapters'][name]
        peft=PeftModel.from_pretrained(base,str(path),adapter_name=name,is_trainable=False)
        peft.set_adapter(name)
        src=load_file(str(f)); got=get_peft_model_state_dict(peft,adapter_name=name,save_embedding_layers=False)
        assert peft.active_adapter==name and got.keys()==src.keys() and all(torch.equal(got[k].cpu(),v) for k,v in src.items())
        base=peft.get_base_model()
    return base.eval()

class Reader:
    def __init__(self,arm,tok,layer=None,direction=None):
        self.net=actor_for(arm)
        self.suffix=tok.encode(SUFFIX,add_special_tokens=False)
        self.digits=[];self.values=[]
        for n in range(10):
            ids={a[0] for s in (str(n),' '+str(n)) if len(a:=tok.encode(s,add_special_tokens=False))==1}
            assert ids
            self.digits.extend(sorted(ids));self.values.extend([n]*len(ids))
        assert len(set(self.digits))==len(self.digits)
        self.layer=layer
        self.direction=None if direction is None else torch.as_tensor(direction,device='cuda',dtype=torch.float32)
    def run(self,ids,positions,asked=False):
        selected=[]
        handles=[]
        def grab(_,args,out):
            h=out[0] if isinstance(out,tuple) else out
            v=h[batch,idx].float()
            selected.append((v if self.direction is None else v@self.direction).cpu().numpy())
        net=self.net; batch,idx=positions
        for j,block in enumerate(net.model.layers):
            if self.layer is None or j==self.layer:handles.append(block.register_forward_hook(grab))
        try:
            h=net.model(input_ids=ids,use_cache=False).last_hidden_state
        finally:
            for hook in handles:hook.remove()
        features=np.stack(selected,axis=1)
        if not asked:return features
        logits=net.lm_head(h[batch,idx]).float()
        dl=logits[:,self.digits]
        mass=(dl.logsumexp(1)-logits.logsumexp(1)).exp()
        ec=dl.softmax(1)@torch.tensor(self.values,device='cuda',dtype=torch.float)
        return features,ec.cpu().numpy(),mass.cpu().numpy()
    def natural(self,sp,y,pp):
        ids=torch.tensor([sp+y],device='cuda')
        idx=torch.as_tensor(pp,device='cuda')+len(sp)-1
        return self.run(ids,(torch.zeros_like(idx),idx))
    def prompt(self,sp,y,pp):
        # Right padding is strictly after each readout, so causal attention never sees it.
        groups=[]
        for start in range(0,len(pp),8):
            seq=[sp+y[:p]+self.suffix for p in pp[start:start+8]]
            lens=torch.tensor([len(x) for x in seq],device='cuda')
            ids=torch.tensor([x+[0]*(max(map(len,seq))-len(x)) for x in seq],device='cuda')
            groups.append(self.run(ids,(torch.arange(len(seq),device='cuda'),lens-1),True))
        return tuple(np.concatenate(a) for a in zip(*groups))

def archive():
    rows={}
    for lane in range(4):
        f=ROOT/'run/P-projector'/f'rollouts-{lane}.pt'
        expected=read(HERE.parent/'stop_decomposition/results.json')['inputs']['archives'][f.name]
        assert digest(f)==expected
        for i,sp,_,y in torch.load(f,weights_only=True):rows[i]=(sp,y)
    return rows

def fit(files,out):
    assert len(files)
    with np.load(files[0]) as a: shape=a['h'].shape[1:]
    pooled=[np.load(f) for f in files]
    ec=np.concatenate([a['ec'] for a in pooled]);valid=np.concatenate([a['valid'] for a in pooled]).astype(bool)
    mid=np.median(ec[~valid]);high=(ec>mid)&~valid;low=(ec<=mid)&~valid
    sums=np.zeros((2,*shape),np.float64); nv=[];offset=0
    for a in pooled:
        h=a['h'];sl=slice(offset,offset+len(h));offset+=len(h)
        sums[0]+=h[low[sl]].sum(axis=0,dtype=np.float64)
        sums[1]+=h[high[sl]].sum(axis=0,dtype=np.float64)
        nv.append(a['natural'][valid[sl]])
    d=sums[1]/high.sum()-sums[0]/low.sum()
    dirs=(d/np.linalg.norm(d,axis=1,keepdims=True)).astype(np.float32)
    nv=np.concatenate(nv)
    for j,d in enumerate(dirs):corr.append(float(spearmanr(nv[:,j]@d,ec[valid]).statistic))
    layer=int(np.nanargmax(corr))
    mass=np.concatenate([a['mass'] for a in pooled]);[a.close() for a in pooled]
    np.save(out/'direction.npy',dirs[layer])
    save(out/'fit.json',dict(layer_index_zero_based=layer,correlations=corr,threshold=float(mid),digit_mass=float(mass.mean()),train=int((~valid).sum()),validation=int(valid.sum())))

def dump(f,**data):
    tmp=f.with_suffix('.tmp.npz');np.savez(tmp,**data);tmp.replace(f)

@torch.inference_mode()
def worker(stage,mode,arm,lane,lanes,reader=None,opportunistic=False):
    m=read(RAW/'manifest.json');out=RAW/mode;out.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(BASE)
    if stage=='fit':return fit(sorted((out/'extract').glob('*.npz')),out)
    if reader is not None:pass
    elif stage=='extract':
        reader=Reader('B',tok)
    else:
        reader=Reader(arm,tok,read(out/'fit.json')['layer_index_zero_based'],np.load(out/'direction.npy'))
    dest=out/(stage if stage=='extract' else stage+'-'+arm);dest.mkdir(exist_ok=True)
    ids=m['pilot_extract' if stage=='extract' else 'pilot_assess'] if mode=='pilot' else m['extract' if stage=='extract' else 'assess']
    if stage=='math':
        ids=m['pilot_math'] if mode=='pilot' else list(range(500))
        math=archived(arm);reference=archived('B') if arm!='B' else math
        assert len(math)==len(reference)==500 and len({r['problem'] for r in reference})==500
        assert all(a['problem']==b['problem'] and a['ground_truth']==b['ground_truth'] for a,b in zip(math,reference))
        word_ids=set(read(REF/'tokens.json')['included_ids'])
        decode=lru_cache(None)(lambda t:tok.decode([t],clean_up_tokenization_spaces=False))
    else:
        fixed=archive();labels=read(ROOT/'run/cond-kl/forcing/reflection/labels.json')
        positions=defaultdict(list)
        for i,*rest in m['train' if stage=='extract' else 'prompts']: positions[i].append(rest)
    import fcntl
    lock=None
    for i in ids[lane::lanes]:
        if lock:lock.close()
        began=time.monotonic()
        f=dest/f'{i}.npz'
        if f.exists():continue
        if opportunistic:
            lock=(dest/f'{i}.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if f.exists():continue
        if stage=='math':
            row=math[i];assert len(row['generations'])==2
            sp=tok.apply_chat_template([dict(role='user',content=row['problem']+'\n\nPlease reason step by step, and put your final answer within \\boxed{}.')],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            record={}
            for k,g in enumerate(row['generations']):
                y=tok.encode(g['full_generation'],add_special_tokens=False)
                ends=[p for p,_ in boundaries(y,decode)]
                rng=np.random.default_rng([SEED,i,k])
                pp=np.array(uniform(ends,20,rng),int)
                ev=np.array([j for j,t in enumerate(y) if t in word_ids and j>=64],int)
                ctrl=rng.integers(64,len(y),size=len(ev)) if len(ev) else ev.copy()
                allp=np.arange(1,len(y)+1)
                z=reader.natural(sp,y,allp).ravel() if len(y) else np.empty(0)
                # j is zero-based event index. z[j-1] is the immediate predecessor.
                lag=np.arange(-64,0)
                for key,v in dict(p=pp,proj=z[pp-1],length=len(y),archived_length=g['generated_tokens'],label=g['correct'],
                    count=len(ev),events=z[ev[:,None]+lag].sum(0,dtype=np.float64),controls=z[ctrl[:,None]+lag].sum(0,dtype=np.float64),
                    drops=[int(((z[t[:,None]+lag][:,-16:].mean(1)-z[t[:,None]+lag][:,:16].mean(1))<0).sum()) for t in (ev,ctrl)]).items():record[f'{key}{k}']=v
            dump(f,elapsed=time.monotonic()-began,**record);continue
        sp,y=fixed[i];pp=np.array([r[0] for r in positions[i]],int)
        if stage=='extract':
            if len(pp):
                h,e,mass=reader.prompt(sp,y,pp)
                dump(f,elapsed=time.monotonic()-began,h=h,natural=reader.natural(sp,y,pp),ec=e,mass=mass,valid=np.array([r[1] for r in positions[i]]))
        else:
            allp=np.array([p for p,_ in labels[str(i)]['boundaries']],int)
            natural=reader.natural(sp,y,allp).ravel()
            ec,mass=reader.prompt(sp,y,pp)[1:] if len(pp) else (np.empty(0),np.empty(0))
            correct=labels[str(i)]['correct']
            dump(f,elapsed=time.monotonic()-began,p=allp,proj=natural,length=len(y),label=-1 if correct is None else int(correct),ask=pp,ec=ec,mass=mass)
if __name__=='__main__':
    stage,mode,arm,lane,lanes=sys.argv[1:]
    RAW.mkdir(exist_ok=True)
    if stage=='manifest':manifest()
    else:worker(stage,mode,arm,int(lane),int(lanes))
