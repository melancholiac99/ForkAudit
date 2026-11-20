"""Frozen BF16/FP32 measurement of archived real and norm-matched null updates."""
import argparse
import hashlib
import os
import time
import zipfile
from collections import defaultdict

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import DynamicCache

from .c_smoke import state_hash
from .c_targets import actor, SEED
from .s21_data import SmokeBatch
from .s21_store import ARMS, RUN as PREVIOUS, unpack_member
from .s21_train import score
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save, stable_seed

RUN=PREVIOUS.parent/'shop_visible_opd_s22_20261120'
SCALES=(1,2,4)
CHUNK=512


def prefix_length(x,y):
    return next((i for i,(a,b) in enumerate(zip(x,y)) if a!=b),min(len(x),len(y)))


def shard_order(batch,rank):
    """Keep a task's shared prefixes on the same device; balance unique suffix work."""
    by_task=defaultdict(list)
    seq=[r['prompt']+r['answer'] for r in batch.rows]
    for i,r in enumerate(batch.rows):by_task[r['goal']].append(i)
    groups=[]
    for indices in by_task.values():
        indices.sort(key=lambda i:seq[i])
        cost=sum(len(seq[i])-prefix_length(seq[i],seq[j]) if j is not None else len(seq[i])
                 for i,j in zip(indices,[None]+indices[:-1]))
        groups.append((cost,indices))
    bins=[[] for _ in range(4)];loads=[0]*4
    for cost,indices in sorted(groups,reverse=True):
        k=min(range(4),key=lambda k:(loads[k],k))
        bins[k].extend(indices);loads[k]+=cost
    return sorted(bins[rank],key=lambda i:seq[i])


@torch.inference_mode()
def fp32_scores(model,batch,order):
    base=model.get_base_model()
    outputs={}
    maxlen=max(len(batch.rows[i]['prompt'])+len(batch.rows[i]['answer']) for i in order)
    hidden=torch.empty(maxlen,model.config.hidden_size,device='cuda',dtype=torch.float32)
    cache=DynamicCache(config=base.config)
    previous=[]
    with sdpa_kernel(SDPBackend.MATH):
        for i in order:
            r=batch.rows[i];seq=r['prompt']+r['answer']
            split=prefix_length(previous,seq)
            if previous:cache.crop(split)
            assert cache.get_seq_length()==split
            for start in range(split,len(seq),CHUNK):
                end=min(start+CHUNK,len(seq))
                ids=torch.tensor([seq[start:end]],device='cuda')
                out=base.model(input_ids=ids,past_key_values=cache,use_cache=True)
                hidden[start:end].copy_(out.last_hidden_state[0])
                cache=out.past_key_values
            h=hidden[len(r['prompt'])-1:len(seq)-1]
            v=base.lm_head(h).gather(1,r['ids'].cuda())
            assert v.dtype==torch.float32 and v.isfinite().all()
            outputs[i]=v.cpu();previous=seq
    return outputs


def matched_null(initial,norm,seed):
    gen=torch.Generator().manual_seed(seed)
    noise={n:torch.randn(v.shape,generator=gen,dtype=torch.float32) for n,v in initial.items()}
    n2=sum(v.double().square().sum().item() for v in noise.values())
    scale=norm/n2**.5
    return {n:v+scale*noise[n] for n,v in initial.items()}


def restore(params,initial,values):
    h=hashlib.sha256();squared=0.
    for n,p in params:
        v=values[n]
        assert v.dtype==p.dtype==torch.float32
        h.update(n.encode());h.update(v.numpy().tobytes())
        squared+=(v.double()-initial[n].double()).square().sum().item()
        p.copy_(v)
    return h.hexdigest(),squared**.5


@torch.inference_mode()
def run(precision):
    os.umask(0o077)
    rank=int(os.environ['LOCAL_RANK'])
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(rank)
    dtype=torch.float32 if precision=='fp32' else torch.bfloat16
    model=actor(dtype=dtype).eval()
    assert model.get_base_model().lm_head.weight.dtype==dtype
    if precision=='fp32':
        assert all(p.dtype==dtype for p in model.parameters())
        assert not torch.backends.cuda.matmul.allow_tf32
    batch=SmokeBatch();order=shard_order(batch,rank) if precision=='fp32' else batch.local(list(range(len(batch.rows))),rank)
    initial=torch.load(PREVIOUS/'original/initial.pt',map_location='cpu',weights_only=True)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    old=read(PREVIOUS/'step_size_diagnostics.json')
    h0=read(PREVIOUS/'grid/R_1/complete.json')['initial_sha256']
    h,norm=restore(params,initial,initial);assert h==h0 and norm==0
    def measure(key,sha,norm,seed=None):
        dest=RUN/precision/key;dest.mkdir(parents=True,exist_ok=True)
        manifest=dest/f'manifest_{rank}.json'
        if manifest.exists():
            prior=read(manifest)
            assert prior['weight_sha256']==sha and digest_file(dest/f'logits_{rank}.pt')==prior['logits_sha256']
            return
        start=time.monotonic()
        logits=(fp32_scores(model,batch,order) if precision=='fp32' else
                {i:score(model,batch.rows[i]).cpu() for i in order})
        assert all(x.isfinite().all() for x in logits.values())
        # Compare cached FP32 against independent full-prefix FP32 on the four fixed probe records.
        errors={}
        if precision=='fp32' and key in ('initial','R_1','null_R_0'):
            with sdpa_kernel(SDPBackend.MATH):
                for i in set(batch.probes)&set(order):
                    err=(logits[i]-score(model,batch.rows[i]).cpu()).abs().max().item()
                    if err>1e-3:raise ValueError(('FP32 prefix mismatch',i,err))
                    errors[str(i)]=err
        if precision=='bf16' and key=='initial':
            ref=torch.load(PREVIOUS/'zero'/f'logits_{rank}.pt',weights_only=True)['before']
            assert all(torch.equal(v,ref[i]) for i,v in logits.items())
        p=dest/f'logits_{rank}.pt'
        torch.save(logits,p.with_suffix('.tmp'));p.with_suffix('.tmp').replace(p)
        save(manifest,dict(precision=precision,checkpoint=key,seed=seed,rank=rank,
            records=len(order),positions=sum(len(v) for v in logits.values()),weight_sha256=sha,
            update_norm=norm,logits_sha256=digest_file(p),probe_direct_max_error=errors,
            seconds=time.monotonic()-start,tf32=False))
        print(rank,precision,key,'complete',flush=True)
    measure('initial',h0,0.)
    for a in ARMS:
        if precision=='fp32':
            for m in SCALES:
                key=f'{a}_{m}';src=PREVIOUS/'grid'/key
                with zipfile.ZipFile(src/'adapter.zip') as z:
                    vals={n:torch.from_numpy(unpack_member(z,n,initial)) for n,_ in params}
                h,norm=restore(params,initial,vals);del vals
                assert h==read(src/'complete.json')['final_sha256']
                assert abs(norm-old['geometry'][str(float(m))]['norms'][a])<1e-8
                measure(key,h,norm)
        goal=old['geometry']['1.0']['norms'][a]
        for j in range(5):
            seed=stable_seed(SEED,'section22',a,j)
            vals=matched_null(initial,goal,seed)
            h,norm=restore(params,initial,vals);del vals
            assert abs(norm/goal-1)<1e-5
            measure(f'null_{a}_{j}',h,norm,seed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('precision',choices=['bf16','fp32'])
    run(p.parse_args().precision)
