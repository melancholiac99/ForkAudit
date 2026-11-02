"""Bounded, disposable E4b input-cache/synchronization benchmark."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

H=Path(__file__).resolve().parent; ROOT=H.parent
BASE=ROOT.parent/'models/Qwen2.5-7B-Instruct'
ADAPTER=ROOT/'exploratory_E4/data/adapters/Aprime_s0_lr5e-06'
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def save(name, x):
    p=H/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def prepare():
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); candidates=[]; hashes={}
    for f in sorted((ROOT/'exploratory_E4b/data/pairs').glob('*.jsonl')):
        hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        for r in map(json.loads,f.read_text().splitlines()):
            text=tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=False,add_generation_prompt=True)
            pi=tok.encode(text,add_special_tokens=False)
            job=dict(arm=f.stem,point_id=r['point_id'])
            for k,field in [('c','chosen'),('r','rejected')]:
                job[k]=[pi,tok.encode(r[field],add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]]
            job['tokens']=sum(len(z) for k in ('c','r') for z in job[k]);candidates.append(job)
    candidates.sort(key=lambda j:(j['tokens'],j['arm'],j['point_id']))
    selected=[candidates[round(i*(len(candidates)-1)/15)] for i in range(16)]
    save('trial_inputs.json',dict(selection='16 evenly spaced token-length ranks, no outcome/loss selection',
                                inputs=hashes,cases=selected,minimum_speedup=1.25,minimum_net_saving_minutes=30))
    print('Prepared',len(selected),'pairs; token lengths',[x['tokens'] for x in selected],flush=True)


def cached(model,jobs):
    dev=next(model.parameters()).device
    return [{k:(torch.tensor([j[k][0]+j[k][1]],device=dev),torch.tensor(j[k][1],device=dev)[:,None],len(j[k][0]))
             for k in ('c','r')} for j in jobs]


def logp(model,pair,ready=None):
    pi,ri=pair;dev=next(model.parameters()).device
    if ready is None:
        ids=torch.tensor([pi+ri],device=dev)
    else:ids=ready[0]
    with torch.autocast(dev.type,dtype=torch.bfloat16):
        logits=model(input_ids=ids,use_cache=False if model.training else None).logits[0,len(pi)-1:-1].float()
    lp=torch.log_softmax(logits,-1).gather(1,torch.tensor(ri,device=dev)[:,None] if ready is None else ready[1])[:,0]
    total=lp.sum()
    _=lp.mean()  # Retain the frozen trainer's unused mean reduction in both variants.
    return total


def refs(model,jobs):
    model.eval()
    with torch.no_grad(),model.disable_adapter():
        return [[float(logp(model,j[k])) for k in ('c','r')] for j in jobs]


def trial(model,jobs,reference,ready,fast,seed,optimizer=None,gradient=False):
    model.train();model.zero_grad(set_to_none=True);torch.manual_seed(seed)
    cuda=next(model.parameters()).is_cuda
    if cuda:torch.cuda.synchronize()
    start=time.perf_counter();losses=[];correct=[]
    for i,j in enumerate(jobs):
        pc=logp(model,j['c'],ready[i]['c'] if fast else None)
        pr=logp(model,j['r'],ready[i]['r'] if fast else None)
        margin=(pc-reference[i][0])-(pr-reference[i][1]);loss=-torch.nn.functional.logsigmoid(.1*margin)
        correct.append((margin.detach()>0) if fast else int(margin.item()>0))
        (loss/16).backward();losses.append(loss.detach() if fast else loss.item())
    if fast:
        values=torch.stack(losses).tolist();acc=int(torch.stack(correct).sum().item())
    else:values=losses;acc=sum(correct)
    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    if cuda:torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    grad=torch.cat([p.grad.detach().flatten().float().cpu() for p in model.parameters() if p.requires_grad]) if gradient else None
    return dict(seconds=elapsed,losses=values,margin_positive=acc),grad


def compare(a,b,ga,gb):
    x,y=ga.double(),gb.double();norm=float(torch.linalg.vector_norm(x));other=float(torch.linalg.vector_norm(y))
    relative=float(torch.linalg.vector_norm(x-y))/max(norm,1e-30);cos=float(torch.dot(x,y))/max(norm*other,1e-30)
    loss=max(abs(u-v) for u,v in zip(a['losses'],b['losses']))
    return dict(passed=loss<=1e-4 and relative<=1e-3 and cos>=.99999 and a['margin_positive']==b['margin_positive'],
                max_loss_difference=loss,gradient_relative_l2=relative,gradient_cosine=cos,
                gradients_bitwise_equal=torch.equal(ga,gb),margin_counts_equal=a['margin_positive']==b['margin_positive'])


def selftest():
    torch.set_num_threads(1);torch.manual_seed(91)
    cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)
    cfg._attn_implementation='sdpa';m=Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16)
    m=get_peft_model(m,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=TARGETS,task_type='CAUSAL_LM'))
    for n,p in m.named_parameters():
        if p.requires_grad:
            p.data=p.data.float()
            if 'lora_B' in n:p.data.normal_(0,.01)
    jobs=[dict(c=[[1,2,3],[4,5,6]],r=[[1,2,3],[7,8]])]*16
    rr=refs(m,jobs);cc=cached(m,jobs)
    a,ga=trial(m,jobs,rr,cc,False,71,gradient=True);b,gb=trial(m,jobs,rr,cc,True,71,gradient=True)
    comparison=compare(a,b,ga,gb);assert comparison['passed'] and comparison['gradients_bitwise_equal']
    params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];updates=[]
    for fast in (False,True):
        for p,x in zip(params,initial):p.data.copy_(x)
        opt=torch.optim.AdamW(params,lr=5e-6,weight_decay=0.)
        trial(m,jobs,rr,cc,fast,72,optimizer=opt)
        updates.append(torch.cat([p.detach().flatten() for p in params]))
    assert torch.equal(*updates)
    save('trial_selftest.json',dict(passed=True,comparison=comparison,adamw_update_bitwise_equal=True,cpu_only=True))
    print('CPU loss/gradient/AdamW update parity passed',flush=True)


def run():
    inputs=json.loads((H/'trial_inputs.json').read_text());jobs=inputs['cases']
    for p,h in inputs['inputs'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    m=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    m.enable_input_require_grads();m.gradient_checkpointing_disable()
    m=PeftModel.from_pretrained(m,ADAPTER,is_trainable=True)
    for p in m.parameters():
        if p.requires_grad:p.data=p.data.float()
    cc=cached(m,jobs);rr=refs(m,jobs)
    torch.use_deterministic_algorithms(True)
    a,ga=trial(m,jobs,rr,cc,False,7301,gradient=True)
    b,gb=trial(m,jobs,rr,cc,True,7301,gradient=True)
    comparison=compare(a,b,ga,gb);save('trial_numerical.json',comparison);del ga,gb
    if not comparison['passed']:
        save('trial_result.json',dict(complete=True,adopt=False,reason='Numerical equivalence failed',comparison=comparison));return
    torch.use_deterministic_algorithms(False)
    params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params]
    timings=[]
    for repeat in range(4):
        for fast in ([False,True] if repeat%2==0 else [True,False]):
            for p,x in zip(params,initial):p.data.copy_(x)
            opt=torch.optim.AdamW(params,lr=5e-6,weight_decay=0.)
            result,_=trial(m,jobs,rr,cc,fast,7302+repeat,optimizer=opt)
            if repeat>0:timings.append(dict(variant='cached_deferred' if fast else 'original',repeat=repeat,**result))
            del opt
            save('trial_timing_progress.json',timings)
    med={v:statistics.median(t['seconds'] for t in timings if t['variant']==v) for v in ('original','cached_deferred')}
    speedup=med['original']/med['cached_deferred'];saving=(med['original']-med['cached_deferred'])*120*14/60
    out=dict(complete=True,comparison=comparison,median_seconds=med,speedup=speedup,
             projected_remaining_training_gross_saving_minutes=saving,remaining_runs_assumed=14,
             numerical_control_deterministic=True,timing_deterministic=False,default_timing_cublas_workspace=':4096:8',
             timings=timings,formal_implementation_changed=False,
             note='16 length-ranked samples, not a measured full-run speedup; GPU setup and projected savings scope disclosed.')
    save('trial_result.json',out);print(json.dumps({k:v for k,v in out.items() if k!='timings'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','selftest','run']);args=p.parse_args();globals()[args.mode]()
