"""Lossless reuse of logged BF16 scorer outputs and measured activation checkpoint policy."""
import hashlib
import time
import torch
import torch.distributed as dist

from .c_smoke import kl_positions
from .s21_train import score
from exploratory_shop_audit.grpo_forced_audit.common import digest_file,read,save


def length(row):
    return len(row['prompt'])+len(row['answer'])


@torch.no_grad()
def initial_logits(model,batch,local,path):
    """Read the exact student logits produced by this update's verified frozen scorer."""
    if not local:return {},True
    manifest=path/'target_manifest.json'
    if manifest.exists():
        sha=read(manifest).get('bf16_before_sha256')
        file=path/'bf16_before.pt'
        if sha:
            assert digest_file(file)==sha
            saved=torch.load(file,weights_only=True)
            ordered=sorted(local,key=lambda i:length(batch.rows[i]))
            checks=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
            if all(torch.equal(saved[i],score(model,batch.rows[i]).cpu()) for i in checks):
                return {i:saved[i] for i in local},True
    return {i:score(model,batch.rows[i]).cpu() for i in local},False


@torch.no_grad()
def verify_nominations(model,batch,local,path,rank):
    """Replay the actual actor on each complete response before the first within-block update."""
    from .c_targets import response
    from .score import support
    saved_path=path/'bf16_before.pt'
    assert digest_file(saved_path)==read(path/'target_manifest.json')['bf16_before_sha256']
    saved=torch.load(saved_path,weights_only=True)
    before={};positions=0;max_prob=0.;max_logits=0.
    for i in local:
        row=batch.rows[i]
        _,raw=response(model,row['prompt'],row['answer'])
        ids,mask=support(raw,torch.tensor(row['answer'],device=raw.device))
        assert torch.equal(ids.cpu(),row['ids']) and torch.equal(mask.cpu(),row['mask'])
        before[i]=raw.gather(1,ids).cpu()
        p=before[i].masked_fill(~row['mask'],-torch.inf).softmax(-1)
        max_prob=max(max_prob,float((p-row['snapshot']).abs().max()))
        max_logits=max(max_logits,float((before[i]-saved[i]).abs().max()))
        positions+=len(ids)
    save(path/f'nomination_replay_{rank}.json',dict(records=len(local),positions=positions,
         exact_candidate_ids_and_mask=True,max_probability_error=max_prob,max_logit_error=max_logits))
    return before,False


def set_checkpoint(modules,enabled):
    for module in modules:
        module.gradient_checkpointing=enabled


def activation_probe(model,batch,q,params,modules,i):
    row=batch.rows[i]
    limit=0
    result={}
    for mode in ('checkpoint','retained'):
        model.zero_grad(set_to_none=True)
        set_checkpoint(modules,mode=='checkpoint')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        try:
            logits=score(model,row)
            loss,_=kl_positions(logits,q[i].cuda(),row['mask'].cuda())
            (loss.sum()*batch.weights[i]).backward()
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            peak=torch.cuda.max_memory_allocated()
            signature=hashlib.sha256()
            for _,p in params:
                signature.update(p.grad.detach().cpu().numpy().tobytes())
            result[mode]=dict(seconds=seconds,peak_bytes=peak,grad_sha=signature.hexdigest(),
                              logits_sha=hashlib.sha256(logits.detach().cpu().numpy().tobytes()).hexdigest())
            del loss,logits
        except torch.cuda.OutOfMemoryError:
            if mode=='checkpoint':raise
            result[mode]=dict(oom=True)
        finally:
            model.zero_grad(set_to_none=True)
            set_checkpoint(modules,True)
    torch.cuda.empty_cache()
    a,b=result['checkpoint'],result['retained']
    if not b.get('oom'):
        same=(a['grad_sha'],a['logits_sha'])==(b['grad_sha'],b['logits_sha'])
        room=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory-b['peak_bytes']
        if same and room>8*1024**3 and b['seconds']<a['seconds']:
            limit=length(row)
    result.update(tokens=length(row),local_limit=limit)
    return result


def local_probe(batch,local,modules,cap,probe):
    eligible=[i for i in local if length(batch.rows[i])<=cap]
    if not eligible or not modules:
        return {}
    i=max(eligible,key=lambda j:length(batch.rows[j]))
    result=probe(i)
    if result['local_limit']==0 and length(batch.rows[i])>4096:
        shorter=[j for j in eligible if length(batch.rows[j])<=4096]
        if shorter:
            fallback=probe(max(shorter,key=lambda j:length(batch.rows[j])))
            fallback['long_probe']=result
            return fallback
    return result


def benchmark(model,batch,q,params,local,modules,cap=4096):
    result=local_probe(batch,local,modules,cap,
                       lambda i:activation_probe(model,batch,q,params,modules,i))
    limit=result.get('local_limit',0)
    shared=torch.tensor(limit,device='cuda')
    dist.all_reduce(shared,op=dist.ReduceOp.MIN)
    return shared.item(),result
