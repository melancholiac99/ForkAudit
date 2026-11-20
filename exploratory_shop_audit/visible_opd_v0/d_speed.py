"""Lossless reuse of logged BF16 scorer outputs and measured activation checkpoint policy."""
import hashlib
import time
import torch
import torch.distributed as dist

from .c_smoke import kl_positions
from .s21_train import score
from exploratory_shop_audit.grpo_forced_audit.common import digest_file,read


def length(row):
    return len(row['prompt'])+len(row['answer'])


@torch.no_grad()
def initial_logits(model,batch,local,path):
    """Read the exact student logits produced by this update's verified frozen scorer."""
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


def set_checkpoint(modules,enabled):
    for module in modules:
        module.gradient_checkpointing=enabled


def benchmark(model,batch,q,params,local,modules):
    # Bound eager activation storage using an actual long row on each GPU.
    eligible=[i for i in local if length(batch.rows[i])<=4096]
    limit=0
    result={}
    if eligible and modules:
        i=max(eligible,key=lambda j:length(batch.rows[j]))
        row=batch.rows[i]
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
    shared=torch.tensor(limit,device='cuda')
    dist.all_reduce(shared,op=dist.ReduceOp.MIN)
    return shared.item(),result
