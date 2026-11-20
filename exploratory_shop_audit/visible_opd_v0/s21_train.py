"""Replay each learning rate with fresh Adam moments on the same five minibatches."""
from pathlib import Path
import argparse
import os
import time

import torch
import torch.distributed as dist
from .c_targets import actor, response, RUN as ORIGINAL
from .c_smoke import kl_positions, same_hash
from .s21_data import SmokeBatch
from .s21_store import RUN, ARMS, GRID, pack
from exploratory_shop_audit.grpo_forced_audit.common import digest_value, read, save


def score(model,row):
    _,v=response(model,row['prompt'],row['answer'])
    return v.gather(1,row['ids'].cuda())


def run(mode):
    os.umask(0o077)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['NCCL_ALGO']='Ring'
    torch.use_deterministic_algorithms(True)
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==4
    batch=SmokeBatch()
    every=list(range(len(batch.rows)))
    model=actor()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    init={n:p.detach().cpu().clone() for n,p in params}
    initial_hash=same_hash(params)
    assert initial_hash==read(ORIGINAL/'arm_R.json')['initial_adapter_sha256']
    @torch.no_grad()
    def capture(subset):
        return {i:score(model,batch.rows[i]).cpu() for i in batch.local(subset,rank)}

    if mode=='grid':
        scales=GRID[1:]
        path=RUN/'zero'
        path.mkdir(exist_ok=True)
        base=capture(every);repeat=capture(every)
        for i,v in base.items():
            row=batch.rows[i]
            p=v.masked_fill(~row['mask'],-torch.inf).softmax(-1)
            assert torch.allclose(p,row['snapshot'],atol=1e-6,rtol=0)
        torch.save(dict(before=base,after=repeat),path/f'logits_{rank}.pt')
        dist.barrier()
    else:
        selected=read(RUN/'selection.json')['selected_multiplier']
        if selected is None:raise ValueError('No eligible common learning rate')
        scales=[selected]
    for scale in scales:
        for arm in ARMS:
            path=RUN/mode/f'{arm}_{scale:g}'
            path.mkdir(parents=True,exist_ok=True)
            if (path/'complete.json').exists():
                continue
            start=time.monotonic()
            targets=batch.targets(arm)
            with torch.no_grad():
                for n,p in params:p.copy_(init[n])
            assert same_hash(params)==initial_hash
            optim=torch.optim.AdamW([p for _,p in params],lr=1e-6*scale,weight_decay=0.)
            norms=[];records=[];probes=[capture(batch.probes)]
            for subset in batch.batches:
                optim.zero_grad(set_to_none=True)
                before={}
                for i in batch.local(subset,rank):
                    row=batch.rows[i]
                    logits=score(model,row)
                    before[i]=logits.detach().cpu()
                    loss,_=kl_positions(logits,targets[i].cuda(),row['mask'].cuda())
                    (loss.sum()*batch.weights[i]*(len(every)/len(subset))).backward()
                for n,p in params:
                    assert p.grad is not None
                    dist.all_reduce(p.grad)
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
                norms.append(float(norm))
                optim.step()
                records.append((before,capture(subset)))
                probes.append(capture(batch.probes))
            assert set(int(s['step'].item()) for s in optim.state.values())=={len(batch.batches)}
            final_hash=same_hash(params)
            final=capture(every)
            torch.save(dict(end=final,steps=records,probes=probes),path/f'logits_{rank}.pt')
            dist.barrier()
            if rank==0:
                pack(path/'adapter.zip',params,init)
                save(path/'complete.json',dict(arm=arm,multiplier=scale,lr=1e-6*scale,
                    initial_sha256=initial_hash,final_sha256=final_hash,replicas=4,optimizer_steps=len(norms),
                    minibatch_rows=[len(b) for b in batch.batches],minibatch_sha256=digest_value(batch.batches),
                    grad_norms=norms,gpu_hours=4*(time.monotonic()-start)/3600))
                print(mode,arm,scale,'finished',flush=True)
            dist.barrier()
            del optim
    dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['grid','rerun'])
    run(p.parse_args().mode)
