"""One synchronized optimizer step from identical adapters for each smoke arm."""
from collections import Counter, defaultdict
import hashlib
import os
import time

import torch
import torch.distributed as dist

from .c_targets import ARMS, RUN, actor, response
from exploratory_shop_audit.grpo_forced_audit.common import read, save


def kl_positions(logits, target, mask):
    """Forward KL with its analytic candidate-logit gradient; exact zero for identical probabilities."""
    if target.requires_grad or not mask.any(1).all() or (target[~mask]!=0).any():
        raise ValueError('invalid detached candidate target')
    with torch.no_grad():
        p = logits.detach().float().masked_fill(~mask,-torch.inf).softmax(-1)
        if not p.isfinite().all() or not torch.allclose(target.sum(1),torch.ones_like(p[:,0]),atol=1e-5):
            raise ValueError('nonfinite logits or non-normalized targets')
        # Compute the value in float64 to retain very small positive divergences.
        lp = logits.detach().double().masked_fill(~mask,-torch.inf).log_softmax(-1).masked_fill(~mask,0.)
        q = target.double()
        q = q/q.sum(1,keepdim=True)
        value = (torch.special.xlogy(q,q)-q*lp).sum(1).clamp_min(0)
        derivative = (p-target).masked_fill(~mask,0)
    return value+((logits-logits.detach())*derivative).sum(1),p


def state_hash(parameters):
    h=hashlib.sha256()
    for name,p in parameters:
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def same_hash(parameters):
    value=state_hash(parameters)
    hashes=[None]*dist.get_world_size()
    dist.all_gather_object(hashes,value)
    if len(set(hashes))!=1:
        raise ValueError('adapter replicas diverged')
    return value


def run(archive_replay=False):
    os.umask(0o077)
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==4
    start=time.monotonic()
    out=RUN
    if archive_replay:
        from .s21_store import RUN as DIAGNOSTIC, pack
        out=DIAGNOSTIC/'original'
        out.mkdir(parents=True,exist_ok=True)
    shard=RUN/f'batch_{rank}'
    rows=torch.load(shard/'student.pt',weights_only=True)
    expected={'goal','rep','step','prompt','answer','ids','mask','classes','snapshot'}
    assert all(set(r)==expected for r in rows)
    counts=Counter()
    for r in rows:counts[r['goal'],r['rep']]+=len(r['answer'])
    assert len(counts)==16 and len({g for g,rep in counts})==8
    model=actor()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    params=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    assert params and all('lora_' in n for n,p in params)
    initial={name:p.detach().cpu().clone() for name,p in params}
    base_hash=same_hash(params)
    results={}
    if archive_replay and rank==0:
        torch.save(initial,out/'initial.pt')
    for arm in ARMS:
        if archive_replay and arm=='N':continue
        with torch.no_grad():
            for name,p in params:p.copy_(initial[name])
        assert same_hash(params)==base_hash
        targets=torch.load(shard/f'{arm}.pt',weights_only=True)
        assert len(targets)==len(rows)
        optim=torch.optim.AdamW([p for _,p in params],lr=1e-6,weight_decay=0.)
        optim.zero_grad(set_to_none=True)
        def scan(backward):
            total=0.;largest=0.;metrics=defaultdict(Counter);saved_logits=[]
            for r,t in zip(rows,targets,strict=True):
                _,logits=response(model,r['prompt'],r['answer'])
                logits=logits.gather(1,r['ids'].cuda())
                if archive_replay:saved_logits.append(logits.detach().cpu())
                target=t.cuda();mask=r['mask'].cuda();snap=r['snapshot'].cuda()
                loss,prob=kl_positions(logits,target,mask)
                weighted=loss.sum()/(64*counts[r['goal'],r['rep']])
                if backward:weighted.backward()
                total+=weighted.item()
                largest=max(largest,float((prob-snap).abs().max()))
                with torch.no_grad():
                    zero=(target-snap).abs().max(1).values<=1e-8
                    grad=(target-prob).abs().sum(1)
                    ent=-torch.special.xlogy(target,target).sum(1)
                    for c in set(r['classes']):
                        m=torch.tensor([x==c for x in r['classes']],device='cuda')
                        v=metrics[c]
                        v.update(dict(n=int(m.sum()),kl_sum=loss[m].detach().sum().item(),
                                      gradient_l1_sum=grad[m].sum().item(),entropy_sum=ent[m].sum().item(),
                                      zero_n=int((m&zero).sum()),zero_kl_sum=loss[m&zero].detach().sum().item()))
            if archive_replay:
                phase='before' if backward else 'after'
                torch.save(saved_logits,out/f'{arm}_{phase}_{rank}.pt')
            return dict(task_weighted_kl=total,max_snapshot_error=largest,by_class=dict(metrics))
        before=scan(True)
        for _,p in params:
            assert p.grad is not None
            dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True).item()
        if arm=='N':
            assert before['max_snapshot_error']<=1e-5 and norm<=1e-5,(before['max_snapshot_error'],norm)
        else:
            assert norm>0
        optim.step()
        final_hash=same_hash(params)
        if arm!='N' and final_hash==base_hash:raise ValueError('positive arm did not update LoRA')
        with torch.no_grad():
            after=scan(False)
        local=dict(before=before,after=after)
        gathered=[None]*4
        dist.all_gather_object(gathered,local)
        def merge(phase):
            metrics=defaultdict(Counter)
            for item in gathered:
                for c,v in item[phase]['by_class'].items():metrics[c].update(v)
            return dict(task_weighted_kl=sum(x[phase]['task_weighted_kl'] for x in gathered),
                        max_snapshot_error=max(x[phase]['max_snapshot_error'] for x in gathered),by_class=dict(metrics))
        results[arm]=dict(before=merge('before'),after=merge('after'),gradient_norm=norm,
                          initial_adapter_sha256=base_hash,final_adapter_sha256=final_hash,
                          synchronized_ranks=4,optimizer_steps=1,epochs=1)
        if rank==0:
            if archive_replay:
                expected=read(RUN/f'arm_{arm}.json')
                assert base_hash==expected['initial_adapter_sha256']
                results[arm]['matches_original_final_hash']=final_hash==expected['final_adapter_sha256']
                pack(out/f'{arm}.zip',params,initial)
            save(out/f'arm_{arm}.json',results[arm])
            print(arm,'completed',norm,flush=True)
        del optim
    dist.barrier()
    if rank==0:
        shards=[read(RUN/f'batch_{k}/manifest.json') for k in range(4)]
        save(out/'smoke_report.json',dict(arms=results,tasks=32,trajectories=64,shards=shards,
            steps=sum(s['steps'] for s in shards),positions=sum(s['positions'] for s in shards),
            optimizer='AdamW',lr=1e-6,weight_decay=0.,gradient_clip=1.,lora_rank=64,lora_alpha=64,
            structural_support_shared=True,reward_fields_in_actor_batch=0,eval_V_tasks=0,
            target_gpu_hours=sum(s['gpu_hours'] for s in shards),train_gpu_hours=4*(time.monotonic()-start)/3600))
    dist.destroy_process_group()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive-replay',action='store_true')
    run(parser.parse_args().archive_replay)
