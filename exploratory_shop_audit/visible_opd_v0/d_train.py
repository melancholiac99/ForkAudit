"""Apply one registered DDP epoch per arm, preserving Adam moments between updates."""
import os
import sys
import time
import torch
import torch.distributed as dist
from .c_targets import actor
from .c_smoke import kl_positions, same_hash
from .d_speed import benchmark, initial_logits, length, set_checkpoint
from .d_data import folder, state_dir, load_batch, snapshot
from .s21_train import score
from .s21_store import ARMS, RUN as PREVIOUS
from .s21_data import components
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, digest_value, read, save

def train(k, *, arms=ARMS, folder=folder, state_dir=state_dir, snapshot=snapshot, load_batch=load_batch, rates=None):
    os.umask(0o077)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['NCCL_ALGO']='Ring'
    torch.use_deterministic_algorithms(True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==4
    model=actor()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in params} if k==1 else None
    checkpoint_modules=[m for m in model.modules() if getattr(m,'gradient_checkpointing',False)]
    fast_limit=None
    for arm in arms:
        path=folder(k,arm);storage=state_dir(k,arm)
        if (path/'optimizer.json').exists():continue
        start=time.monotonic()
        if k==1:
            with torch.no_grad():
                for n,p in params:p.copy_(initial[n])
        else:snapshot(model,k-1,arm)
        before_sha=same_hash(params)
        expected=(read(PREVIOUS/'grid/R_1/complete.json')['initial_sha256'] if k==1
                  else read(path/'target_manifest.json')['scorer_sha'])
        assert expected==before_sha
        batch,q=load_batch(k,arm);every=list(range(len(batch.rows)))
        local=batch.local(every,rank)
        with torch.no_grad():
            before,reused=initial_logits(model,batch,local,path)
            for i,v in before.items():
                p0=v.masked_fill(~batch.rows[i]['mask'],-torch.inf).softmax(-1)
                assert torch.allclose(p0,batch.rows[i]['snapshot'],atol=1e-6,rtol=0)
        optim=torch.optim.AdamW([p for _,p in params],lr=1e-6 if rates is None else rates[arm],weight_decay=0.)
        prior_steps=0
        if k>1:
            prev=state_dir(k-1,arm)/'adam.pt'
            meta=read(folder(k-1,arm)/'optimizer.json')
            assert digest_file(prev)==meta['adam_sha256']
            optim.load_state_dict(torch.load(prev,map_location='cpu',weights_only=True))
            prior_steps=meta['cumulative_steps']
        if k>1 and fast_limit is None:
            fast_limit,probe=benchmark(model,batch,q,params,local,checkpoint_modules)
            save(path.parent/f'activation_check_rank{rank}.json',dict(limit=fast_limit,**probe))
        norms=[]
        quick_records=0
        train_started=time.monotonic()
        total_records=getattr(batch,'total_records',len(every))
        batch_sizes=getattr(batch,'batch_sizes',list(map(len,batch.batches)))
        for subset,physical_size in zip(batch.batches,batch_sizes,strict=True):
            optim.zero_grad(set_to_none=True)
            for i in batch.local(subset,rank):
                row=batch.rows[i]
                retained=k>1 and length(row)<=fast_limit
                set_checkpoint(checkpoint_modules,not retained)
                quick_records+=int(retained)
                logits=score(model,row)
                loss,_=kl_positions(logits,q[i].cuda(),row['mask'].cuda())
                (loss.sum()*batch.weights[i]*(total_records/physical_size)).backward()
            for _,p in params:
                if p.grad is None:
                    p.grad=torch.zeros_like(p)
                dist.all_reduce(p.grad)
            norms.append(float(torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)))
            optim.step()
        set_checkpoint(checkpoint_modules,True)
        train_seconds=time.monotonic()-train_started
        n_quick=torch.tensor(quick_records,device='cuda')
        dist.all_reduce(n_quick)
        steps=prior_steps+len(batch.batches)
        assert {int(st['step'].item()) for st in optim.state.values()}=={steps}
        after_sha=same_hash(params)
        if k==1 and arm in ARMS and (rates is None or rates[arm]==1e-6):
            assert after_sha==read(PREVIOUS/'grid'/f'{arm}_1'/'complete.json')['final_sha256'],(arm,after_sha)
        values=torch.zeros(5,dtype=torch.float64,device='cuda')
        with torch.no_grad():
            for i in local:
                r=batch.rows[i]
                v=components(before[i],score(model,r).cpu(),q[i],r['mask']).sum(0)*batch.weights[i]
                values+=v.cuda()
        dist.all_reduce(values)
        if rank==0:
            storage.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(storage/'adapter',save_embedding_layers=False)
            torch.save(optim.state_dict(),storage/'adam.pt')
            save(path/'optimizer.json',dict(update=k,arm=arm,before_sha=before_sha,after_sha=after_sha,
                before_logits_reused_on_rank0=reused,activation_limit=fast_limit,retained_records=int(n_quick),
                training_loop_seconds=train_seconds,optimizer_steps=len(norms),cumulative_steps=steps,grad_norms=norms,
                minibatch_rows=batch_sizes,scored_minibatch_rows=list(map(len,batch.batches)),minibatch_sha256=digest_value(batch.batches),
                bf16_fixed_target=dict(zip(('kl_before','kl_after','delta_kl','alignment','movement'),values.tolist())),
                adam_sha256=digest_file(storage/'adam.pt'),safetensors_sha256=digest_file(storage/'adapter/adapter_model.safetensors'),
                elapsed_seconds=time.monotonic()-start))
            print(k,arm,'optimized',steps,flush=True)
        dist.barrier()
        del optim
    dist.destroy_process_group()

if __name__=='__main__':train(int(sys.argv[1]))
