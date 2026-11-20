"""Reset both AdamW moments and its step counter on the nine designated one-step replays."""
import sys
import time
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from .c_smoke import kl_positions, same_hash, state_hash
from .c_targets import actor
from .d_data import load_batch
from .d_speed import length,set_checkpoint
from .post_d_common import CASES, POST, case_dir, configure, folder, start_at, train_model
from .s21_data import summarize
from .s21_train import score
from .s22_measure import fp32_scores,shard_order
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, digest_value, read, save


def gated():
    gate=read(POST/'snapshot_summary.json')
    assert gate['cells']==40 and gate['passed']


def replay(rank):
    gated()
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==4
    model=train_model()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    checkpoint_modules=[m for m in model.modules() if getattr(m,'gradient_checkpointing',False)]
    for k,arm in CASES:
        dst=case_dir(k,arm)
        if (dst/'optimizer.json').exists():continue
        start=time.monotonic()
        meta=read(folder(k,arm)/'optimizer.json')
        before=start_at(model,k,arm)
        assert before==meta['before_sha']==same_hash(params)
        batch,q=load_batch(k,arm)
        assert digest_value(batch.batches)==meta['minibatch_sha256'] and batch.batch_sizes==meta['minibatch_rows']
        optim=torch.optim.AdamW([p for _,p in params],lr=1e-6,weight_decay=0.)
        assert not optim.state and optim.param_groups[0]['betas']==(.9,.999)
        norms=[]
        for subset,n in zip(batch.batches,batch.batch_sizes,strict=True):
            optim.zero_grad(set_to_none=True)
            for i in batch.local(subset,rank):
                row=batch.rows[i]
                set_checkpoint(checkpoint_modules,length(row)>meta['activation_limit'])
                logits=score(model,row)
                loss,_=kl_positions(logits,q[i].cuda(),row['mask'].cuda())
                (loss.sum()*batch.weights[i]*(batch.total_records/n)).backward()
            for _,p in params:
                if p.grad is None:p.grad=torch.zeros_like(p)
                dist.all_reduce(p.grad)
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
            norms.append(float(norm))
            optim.step()
        set_checkpoint(checkpoint_modules,True)
        assert {int(v['step'].item()) for v in optim.state.values()}=={len(batch.batches)}
        after=same_hash(params)
        if rank==0:
            dst.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(dst/'adapter',save_embedding_layers=False)
            save(dst/'optimizer.json',dict(arm=arm,update=k,before_sha=before,after_sha=after,
                 reset_step_before=0,reset_step_after=len(norms),historical_step_before=meta['cumulative_steps']-len(norms),
                 minibatch_sha256=meta['minibatch_sha256'],minibatch_rows=batch.batch_sizes,
                 target_sha256=read(folder(k,arm)/'target_manifest.json')['target_sha256'],grad_norms=norms,
                 adapter_file_sha256=digest_file(dst/'adapter/adapter_model.safetensors'),seconds=time.monotonic()-start))
            print(k,arm,'reset optimizer completed',flush=True)
        dist.barrier()
        del optim
    dist.destroy_process_group()


@torch.inference_mode()
def measure(rank):
    gated()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=actor(dtype=torch.float32).eval()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    for k,arm in CASES[rank::4]:
        dst=case_dir(k,arm)
        if (dst/'fp32.json').exists():continue
        start=time.monotonic()
        meta=read(dst/'optimizer.json');original=read(folder(k,arm)/'fp32.json')['fixed_target']
        assert start_at(model,k,arm)==meta['before_sha']
        batch,q=load_batch(k,arm);ids=[i for j in range(4) for i in shard_order(batch,j)]
        before=fp32_scores(model,batch,ids)
        adapter=dst/'adapter/adapter_model.safetensors'
        assert digest_file(adapter)==meta['adapter_file_sha256']
        vals=load_file(str(adapter))
        for n,p in params:p.copy_(vals.pop(n.replace('.default.weight','.weight')))
        assert not vals and state_hash(params)==meta['after_sha']
        after=fp32_scores(model,batch,ids)
        metrics=summarize(batch,before,after,q,range(len(q)))
        assert abs(metrics['kl_before']-original['kl_before'])<1e-10
        save(dst/'fp32.json',dict(arm=arm,update=k,reset=metrics,historical=original,
             initial_sha=meta['before_sha'],reset_sha=meta['after_sha'],seconds=time.monotonic()-start))
        print(k,arm,'reset FP32 completed',flush=True)

if __name__=='__main__':
    rank=configure()
    if sys.argv[1]=='train':replay(rank)
    elif sys.argv[1]=='measure':measure(rank)
    else:raise ValueError('Expected train or measure')
