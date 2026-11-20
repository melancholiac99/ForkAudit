"""Post hoc FP32 reanchoring of the original corrections on their frozen supports."""
import torch

from .c_targets import actor
from .d_data import load_batch
from .post_d_common import ARMS, POST, configure, folder, start_at
from .s21_data import summarize
from .s22_measure import fp32_scores, shard_order
from .s22_report import checkpoint
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def reanchor(bf16, fp32, q, mask):
    if (q[~mask]!=0).any() or (q<0).any() or not torch.isfinite(q).all() or not (q.sum(-1)>0).all():
        raise ValueError('invalid original target')
    log_b=bf16.double().masked_fill(~mask,-torch.inf).log_softmax(-1)
    valid=mask & (q>0)
    out=torch.full_like(log_b,-torch.inf)
    out[valid]=fp32[valid].double()+q[valid].double().log()-log_b[valid]
    return out.softmax(-1)


def location(k,arm):
    return POST/'precision'/f'{k:02}'/arm


@torch.inference_mode()
def measure(model,k,arm):
    dst=location(k,arm);meta=dst/'result.json'
    if meta.exists():return
    assert read(POST/'snapshot'/f'{k:02}_{arm}.json')['passed']
    hist=read(folder(k,arm)/'fp32.json')
    batch,q=load_batch(k,arm)
    order=[i for j in range(4) for i in shard_order(batch,j)]
    assert start_at(model,k,arm)==hist['actor_before_sha']
    # The original smoke ran four independent prefix caches. Online steps used one
    # cache per arm across the concatenated, balanced shards.
    if k==1:
        before={}
        for j in range(4):
            before.update(fp32_scores(model,batch,shard_order(batch,j)))
    else:
        before=fp32_scores(model,batch,order)
    if k==1:
        bf16=checkpoint('bf16','initial')[0]
        after=checkpoint('fp32',f'{arm}_1')[0]
    else:
        p=folder(k,arm)/'bf16_before.pt'
        assert digest_file(p)==read(p.parent/'target_manifest.json')['bf16_before_sha256']
        bf16=dict(enumerate(torch.load(p,weights_only=True)))
        assert start_at(model,k+1,arm)==hist['actor_after_sha']
        after=fp32_scores(model,batch,order)
    ids=range(len(q))
    targets=[reanchor(bf16[i],before[i],q[i],batch.rows[i]['mask']) for i in ids]
    cross=summarize(batch,bf16,before,q,ids)['movement']
    same=summarize(batch,bf16,bf16,q,ids)['kl_before']
    mixed=summarize(batch,before,after,q,ids)
    for field in ('kl_before','kl_after','alignment','movement','delta_kl'):
        assert abs(hist['fixed_target'][field]-mixed[field])<1e-10,(k,arm,field,hist['fixed_target'][field],mixed[field])
    anchored=summarize(batch,before,after,targets,ids)
    assert abs(anchored['movement']-mixed['movement'])<1e-12
    dst.mkdir(parents=True,exist_ok=True)
    tape=dst/'target_and_start.pt';torch.save(dict(before=before,targets=targets),tape)
    save(meta,dict(update=k,arm=arm,precision_kl_bf16_to_fp32=cross,bf16_target_strength=same,
         bf16_anchor_evidence='reconstructed' if k==1 else 'stored',original=mixed,reanchored=anchored,
         initial_sha=hist['actor_before_sha'],final_sha=hist['actor_after_sha'],tape_sha=digest_file(tape)))
    print(k,arm,'reanchored',anchored['delta_kl'],flush=True)


if __name__=='__main__':
    rank=configure()
    assert read(POST/'snapshot_summary.json')['passed']
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=actor(dtype=torch.float32).eval()
    for step in range(1,11):measure(model,step,ARMS[rank])
