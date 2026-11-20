"""Full weighted, unclipped gradients at successive original pilot start snapshots."""
import hashlib
import time
import torch
from .c_smoke import kl_positions
from .d_data import load_batch
from .d_speed import length,set_checkpoint
from .post_d_common import POST,ARMS,configure,folder,start_at,train_model
from .s21_train import score
from exploratory_shop_audit.grpo_forced_audit.common import digest_file,read,save

def gradient_at(model,k,arm,params,modules):
    dst=POST/'gradients'/arm/f'{k:02}'
    if (dst/'manifest.json').exists():return
    start=time.monotonic()
    original=read(folder(k,arm)/'optimizer.json')
    sha=start_at(model,k,arm);assert sha==original['before_sha']
    batch,q=load_batch(k,arm)
    model.zero_grad(set_to_none=True)
    limit=original.get('activation_limit') or 0
    for i,row in enumerate(batch.rows):
        set_checkpoint(modules,length(row)>limit)
        logits=score(model,row)
        loss,_=kl_positions(logits,q[i].cuda(),row['mask'].cuda())
        (loss.sum()*batch.weights[i]).backward()
    set_checkpoint(modules,True)
    grads={n:p.grad.detach().cpu() for n,p in params}
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in grads.values())
    dst.mkdir(parents=True,exist_ok=True)
    tmp=dst/'gradient.pending'
    torch.save(grads,tmp);tmp.replace(dst/'gradient.pt')
    save(dst/'manifest.json',dict(arm=arm,update=k,weight_sha=sha,records=len(q),
         positions=sum(len(r['answer']) for r in batch.rows),parameters=sum(v.numel() for v in grads.values()),
         gradient_file_sha256=digest_file(dst/'gradient.pt'),seconds=time.monotonic()-start))
    print(k,arm,'fixed-start gradient completed',flush=True)


def geometry(left,right):
    if list(left)!=list(right):raise ValueError('different LoRA coordinate systems')
    cross=aa=bb=0.
    for n in left:
        a,b=left[n].double().flatten(),right[n].double().flatten()
        assert a.shape==b.shape
        cross+=torch.dot(a,b).item();aa+=torch.dot(a,a).item();bb+=torch.dot(b,b).item()
    na,nb=aa**.5,bb**.5
    return dict(dot_fp64=cross,left_norm=na,right_norm=nb,cosine=cross/(na*nb) if na and nb else None)

if __name__=='__main__':
    rank=configure();arm=ARMS[rank]
    assert read(POST/'snapshot_summary.json')['passed']
    model=train_model()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    modules=[m for m in model.modules() if getattr(m,'gradient_checkpointing',False)]
    for k in range(1,11):gradient_at(model,k,arm,params,modules)
