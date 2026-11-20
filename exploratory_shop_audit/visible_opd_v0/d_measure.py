"""FP32 current-batch endpoint decomposition and original fixed-support probes."""
import os
import sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from .c_targets import actor
from .c_smoke import state_hash
from .d_data import folder, load_batch, snapshot
from .s21_data import SmokeBatch,summarize
from .s21_store import ARMS, RUN as SECTION21
from .s21_train import score
from .s22_measure import fp32_scores,shard_order,restore
from .s22_report import checkpoint
from exploratory_shop_audit.grpo_forced_audit.common import read, save


@torch.inference_mode()
def measure(k,arm):
    path=folder(k,arm)
    if (path/'fp32.json').exists():return
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=actor(dtype=torch.float32).eval()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    batch,q=load_batch(k,arm)
    probes=SmokeBatch()
    meta=read(path/'optimizer.json')
    ids=[i for j in range(4) for i in shard_order(batch,j)]
    def endpoint(step,sha):
        params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if step==0:
            # PEFT initialization depends on the base dtype. Use the exact archived BF16-run LoRA.
            initial=torch.load(SECTION21/'original/initial.pt',map_location='cpu',weights_only=True)
            actual,_=restore(params,initial,initial)
        else:
            snapshot(model,step,arm)
            actual=state_hash(params)
        assert actual==sha,(step,arm,actual,sha)
        # Update one replayed the same five steps byte-for-byte, including its current-batch endpoints.
        logits=checkpoint('fp32','initial' if step==0 else f'{arm}_1')[0] if k==1 else fp32_scores(model,batch,ids)
        with sdpa_kernel(SDPBackend.MATH):
            p={i:score(model,probes.rows[i]).cpu() for i in probes.probes}
        return logits,p
    before,p0=endpoint(k-1,meta['before_sha'])
    after,p1=endpoint(k,meta['after_sha'])
    if k>1:
        last=torch.load(folder(k-1,arm)/'probes.pt',weights_only=True)['after']
        assert all(torch.equal(p0[i],last[i]) for i in p0)
    torch.save(dict(before=p0,after=p1),path/'probes.pt')
    metrics=summarize(batch,before,after,q,range(len(q)))
    p=summarize(probes,p0,p1,probes.targets(arm),probes.probes)
    save(path/'fp32.json',dict(precision='fp32',number=k,fixed_target=metrics,
         probe_movement=p['movement'],actor_before_sha=meta['before_sha'],actor_after_sha=meta['after_sha']))
    print(k,arm,'FP32',metrics['delta_kl'],flush=True)

if __name__=='__main__':
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    measure(int(sys.argv[1]),ARMS[rank])
