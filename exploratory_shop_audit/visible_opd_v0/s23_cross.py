"""Replay each registered update–target pair with an independent FP32 endpoint forward."""
import os
import zipfile
from pathlib import Path
import torch

from .s22_measure import fp32_scores, restore, shard_order
from .s22_report import checkpoint, tsv
from .s21_data import SmokeBatch, summarize
from .s21_store import ARMS, RUN as PREVIOUS, unpack_member
from .c_targets import actor
from exploratory_shop_audit.grpo_forced_audit.common import read, save, digest_file

RUN = PREVIOUS.parent / 'shop_visible_opd_s23_20261120'
EVIDENCE = Path(__file__).parent / 'evidence/section23_20261120'


def run():
    os.umask(0o077)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    target = ARMS[rank]
    batch = SmokeBatch()
    ids = [i for k in range(4) for i in shard_order(batch,k)]
    before, _ = checkpoint('fp32','initial')
    initial = torch.load(PREVIOUS/'original/initial.pt', weights_only=True)
    model = actor(dtype=torch.float32).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    with torch.inference_mode():
        for arm in ARMS:
            origin = PREVIOUS/'grid'/f'{arm}_1'
            with zipfile.ZipFile(origin/'adapter.zip') as archive:
                values = {n:torch.from_numpy(unpack_member(archive,n,initial)) for n,_ in params}
            sha,_ = restore(params,initial,values)
            assert sha == read(origin/'complete.json')['final_sha256']
            del values
            after = fp32_scores(model,batch,ids)
            ref,_ = checkpoint('fp32',f'{arm}_1')
            err = max((v-ref[i]).abs().max().item() for i,v in after.items())
            assert err < .001
            metrics = summarize(batch,before,after,batch.targets(target),range(len(batch.rows)))
            save(RUN/'cross'/f'{arm}_{target}.json', dict(update_arm=arm,target_arm=target,
                 **metrics,fp32_replay_max_abs_error=err,adapter_sha256=sha))
            print(arm,target,metrics['delta_kl'],flush=True)


def report():
    rows = [read(RUN/'cross'/f'{u}_{t}.json') for u in ARMS for t in ARMS]
    diagonal = {r['target_arm']:r['delta_kl'] for r in rows if r['update_arm']==r['target_arm']}
    for r in rows:
        r['ratio_to_target_self'] = r['delta_kl']/diagonal[r['target_arm']]
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    tsv(EVIDENCE/'cross_alignment.tsv',rows)

if __name__=='__main__':
    import sys
    if 'report' in sys.argv: report()
    else: run()
