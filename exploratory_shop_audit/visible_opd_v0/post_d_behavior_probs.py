"""Compare full-vocabulary T=1 log probabilities along a fixed native behavior batch."""
import contextlib
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from .c_targets import actor, response
from .d_data import PARENT, state_dir, snapshot
from .d_eval import episodes
from .post_d_common import ARMS, POST, RUN, configure, folder
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner

OFFSET=992
MODELS=(*ARMS, 'B')


def native(rank,arm,path):
    if (path/'native.json').exists():return
    splits=read(PARENT/'asin_split_manifest.json')
    jobs=[(pool,t['goal_index'],rep) for pool in ('V','T_opd_eval') for t in splits[pool] for rep in range(5)]
    jobs=jobs[OFFSET:OFFSET+32]
    origin_rank=(OFFSET//32)%4 if arm=='B' else ARMS.index(arm)
    previous=RUN/'evaluation'/arm/f'rank{origin_rank}'/f'episodes_{OFFSET:04}.json'
    key=lambda e:(e['split'],e['goal'],e['rep'])
    old={key(e):e for e in read(previous)}
    assert set(old)==set(jobs)
    if not (path/'snapshots').exists():(path/'snapshots').symlink_to(PARENT/'snapshots',target_is_directory=True)
    os.environ.update(SHOP_GPU_IDS=str(rank),OPD_RAW_LOGPROBS='1',OPD_ENTROPY_TAPE=str(path/'entropy.bin'))
    checkpoint='base' if arm=='B' else state_dir(10,arm)/'adapter'
    if arm!='B':assert digest_file(checkpoint/'adapter_model.safetensors')==read(folder(10,arm)/'optimizer.json')['safetensors_sha256']
    runner=Runner(path,checkpoint)
    trace=[]
    try:out=episodes(runner,jobs,trace)
    finally:runner.close()
    matches={key(e):old[key(e)]['actions_sha']==e['actions_sha'] for e in out}
    assert matches.keys()==old.keys() and len(trace)==sum(e['length'] for e in out)
    for row in trace:
        row['archived_actions_match']=matches[key(row)]
        assert len(row['answer'])==len(row['native'])==len(row['classes']) and np.isfinite(row['native']).all()
    torch.save(trace,path/'native.pt')
    save(path/'native.json',dict(model=arm,episodes=len(jobs),archived_actions_matched=sum(matches.values()),
         snapshot_sha=runner.checkpoint_hash,archive_sha=digest_file(previous),trace_sha=digest_file(path/'native.pt')))


@torch.inference_mode()
def hf(arm,path):
    meta=read(path/'native.json')
    assert digest_file(path/'native.pt')==meta['trace_sha']
    trace=torch.load(path/'native.pt',weights_only=True,map_location='cpu')
    for precision,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
        dst=path/f'{precision}.json'
        if dst.exists():continue
        model=actor(dtype=dtype).eval()
        if arm!='B':snapshot(model,10,arm)
        groups=defaultdict(list)
        with model.disable_adapter() if arm=='B' else contextlib.nullcontext():
            for e in trace:
                _,raw=response(model,e['prefix'],e['answer'])
                token=torch.tensor(e['answer'],device=raw.device)[:,None]
                logp=raw.log_softmax(-1).gather(1,token).squeeze(1).double().cpu().numpy()
                diff=logp-np.asarray(e['native'],dtype=np.float64)
                assert np.isfinite(diff).all()
                for cls,value in zip(e['classes'],diff,strict=True):
                    for key in [('ALL','ALL','ALL'),(e['split'],'greedy' if e['rep']==4 else 'sample',cls)]:
                        groups[(*key,'all_replay')].append(value)
                        if e['archived_actions_match']:groups[(*key,'archived_action_match')].append(value)
        rows=[]
        for (pool,decode,cls,subset),values in sorted(groups.items()):
            x=np.array(values);z=np.abs(x)
            rows.append(dict(model=arm,precision=precision,pool=pool,decode=decode,token_class=cls,subset=subset,
                 tokens=len(x),mean_hf_minus_vllm=float(x.mean()),mae=float(z.mean()),
                 rmse=float(np.sqrt(np.mean(x*x))),median_abs=float(np.median(z)),
                 p95_abs=float(np.quantile(z,.95)),max_abs=float(z.max())))
        save(dst,dict(metadata=meta,rows=rows))
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':
    rank=configure()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    for arm in (ARMS[rank], *(['B'] if rank==3 else [])):
        path=POST/'behavior_logprobs'/arm;path.mkdir(parents=True,exist_ok=True)
        if sys.argv[1]=='native':native(rank,arm,path)
        elif sys.argv[1]=='hf':hf(arm,path)
        else:raise ValueError('Expected native or hf')
