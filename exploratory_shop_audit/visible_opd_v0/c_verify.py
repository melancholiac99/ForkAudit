"""Verify smoke evidence against the immutable Stage B registration and saved native batch."""
from collections import Counter
import math
from pathlib import Path

import torch
from .c_targets import ARMS, RUN, SEED, STAGE_B, tasks
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, digest_value, read, read_rows, save, stable_seed

def verify():
    published=Path(__file__).parent/'evidence/content_b_20261120'
    acceptance=read(published/'acceptance.json')
    lineage=read(published/'lineage.json')
    for name,sha in acceptance['reports'].items():
        assert digest_file(STAGE_B/'reports'/f'{name}.json')==sha==digest_file(published/f'{name}.json')
    for name,sha in lineage['projectors'].items():
        assert digest_file(STAGE_B/'projector_versions'/name)==sha
    report=read(RUN/'smoke_report.json')
    eligible={r['goal_index'] for r in tasks()}
    lookup={};terminal=Counter();length=Counter()
    for rep in range(2):
        for row in read_rows(RUN/f'rollout_{rep}.jsonl'):
            g=row['goal'];t=row['step'];key=g,rep
            assert g in eligible and row['rep']==rep and length[key]==t and not terminal[key]
            assert row['seed']==stable_seed(SEED,'T_opd_smoke',g,rep,t)
            length[key]+=1;terminal[key]+=row['done'];lookup[g,rep,t]=row
    assert len(length)==64 and set(terminal.values())=={1}
    count=Counter();files={}
    for rank in range(4):
        shard=RUN/f'batch_{rank}'
        rows=torch.load(shard/'student.pt',weights_only=True)
        files[f'batch_{rank}/student.pt']=digest_file(shard/'student.pt')
        assert digest_value([(r['goal'],r['rep'],r['step'],r['ids'].tolist(),r['mask'].tolist()) for r in rows])==report['shards'][rank]['support_sha256']
        targets={arm:torch.load(shard/f'{arm}.pt',weights_only=True) for arm in ARMS}
        for arm in ARMS:
            assert len(targets[arm])==len(rows)
            files[f'batch_{rank}/{arm}.pt']=digest_file(shard/f'{arm}.pt')
        for i,r in enumerate(rows):
            raw=lookup.pop((r['goal'],r['rep'],r['step']))
            assert (r['prompt'],r['answer'],r['classes'])==(raw['views']['none'],raw['response'],raw['classes'])
            assert all(not v for v in raw['na'].values())
            ids=r['ids'];mask=r['mask'];snap=r['snapshot']
            assert torch.all(((ids==torch.tensor(r['answer'])[:,None])&mask).sum(1)==1)
            assert torch.equal(targets['N'][i],snap)
            for arm in ARMS:
                t=targets[arm][i]
                assert t.shape==mask.shape and not t.requires_grad and not t[~mask].any()
                assert (t>=0).all() and torch.allclose(t.sum(1),torch.ones(len(t)),atol=1e-5)
            count.update(r['classes'])
    assert not lookup and sum(count.values())==report['positions']
    init={report['arms'][a]['initial_adapter_sha256'] for a in ARMS}
    assert len(init)==1
    for a in ARMS:
        v=report['arms'][a]
        assert v['epochs']==v['optimizer_steps']==1 and v['synchronized_ranks']==4
        assert math.isfinite(v['gradient_norm'])
        assert v['before']['max_snapshot_error']<=1e-5
        for when in ('before','after'):
            assert {k:val['n'] for k,val in v[when]['by_class'].items()}==count
            assert math.isfinite(v[when]['task_weighted_kl'])
        if a=='N':
            assert v['gradient_norm']<=1e-5
        else:
            assert v['gradient_norm']>0 and v['final_adapter_sha256'] not in init
    costs=dict(collection=sum(x['gpu_hours'] for x in read_rows(RUN/'cost.jsonl')),
               targets=report['target_gpu_hours'],updates=report['train_gpu_hours'])
    save(RUN/'smoke_acceptance.json',dict(passed=True,section19_cpu_only=True,seed=SEED,
        groups=list(ARMS),batch_tasks=len(eligible),trajectories=len(length),steps=sum(length.values()),
        positions=sum(count.values()),classes=count,replica_count=4,
        staged_tensor_sha256=files,rollouts={f.name:digest_file(f) for f in RUN.glob('rollout_*.jsonl')},
        report_sha256=digest_file(RUN/'smoke_report.json'),frozen_strength_sha256=acceptance['reports']['calibration_strength_report'],
        frozen_p2_sha256=acceptance['reports']['p2_minima'],costs_gpu_hours=costs,total_gpu_hours=sum(costs.values())))

if __name__=='__main__':verify()
