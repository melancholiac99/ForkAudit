"""Reconcile the native GRPO per-update rows with the persisted Adam state on every rank."""
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import torch
from .s21_store import RUN
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, digest_file, read, read_rows, save


def verify():
    src=ROOT/'.runtime/runs/shop_grpo_forced_20261117_01'
    checks=defaultdict(set)
    for v in read_rows(src/'advantage_checks.jsonl'):
        checks[v['update']].add(v['rows'])
    assert set(checks)==set(range(1,101)) and all(len(v)==1 for v in checks.values())
    counter={};settings={}
    for t in (1,100):
        config=read(src/f'train_launch_{t:04d}.json')['settings']
        settings[t]={k:config[k] for k in ('actor_rollout_ref.actor.ppo_mini_batch_size',
            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu','actor_rollout_ref.actor.ppo_epochs',
            'data.train_batch_size','env.rollout.n','trainer.n_gpus_per_node')}
        assert settings[t]['actor_rollout_ref.actor.ppo_mini_batch_size']==64 and settings[t]['actor_rollout_ref.actor.ppo_epochs']==1
    for t,entries in sorted(checks.items()):
        n=next(iter(entries))
        with (src/f'training/rollouts/{t}.jsonl').open('rb') as f:
            assert sum(1 for _ in f)==n
        counter[t]=dict(rows=n,optimizer_steps=ceil(n/64))
    states=[]
    for t in (90,100):
        expected=sum(v['optimizer_steps'] for i,v in counter.items() if i<=t)
        for rank in range(4):
            path=src/f'training/checkpoints/global_step_{t}/actor/optim_world_size_4_rank_{rank}.pt'
            v=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
            steps=Counter(int(x['step'].item()) for x in v['state'].values())
            assert set(steps)=={expected}
            keys=('lr','betas','eps','weight_decay')
            states.append(dict(update=t,rank=rank,per_parameter_counts=steps,
                param_groups=[{k:g[k] for k in keys} for g in v['param_groups']],checkpoint=path.name))
    save(RUN/'baseline_optimizer_audit.json',dict(native_settings=settings,global_minibatch_rows=64,
        unit='native action response rows after padding',updates=counter,checkpoints=states,
        histogram=Counter(v['optimizer_steps'] for v in counter.values()),total=sum(v['optimizer_steps'] for v in counter.values()),
        advantage_checks_sha256=digest_file(src/'advantage_checks.jsonl'),
        config_sha256=digest_file(src/'training/hydra/.hydra/config.yaml'),opd_weight_decay=0.0))
if __name__=='__main__':verify()
