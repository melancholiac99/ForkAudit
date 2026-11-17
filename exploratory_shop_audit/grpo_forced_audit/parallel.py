"""Experiment-local replicated DP with independent TP1 rollout engines.

Keep global minibatches and four-row microbatches intact across DP ranks.
The frozen base fits on each A100; only LoRA gradients require synchronization.
"""
import functools
import inspect
import json
import os
from pathlib import Path
import textwrap


def partitions(lengths, world, mini=64, micro=4):
    """Balance whole microbatches within each original global minibatch."""
    assert mini % (world * micro) == 0 and len(lengths) % (world * micro) == 0
    ranks = [[] for _ in range(world)]
    for start in range(0, len(lengths), mini):
        groups = [list(range(i, min(i + micro, len(lengths))))
                  for i in range(start, min(start + mini, len(lengths)), micro)]
        capacity = len(groups) // world
        local = [[] for _ in range(world)]
        loads = [0] * world
        for group in sorted(groups, key=lambda g: (-sum(lengths[i] for i in g), g[0])):
            rank = min((r for r in range(world) if len(local[r]) < capacity), key=lambda r: (loads[r], r))
            local[rank].append(group)
            loads[rank] += sum(lengths[i] for i in group)
        for rank in range(world):
            ranks[rank].extend(i for group in sorted(local[rank], key=lambda g: g[0]) for i in group)
    return ranks


def begin(worker):
    from rollout_session import state, status
    assert worker.rollout_sharding_manager.tp_size == 1 and not worker._is_offload_param
    state(worker).begin()
    return status(worker)


def control(group, fn):
    records = group.execute_all_sync('actor_rollout_execute_with_func_generator', fn)
    assert len(records) == group.world_size
    for name in ['open', 'resident', 'poisoned', 'enters', 'exits']:
        assert len({r[name] for r in records}) == 1, f'DP rollout sessions disagree: {name}'
    return {**records[0], 'world_size': group.world_size, 'ranks': records}


def install_collector():
    import torch
    import rollout_session
    from agent_system.multi_turn_rollout import rollout_loop
    from verl.trainer.ppo import ray_trainer
    cls = rollout_loop.TrajectoryCollector
    if getattr(cls, '_shop_parallel', False):
        return
    cls._shop_parallel = True
    # meta_info is replicated, whereas non_tensor_batch is sliced with requests.
    source = textwrap.dedent(inspect.getsource(cls.vanilla_multi_turn_loop))
    anchor = 'non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]'
    assert source.count(anchor) == 1
    source = source.replace(anchor, 'non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "shop_seed"]')
    scope = dict(rollout_loop.__dict__)
    exec(compile(source, '<shop_dp_seed_dispatch>', 'exec'), scope)
    cls.vanilla_multi_turn_loop = scope['vanilla_multi_turn_loop']
    rollout_session.begin, rollout_session.control = begin, control
    rollout_session.install_collector()

    def balance(self, batch, metrics, logging_prefix='global_seqlen'):
        lengths = batch.batch['attention_mask'].sum(-1).tolist()
        groups = partitions(lengths, self.actor_rollout_wg.world_size)
        batch.reorder(torch.tensor([i for group in groups for i in group]))
        totals = [sum(lengths[i] for i in g) for g in groups]
        metrics[f'{logging_prefix}/dp_max_tokens'] = max(totals)
        metrics[f'{logging_prefix}/dp_min_tokens'] = min(totals)
    ray_trainer.RayPPOTrainer._balance_batch = balance


def install_worker():
    from torch.distributed.fsdp import ShardingStrategy
    from verl.workers import fsdp_workers
    fsdp_workers.get_sharding_strategy = lambda mesh: ShardingStrategy.NO_SHARD
    worker = fsdp_workers.ActorRolloutRefWorker
    if not getattr(worker, '_shop_no_shard_cleanup', False):
        # Upstream manually frees sharded full parameters after logprob. A
        # replicated NO_SHARD handle owns persistent full parameters instead.
        for name, model in [('compute_log_prob', 'self.actor.actor_module'),
                            ('compute_ref_log_prob', 'self.ref_policy.actor_module')]:
            source = textwrap.dedent(inspect.getsource(getattr(worker, name)))
            anchor = f'if self.world_size > 1 and fsdp_version({model}) == 1:'
            assert source.count(anchor) == 1
            source = source.replace(anchor, anchor[:-1] + f' and {model}._handle.uses_sharded_strategy:')
            scope = dict(fsdp_workers.__dict__)
            exec(compile(source, '<shop_no_shard_logprob_cleanup>', 'exec'), scope)
            setattr(worker, name, scope[name])
        worker._shop_no_shard_cleanup = True
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager as cls
    if getattr(cls, '_shop_replicated_checkpoint', False):
        return
    cls._shop_replicated_checkpoint = True
    original = cls.save_checkpoint

    @functools.wraps(original)
    def save(self, local_path, *args, **kwargs):
        import torch
        import torch.distributed as dist
        from checkpoint_evidence import fingerprint
        assert self.model.sharding_strategy == ShardingStrategy.NO_SHARD
        if not local_path:
            return original(self, local_path, *args, **kwargs)
        record = fingerprint(self)
        records = [None] * self.world_size
        dist.all_gather_object(records, record)
        assert all(r == record for r in records), 'Replicated checkpoint tensors diverged'
        # Retain the native serializer and per-rank RNG files. Deduplicate only
        # byte-equivalent model/optimizer tensors verified above, on shared disk.
        real_save = torch.save
        duplicated = {str(Path(local_path) / f'{kind}_world_size_{self.world_size}_rank_{self.rank}.pt')
                      for kind in ['model', 'optim']} if self.rank else set()
        def save_unique(value, path, *a, **kw):
            if str(path) not in duplicated:
                return real_save(value, path, *a, **kw)
        torch.save = save_unique
        try:
            result = original(self, local_path, *args, **kwargs)
        finally:
            torch.save = real_save
        if self.rank:
            for kind in ['model', 'optim']:
                target = Path(local_path) / f'{kind}_world_size_{self.world_size}_rank_{self.rank}.pt'
                source = Path(local_path) / f'{kind}_world_size_{self.world_size}_rank_0.pt'
                os.link(source, target)
        dist.barrier()
        if self.rank == 0:
            (Path(local_path) / 'replicated_checkpoint.json').write_text(json.dumps({
                'world_size': self.world_size, 'all_rank_fingerprints_equal': True,
                'model_optimizer_storage': 'rank0 full tensors, hardlinked rank aliases',
                'rng_storage': 'independent native extra_state per rank'}, indent=2))
        return result
    cls.save_checkpoint = save


def prepare_restore(source, world):
    """Add portable NO_SHARD aliases without changing the original checkpoint."""
    source = Path(source)
    actor = source / 'actor'
    if (actor / 'replicated_checkpoint.json').exists():
        assert json.loads((actor / 'replicated_checkpoint.json').read_text())['world_size'] == world
        return source
    assert (actor / 'model_world_size_1_rank_0.pt').exists()
    for rank in range(world):
        for kind in ['model', 'optim', 'extra_state']:
            original=actor / f'{kind}_world_size_1_rank_0.pt'
            target=actor / f'{kind}_world_size_{world}_rank_{rank}.pt'
            if target.exists():assert os.path.samefile(original,target)
            else:os.link(original,target)
        if rank:
            target=actor / f'live_fingerprint_rank_{rank}.json'
            if target.exists():assert os.path.samefile(actor/'live_fingerprint_rank_0.json',target)
            else:os.link(actor / 'live_fingerprint_rank_0.json',target)
    (actor / 'dp_migration.json').write_text(json.dumps({
        'source_world_size': 1, 'target_world_size': world,
        'note': 'Identical full tensors/optimizer/scheduler; initial RNG cloned, future decision seeds remain per request.'}, indent=2))
    return source


def prune_full_checkpoints(run, keep=2):
    """Enforce native keep-two retention across separately launched phases."""
    from .common import save, utc
    folders=sorted((Path(run)/'training/checkpoints').glob('global_step_*'),key=lambda p:int(p.name.split('_')[-1]))
    complete=[p for p in folders if (p/'actor/live_fingerprint_rank_0.json').exists()
              and (p/'actor/lora_adapter/adapter_model.safetensors').exists()
              and list((p/'actor').glob('model_world_size_*_rank_0.pt'))]
    for folder in complete[:-keep]:
        actor=folder/'actor'
        files=[p for kind in ['model','optim','extra_state'] for p in actor.glob(f'{kind}_world_size_*_rank_*.pt')]
        save(folder/'full_checkpoint_retention.json',{'time':utc(),'reason':'Configured max_actor_ckpt_to_keep=2 across process restarts',
            'removed_files':[p.name for p in files],'retained':'LoRA adapter, fingerprints, dataloader state and all measurements'})
        for path in files:path.unlink()
