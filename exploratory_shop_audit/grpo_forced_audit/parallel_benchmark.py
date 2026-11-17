"""Fixed-token native GRPO throughput probe; optimizer updates are disabled."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf,open_dict
    from torch.distributed.fsdp import ShardingStrategy
    from verl import DataProto
    from verl.workers import fsdp_workers
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from .common import DEFAULT_RUN,SEED,save
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',dest='run',type=Path,default=DEFAULT_RUN);p.add_argument('--strategy',choices=['replicated','sharded'],default='replicated');args=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    folder=args.run/'parallel_benchmark'/f'{args.strategy}_{world}';folder.mkdir(parents=True,exist_ok=True)
    os.environ.update(FORKAUDIT_VERL_OUTPUT=str(folder),FORKAUDIT_MODEL_SEED=str(SEED))
    os.environ.pop('FORKAUDIT_ROLLOUT_SESSION',None)
    if args.strategy=='replicated':fsdp_workers.get_sharding_strategy=lambda mesh:ShardingStrategy.NO_SHARD
    config=OmegaConf.load(args.run/'resolved_train_config.yaml').actor_rollout_ref
    with open_dict(config):
        config.model.external_lib='checkpoint_evidence'
        config.actor.optim.total_training_steps=100
    started=time.monotonic()
    worker=fsdp_workers.ActorRolloutRefWorker(config,role='actor');worker.init_model()
    source=args.run/'training/checkpoints/global_step_1/actor'
    # Replicated NO_SHARD has the same full tensors and optimizer layout as
    # the original world-size-one checkpoint. Aliases do not duplicate files.
    if args.strategy=='replicated':
        from torch.distributed.fsdp import StateDictType,FullStateDictConfig
        state=torch.load(source/'model_world_size_1_rank_0.pt',map_location='cpu',weights_only=False,mmap=True)
        with torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type(worker.actor_module_fsdp,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False)):
            worker.actor_module_fsdp.load_state_dict(state)
        del state
        worker.actor_optimizer.load_state_dict(torch.load(source/'optim_world_size_1_rank_0.pt',map_location='cpu',weights_only=False,mmap=True))
        extra=torch.load(source/'extra_state_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
        worker.actor_lr_scheduler.load_state_dict(extra['lr_scheduler'])
    import checkpoint_evidence
    if args.strategy=='replicated':
        fingerprint=checkpoint_evidence.fingerprint(worker.checkpoint_manager)
        assert fingerprint==json.loads((source/'live_fingerprint_rank_0.json').read_text()),'Checkpoint global state changed'
    init_seconds=time.monotonic()-started
    # Freeze selection by token lengths and stable row indices, never rewards.
    packet=DataProto.load_from_disk(str(next((args.run/'native_batch_cache').glob('train_0001_sample_*.pkl'))))
    width=packet.batch['responses'].shape[1];mask=packet.batch['attention_mask'][:,-width:]
    scores=torch.zeros_like(mask,dtype=torch.float32)
    rewards=torch.tensor(packet.non_tensor_batch['episode_rewards'].astype(float),dtype=torch.float32)
    scores[torch.arange(len(packet)),mask.sum(-1).long()-1]=rewards
    advantage,_=compute_grpo_outcome_advantage(scores,mask,packet.non_tensor_batch['uid'],packet.non_tensor_batch['traj_uid'])
    packet.batch['response_mask']=mask;packet.batch['advantages']=advantage
    lengths=packet.batch['attention_mask'].sum(-1).cpu().numpy()
    order=np.argsort(lengths,kind='stable');indices=order[np.linspace(0,len(order)-1,64,dtype=int)]
    # Keep four-row microbatches identical across worlds, balanced round-robin.
    microgroups=[indices[i:i+4] for i in range(0,64,4)]
    assigned=np.concatenate(microgroups[rank::world]);data=packet[assigned]
    data.meta_info.update(temperature=.7,micro_batch_size=4,use_dynamic_bsz=False)
    data=data.to(torch.cuda.current_device())
    torch.cuda.synchronize();dist.barrier();timings={}
    begin=time.monotonic();logp=worker.actor.compute_log_prob(data,calculate_entropy=True)[0];torch.cuda.synchronize();timings['actor_logprob']=time.monotonic()-begin
    data.batch['old_log_probs']=logp
    begin=time.monotonic()
    with worker.actor_module_fsdp.disable_adapter():ref=worker.actor.compute_log_prob(data,calculate_entropy=False)[0]
    torch.cuda.synchronize();timings['reference_logprob']=time.monotonic()-begin;data.batch['ref_log_prob']=ref
    if 'rollout_log_probs' in data.batch:
        from correction import metrics
        mismatch=metrics(logp,data.batch['rollout_log_probs'],mask[assigned].to(logp.device))
    else:mismatch=None
    before=checkpoint_evidence.digest({n:p for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad})
    old_step=worker.actor_optimizer.step;worker.actor_optimizer.step=lambda *a,**k:None
    gradient_hashes=[];original_step=worker.actor._optimizer_step
    def observe_step():
        result=original_step()
        gradient_hashes.append(checkpoint_evidence.digest({n:p.grad for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad and p.grad is not None}))
        return result
    worker.actor._optimizer_step=observe_step
    begin=time.monotonic();metrics=worker.actor.update_policy(data);torch.cuda.synchronize();timings['backward_and_clip']=time.monotonic()-begin
    worker.actor_optimizer.step=old_step
    after=checkpoint_evidence.digest({n:p for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad});assert before==after
    record={'rank':rank,'world_size':world,'strategy':args.strategy,'initialization_seconds':init_seconds,
        'timings_seconds':timings,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'optimizer_updates':0,'weights_unchanged':True,'gradient_hashes':gradient_hashes,
        'selected_indices':assigned.tolist(),'total_global_rows':64,'global_microbatch_rows':4*world,'global_minibatch_rows':64,
        'metrics':{k:float(np.mean(v)) for k,v in metrics.items()},'sampler_trainer_mismatch':mismatch}
    save(folder/f'rank_{rank}.json',record)
    all_records=[None]*world;dist.all_gather_object(all_records,record)
    if rank==0:
        assert len({tuple(r['gradient_hashes']) for r in all_records})==1,'Replicas did not receive identical synchronized gradients'
        elapsed={k:max(r['timings_seconds'][k] for r in all_records) for k in timings}
        save(folder/'summary.json',{'world_size':world,'strategy':args.strategy,'ranks':all_records,
            'seconds':elapsed,'total_compute_seconds':sum(elapsed.values()),'compute_gpu_seconds':world*sum(elapsed.values()),
            'gradient_sync_verified':True,'checkpoint_state_verified':args.strategy=='replicated','optimizer_updates':0})
        print('BENCHMARK',json.dumps({'world':world,'seconds':elapsed}),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
