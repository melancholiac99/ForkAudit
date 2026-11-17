"""Reuse completed native batches without recollecting or choosing retry outcomes."""
from collections import defaultdict
import json
from .common import *


def batch_identity(gen_batch, envs, is_train):
    tasks=[dict(t) for t in gen_batch.non_tensor_batch['env_kwargs']]
    if is_train:
        tasks=[t for t in tasks for _ in range(envs.config.env.rollout.n)]
    counts=defaultdict(int);keys=[]
    for t in tasks:
        tid=t['task_id'];rep=counts[tid];counts[tid]+=1
        keys.append(f'{envs.stage}:{envs.checkpoint}:{envs.decode}:{tid}:{rep}')
    return tasks,keys


def save_batch(data,gen_batch,envs,is_train):
    path=envs._shop_batch_cache
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    # DataProto.__getstate__ replaces its batch with a consolidated, locked
    # TensorDict. Serialize a separate container so caching cannot lock the
    # live batch before the trainer adds response_mask (even without padding).
    # Share tensor storage; this synchronous write changes no tensor values.
    snapshot=type(data)(batch=data.batch.clone(recurse=False),
                        non_tensor_batch=data.non_tensor_batch,meta_info=data.meta_info)
    snapshot.save_to_disk(str(tmp));tmp.replace(path)


def restore_batch(collector,gen_batch,envs,is_train):
    """Prefer exact serialized tensors; bootstrap old completed JSON-only batches.

    Native PPO recomputes old actor log probabilities before every update. The
    historical JSON-only path omits unavailable rollout log-prob diagnostics;
    it never fabricates those values or changes the training likelihoods.
    """
    import torch
    import numpy as np
    from verl.protocol import DataProto
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import get_response_mask
    run=Path(os.environ['SHOP_GRPO_RUN'])
    tasks,keys=batch_identity(gen_batch,envs,is_train)
    path=run/'native_batch_cache'/f'{envs.stage}_{envs.checkpoint:04d}_{envs.decode}_{digest_value(keys)}.pkl'
    envs._shop_batch_cache=path
    if path.exists():
        result=DataProto.load_from_disk(str(path))
        append(run/'native_cache_reuse.jsonl',{'time':utc(),'path':str(path),'source':'exact_tensors','episodes':len(keys)})
        return result
    stem=f'{envs.stage}_{envs.checkpoint:04d}_{envs.decode}'
    episodes={r['trajectory_id']:r for r in read_rows(run/'rollouts'/f'{stem}_episodes.jsonl')}
    selected=set(keys)&episodes.keys()
    if not selected:return None
    assert selected==set(keys),'Partial native batch requires recovery; do not resample completed trajectories'
    all_steps=read_rows(run/'rollouts'/f'{stem}_steps.jsonl')
    tokens={(r['trajectory_id'],r['step']):r for r in read_rows(run/'rollouts'/f'{stem}_tokens.jsonl')}
    by_id=defaultdict(list)
    for row in all_steps:
        if row['trajectory_id'] in selected:by_id[row['trajectory_id']].append(row)
    records=[];pad=collector.tokenizer.pad_token_id;prompt_len=envs.config.data.max_prompt_length;response_len=envs.config.data.max_response_length
    for i,key in enumerate(keys):
        rows=sorted(by_id[key],key=lambda r:r['step']);ep=episodes[key]
        assert len(rows)==ep['length'] and [r['step'] for r in rows]==list(range(ep['length']))
        trajectory=[]
        for row in rows:
            assert row['seed']==stable_seed(SEED,key,row['step'])
            assert row['interface_version']==INTERFACE and row['checkpoint']==envs.checkpoint
            response_ids=tokens[(key,row['step'])]['response_token_ids'];prompt_ids=row['input']['token_ids']
            assert len(response_ids)<=response_len and len(prompt_ids)<=prompt_len
            assert collector.tokenizer.decode(response_ids,skip_special_tokens=True)==row['generation_text']
            prompts=torch.full((prompt_len,),pad,dtype=torch.long);prompts[-len(prompt_ids):]=torch.tensor(prompt_ids)
            pmask=torch.zeros(prompt_len,dtype=torch.long);pmask[-len(prompt_ids):]=1
            responses=torch.full((response_len,),pad,dtype=torch.long);responses[:len(response_ids)]=torch.tensor(response_ids)
            rmask=get_response_mask(responses[None],eos_token=collector.tokenizer.eos_token_id,dtype=torch.long)[0]
            assert int(rmask.sum())==len(response_ids)
            positions=compute_position_id_with_mask(pmask[None])[0]
            trajectory.append({'prompts':prompts,'responses':responses,'input_ids':torch.cat([prompts,responses]),
                'attention_mask':torch.cat([pmask,rmask]),'position_ids':torch.cat([positions,positions[-1]+torch.arange(1,response_len+1)]),
                'anchor_obs':row['before']['observation'],'index':i,'data_source':'shop_reward_v3',
                'shop_decision':json.dumps({'trajectory_id':key,'step':row['step'],'task_id':row['task_id']}),
                'uid':f'{envs.stage}:{envs.checkpoint}:{i//envs.config.env.rollout.n}',
                'traj_uid':key,'is_action_valid':np.bool_(row['legal']),'rewards':row['reward'],'active_masks':True})
        records.append(trajectory)
    rewards=np.asarray([sum(r['reward'] for r in by_id[k]) for k in keys],dtype=np.float32)
    lengths=np.asarray([episodes[k]['length'] for k in keys],dtype=np.float32)
    success={'success_rate':[episodes[k]['strict_success'] for k in keys]}
    data=collector.gather_rollout_data(records,rewards,lengths,success,np.asarray(keys,dtype=object),np.zeros(len(keys),dtype=np.float32))
    save_batch(data,gen_batch,envs,is_train)
    append(run/'native_cache_reuse.jsonl',{'time':utc(),'path':str(path),'source':'saved_exact_prompt_response_tokens',
        'episodes':len(keys),'missing_diagnostic':'rollout_log_probs; native actor old_log_probs recomputed normally'})
    return data
