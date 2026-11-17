"""Recompute diagnostic summaries exclusively from archived arrays and batches."""
import json
import numpy as np
from .common import DEFAULT_RUN, save
from .alignment_metrics import summarize


def main():
    import torch
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    run=DEFAULT_RUN.resolve();folder=run/'alignment_diagnostic/checkpoint_0002'
    arrays=np.load(folder/'probabilities.npz')
    packet=DataProto.load_from_disk(str(next((run/'native_batch_cache').glob('train_0002_sample_*.pkl'))))
    packet.to('cpu')
    mask=torch.tensor(arrays['mask']);scores=torch.zeros(mask.shape,dtype=torch.float32)
    rewards=torch.tensor(packet.non_tensor_batch['episode_rewards'].astype(float),dtype=torch.float32)
    scores[torch.arange(len(packet)),mask.sum(-1).long()-1]=rewards
    advantages,_=compute_grpo_outcome_advantage(scores,mask,packet.non_tensor_batch['uid'],packet.non_tensor_batch['traj_uid'])
    decisions=[json.loads(str(x)) for x in packet.non_tensor_batch['shop_decision']]
    result=summarize(arrays['actor'],arrays['rollout'],arrays['mask'],advantages.numpy(),
                     [r['trajectory_id'] for r in decisions])
    result['limitations']=['Read-only microbatch1 recomputation, not archived original pre-update logits.',
                          'Advantages recomputed without DP padding duplicates; original update includes native padding.',
                          'ESS and surrogate coefficients diagnose likelihood mismatch, not measured final training performance.']
    save(folder/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
