"""Paired frozen-model request-tape benchmark. No environment or optimizer steps."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch


def live_state(worker):
    from checkpoint_evidence import fingerprint
    return fingerprint(worker.checkpoint_manager)


def install():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
    from correction import metrics
    from rollout_session import control, begin, end, status

    def validate(self):
        assert self.config.trainer.val_only and self.config.actor_rollout_ref.rollout.name == 'vllm'
        group=self.actor_rollout_wg
        folder=Path(os.environ['FORKAUDIT_VERL_OUTPUT'])
        path=Path(os.environ['FORKAUDIT_REFERENCE_PACKET'])
        data=DataProto.load_from_disk(str(path))
        groups=[np.flatnonzero(data.non_tensor_batch['traj_uid']==uid) for uid in dict.fromkeys(data.non_tensor_batch['traj_uid'])]
        assert len(groups)==4
        tape=[]; indices=[]
        for step in range(6):
            idx=np.array([g[(step*4+j)%len(g)] for g in groups for j in range(4)])
            x=data[idx];width=x.batch['prompts'].shape[1]
            tape.append(DataProto.from_dict(tensors=dict(input_ids=x.batch['prompts'],
                attention_mask=x.batch['attention_mask'][:,:width],position_ids=x.batch['position_ids'][:,:width]),
                meta_info=dict(do_sample=False,validate=False)))
            indices.append(idx.tolist())
        (folder/'request_tape.json').write_text(json.dumps(dict(source=str(path),indices=indices),indent=2))
        before=control(group,live_state)
        fixed=DataProto.load_from_disk(str(path.parent/'ppo_batch_step_1.pkl'))[:16]
        def loss_check():
            from verl.trainer.ppo.core_algos import compute_policy_loss, agg_loss
            scored=group.compute_log_prob(fixed)
            p=compute_policy_loss(old_log_prob=fixed.batch['old_log_probs'],log_prob=scored.batch['old_log_probs'],
                advantages=fixed.batch['advantages'],response_mask=fixed.batch['response_mask'],cliprange=.2,
                cliprange_low=.2,cliprange_high=.2,clip_ratio_c=3.,loss_agg_mode='token-mean')[0]
            loss=p-.001*agg_loss(scored.batch['entropys'],fixed.batch['response_mask'],'token-mean')
            return float(loss),scored.batch['old_log_probs'].numpy()
        loss_before, scores_before=loss_check()
        group.generate_sequences(tape[0])  # common graph/kernel warmup, outside timings
        blocks=[]; generated_blocks=[]
        for i,mode in enumerate(['legacy','session','session','legacy']):
            counts_before=control(group,status); outputs=[]
            start=time.monotonic()
            if mode=='session':control(group,begin)
            try:
                for prompt in tape:outputs.append(group.generate_sequences(prompt))
            finally:
                if mode=='session':control(group,end)
            elapsed=time.monotonic()-start
            counts_after=control(group,status)
            batch=DataProto.concat(outputs);batch.save_to_disk(str(folder/f'generated_{i}_{mode}.pkl'))
            mask=compute_response_mask(batch)
            scored=group.compute_log_prob(batch).batch['old_log_probs']
            stats=metrics(scored,batch.batch['rollout_log_probs'],mask)
            assert stats['sampled_behavior_k3']<=.02 and stats['clip_fraction']<=.30
            np.savez_compressed(folder/f'probabilities_{i}_{mode}.npz',actor=scored.numpy(),
                rollout=batch.batch['rollout_log_probs'].numpy(),mask=mask.numpy(),responses=batch.batch['responses'].numpy())
            row=dict(mode=mode,seconds=elapsed,tokens=int(mask.sum()),tokens_per_second=int(mask.sum())/elapsed,
                manager_enters=counts_after['enters']-counts_before['enters'],
                manager_exits=counts_after['exits']-counts_before['exits'],
                sync_wake_host_seconds=counts_after['enter_seconds']-counts_before['enter_seconds'],
                sleep_cleanup_host_seconds=counts_after['exit_seconds']-counts_before['exit_seconds'],
                raw_model_score_checks=stats)
            assert row['manager_enters']==row['manager_exits']==(1 if mode=='session' else 6)
            blocks.append(row);generated_blocks.append(batch.batch['responses'].clone())
            (folder/'benchmark_progress.json').write_text(json.dumps(blocks,indent=2))
        sampled=[]
        for mode in ['legacy','session']:
            prompt=tape[0];prompt.meta_info=dict(do_sample=True,validate=False)
            if mode=='session':control(group,begin)
            try:batch=group.generate_sequences(prompt)
            finally:
                if mode=='session':control(group,end)
            actor=group.compute_log_prob(batch).batch['old_log_probs'];mask=compute_response_mask(batch)
            stats=metrics(actor,batch.batch['rollout_log_probs'],mask)
            assert stats['sampled_behavior_k3']<=.02 and stats['clip_fraction']<=.30
            np.savez_compressed(folder/f'sampled_{mode}.npz',actor=actor.numpy(),rollout=batch.batch['rollout_log_probs'].numpy(),mask=mask.numpy())
            batch.save_to_disk(str(folder/f'sampled_{mode}.pkl'));sampled.append(dict(mode=mode,**stats))
        loss_after,scores_after=loss_check();after=control(group,live_state)
        assert before==after,'Frozen benchmark changed weights/AdamW/scheduler'
        assert np.array_equal(scores_before,scores_after) and loss_before==loss_after,'Fixed-data training loss changed'
        result=dict(passed=True,optimizer_updates=0,environment_transitions=0,blocks=blocks,sampled_checks=sampled,
            exact_greedy_responses_equal=all(torch.equal(generated_blocks[0],x) for x in generated_blocks[1:]),
            native_full_response_loss_before=loss_before,native_full_response_loss_after=loss_after,
            weights_optimizer_scheduler_equal=True,live_state=after,
            median_wall_speedup=float(np.median([x['seconds'] for x in blocks if x['mode']=='legacy'])/np.median([x['seconds'] for x in blocks if x['mode']=='session'])),
            limitation='Fixed real prompt tape; no ALFWorld transitions or learning. Host timing sections include native blocking transfers but are not CUDA kernel timings.')
        (folder/'session_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        return {'benchmark/speedup':result['median_wall_speedup']}
    RayPPOTrainer._validate=validate
