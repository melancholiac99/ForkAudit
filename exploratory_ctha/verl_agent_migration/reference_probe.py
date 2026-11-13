"""Fixed real-token scoring through native workers; no generation or updates."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def install():
    if not os.environ.get('FORKAUDIT_REFERENCE_PACKET'):
        return
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
    from correction import metrics

    def validate(self):
        assert self.config.trainer.val_only, 'Reference scoring must never train'
        path = Path(os.environ['FORKAUDIT_REFERENCE_PACKET'])
        data = DataProto.load_from_disk(str(path))
        mask = compute_response_mask(data)
        previous = np.load(path.with_name('validation_alignment_0001.npz'))['actor']
        folder = Path(os.environ['FORKAUDIT_VERL_OUTPUT'])
        if os.environ.get('FORKAUDIT_GRADIENT_BENCH_PACKET'):
            real = DataProto.load_from_disk(os.environ['FORKAUDIT_GRADIENT_BENCH_PACKET'])
            ids = real.non_tensor_batch['forkaudit_row_id']
            _, unique_indices = np.unique(ids, return_index=True)
            lengths = real.batch['attention_mask'].sum(-1).cpu().numpy()
            selected = unique_indices[np.argsort(-lengths[unique_indices], kind='stable')[:32]]
            bench = real[selected]
            bench.meta_info['global_token_num'] = bench.batch['attention_mask'].sum(-1).tolist()
            bench.meta_info['multi_turn'] = False
            started = time.time()
            result = self.actor_rollout_wg.update_actor(bench)
            from verl.trainer.ppo.metric_utils import reduce_metrics
            stats = reduce_metrics(result.meta_info['metrics'])
            assert all(np.isfinite(float(stats[k])) for k in ['actor/grad_norm', 'actor/pg_loss'])
            record = dict(rows=len(bench), selected_original_row_ids=ids[selected].tolist(),
                started_unix=started, finished_unix=time.time(), optimizer_updates=0,
                description='32 longest unique real response records; native loss/backward/clipping, AdamW.step disabled and hashes unchanged; not on-policy learning',
                metrics={k: float(v) for k, v in stats.items()})
            (folder / 'gradient_benchmark.json').write_text(json.dumps(record, indent=2))
            # Native update_actor advances its scheduler; restore the entire saved
            # checkpoint before numerical scoring, without a new online update.
            self._load_checkpoint()
        rows = []
        log = {}
        for size in [32, 16, 8, 1]:
            start = time.monotonic()
            values = []
            for left in range(0, len(data), size):
                chunk = data[left:left + size]
                values.append(self.actor_rollout_wg.compute_log_prob(chunk).batch['old_log_probs'])
            actor = torch.cat(values)
            ref = torch.from_numpy(previous)
            old_stats = metrics(actor, ref, mask)
            rollout_stats = metrics(actor, data.batch['rollout_log_probs'], mask)
            elapsed = time.monotonic() - start
            np.savez_compressed(folder / f'reference_batch_{size}.npz',
                actor=actor.float().cpu().numpy(), reference_actor=previous,
                rollout=data.batch['rollout_log_probs'].float().cpu().numpy(), mask=mask.cpu().numpy())
            passed = all(s['sampled_behavior_k3'] <= .02 and s['clip_fraction'] <= .30
                         for s in [old_stats, rollout_stats])
            rows.append(dict(batch_size=size, seconds=elapsed, passed=passed,
                versus_original_GPU_HF=old_stats, versus_original_vllm_raw_scores=rollout_stats))
            log[f'probe/batch_{size}/k3'] = old_stats['sampled_behavior_k3']
            log[f'probe/batch_{size}/clip'] = old_stats['clip_fraction']
        result = dict(protocol='User-approved statistical consistency; trajectory differences separate',
            source_packet=str(path), rows=len(data), token_count=int(mask.sum()),
            generating_new_responses=False, optimizer_updates=0,
            limitation='Teacher-forced raw model-score comparison on fixed real tokens; not greedy-policy importance sampling or exact trajectory reproduction',
            checks=rows, passed=all(r['passed'] for r in rows))
        if self.config.actor_rollout_ref.rollout.name == 'vllm':
            # Two real contexts per fixed task; this is a numerical probe, without
            # an environment transition, reward, or optimizer update.
            selected = []
            for uid in dict.fromkeys(data.non_tensor_batch['traj_uid']):
                selected.extend(np.flatnonzero(data.non_tensor_batch['traj_uid'] == uid)[:2].tolist())
            chosen = data[np.array(selected)]
            prompt_ids = chosen.batch['prompts']
            width = prompt_ids.shape[1]
            prompts = DataProto.from_dict(tensors=dict(input_ids=prompt_ids,
                attention_mask=chosen.batch['attention_mask'][:, :width],
                position_ids=chosen.batch['position_ids'][:, :width]),
                meta_info=dict(do_sample=True, validate=False))
            generated = self.actor_rollout_wg.generate_sequences(prompts)
            generated.save_to_disk(str(folder / 'fresh_eager_batch.pkl'))
            actual = self.actor_rollout_wg.compute_log_prob(generated).batch['old_log_probs']
            sample_mask = compute_response_mask(generated)
            scores = metrics(actual, generated.batch['rollout_log_probs'], sample_mask)
            np.savez_compressed(folder / 'fresh_eager_alignment.npz', actor=actual.float().cpu().numpy(),
                rollout=generated.batch['rollout_log_probs'].float().cpu().numpy(), mask=sample_mask.cpu().numpy())
            result['fresh_real_context_sampling'] = scores
            result['passed'] &= scores['sampled_behavior_k3'] <= .02 and scores['clip_fraction'] <= .30
        (folder / 'reference_probe.json').write_text(json.dumps(result, indent=2) + '\n')
        assert result['passed'], 'Fixed-token statistical consistency gate failed'
        return log

    RayPPOTrainer._validate = validate
