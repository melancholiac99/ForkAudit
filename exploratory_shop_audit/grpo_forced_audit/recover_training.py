"""Reconstruct uncheckpointed updates in isolation, with exact weight guards."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from .common import DEFAULT_RUN, ROOT, HERE, GPU_PYTHON, SEED, append, digest_file, read, read_rows, save, utc, write_once


def decision_key(row):
    if isinstance(row, str):
        row = json.loads(row)
    return row['trajectory_id'], row['step'], row['task_id']


def replay_indices(source, recorded):
    keys = [decision_key(r) for r in source]
    assert len(set(keys)) == len(keys), 'Unpadded source must have unique decisions'
    lookup = {key: i for i, key in enumerate(keys)}
    target = [decision_key(r) for r in recorded]
    assert set(target) == set(keys), 'Saved order must cover exactly the original decisions'
    padding = (-len(keys)) % 16
    counts = Counter(target)
    assert len(target) == len(keys) + padding
    assert all(n in (1, 2) for n in counts.values()), 'Native padding samples without replacement'
    return [lookup[key] for key in target]


def prepare(run):
    folder = run / 'recovery_0042'
    request_path = folder / 'request.json'
    if request_path.exists():
        return read(request_path)
    metrics = read_rows(run / 'training_metrics.jsonl')
    weights = read_rows(run / 'weight_updates.jsonl')
    assert [r['update'] for r in metrics] == list(range(1, 43))
    assert len(weights) == 42 and all(r['all_ranks_equal'] for r in weights[-3:])
    for a, b in zip(weights[-3:], weights[-2:]):
        assert a['after'] == b['before']
    captures = {}
    for step in [41, 42]:
        paths = list((run / 'alignment_live').glob(f'step_{step:04d}_*_decisions.json'))
        assert len(paths) == 1, (step, paths)
        captures[str(step)] = str(paths[0])
    request = {'time': utc(), 'source_run': str(run), 'from_update': 40, 'through_update': 42,
               'authorization': 'Researcher: 继续训啊; isolate reconstruction, preserve original samples and evidence.',
               'expected_weights': {str(i): weights[i - 1] for i in [41, 42]},
               'decision_files': captures,
               'decision_sha256': {k: digest_file(Path(p)) for k, p in captures.items()},
               'method': 'Replay original cached samples with recorded padding multiplicity and post-balance rank order; require exact before/after trainable-weight hashes at both updates.',
               'limitations': 'Original optimizer states at 41/42 were not saved; original RNG trajectory is not independently recoverable. Matching trainable-weight hashes are necessary, not a separate proof of optimizer-state identity.',
               'mismatch_action': 'Stop isolated reconstruction; do not promote or silently reset the main run.'}
    folder.mkdir(exist_ok=True)
    write_once(request_path, request)
    save(folder / 'status.json', {'status': 'prepared', 'time': utc()})
    return request


def attempt_path(run):
    selection = run / 'recovery_0042/selected_attempt.json'
    name = read(selection)['name'] if selection.exists() else 'attempt_01'
    assert name in ['attempt_01', 'attempt_02']
    return run / 'recovery_0042' / name


def prepare_workspace(run, request):
    target = attempt_path(run)
    if target.exists():
        return target
    target.mkdir(parents=True)
    for name in ['splits.json', 'PROTOCOL.md', 'registration.json', 'preflight_report.json',
                 'pilot_summary.json', 'resource_authorization.json']:
        shutil.copy2(run / name, target / name)
    for name in ['snapshots', 'training_data']:
        shutil.copytree(run / name, target / name, copy_function=os.link)
    for name in ['native_batch_cache', 'rollouts', 'training', 'evaluation']:
        (target / name).mkdir()
    for step in [40, 41, 42]:
        paths = list((run / 'native_batch_cache').glob(f'train_{step:04d}_sample_*.pkl'))
        assert len(paths) == 1
        shutil.copy2(paths[0], target / 'native_batch_cache' / paths[0].name)
        for path in (run / 'rollouts').glob(f'train_{step:04d}_sample_*.jsonl'):
            shutil.copy2(path, target / 'rollouts' / path.name)
    save(target / 'reconstruction.json', request)
    return target


def install():
    """Installed only in the isolated recovery trainer, never normal training."""
    if not os.environ.get('SHOP_RECONSTRUCT_UPDATES'):
        return
    import numpy as np
    import torch
    from verl.trainer.ppo import ray_trainer
    from . import native
    request = read(Path(os.environ.get('SHOP_RECONSTRUCTION_REQUEST', str(Path(os.environ['SHOP_GRPO_RUN']) / 'reconstruction.json'))))
    documented_resume = request.get('researcher_approved_non_bitwise_resume') is True
    if getattr(ray_trainer, '_shop_reconstruction', False):
        return
    ray_trainer._shop_reconstruction = True
    original_adjust = ray_trainer.adjust_batch
    original_balance = ray_trainer.RayPPOTrainer._balance_batch

    def adjust(config, data, mode='copy'):
        step = int(os.environ['SHOP_COMPLETED_UPDATES']) + 1
        if documented_resume and step > 42:
            return original_adjust(config, data, mode)
        assert step in [41, 42], 'Reconstruction must not generate/update beyond frozen steps'
        original_adjust(config, data, mode)  # retain the native driver RNG draw
        path = Path(request['decision_files'][str(step)])
        assert digest_file(path) == request['decision_sha256'][str(step)]
        order = replay_indices(data.non_tensor_batch['shop_decision'], read(path))
        result = data.select_idxs(np.asarray(order))
        result.meta_info['reconstruction_update'] = step
        append(Path(os.environ['SHOP_GRPO_RUN']) / 'replay_order.jsonl',
               {'update': step, 'source_rows': len(data), 'rows': len(result), 'indices': order})
        return result

    def balance(self, batch, metrics, logging_prefix='global_seqlen'):
        if documented_resume and int(os.environ['SHOP_COMPLETED_UPDATES']) >= 42:
            return original_balance(self, batch, metrics, logging_prefix)
        assert 'reconstruction_update' in batch.meta_info, 'Recovery padding hook was bypassed'
        assert self.actor_rollout_wg.world_size == 4
        step = batch.meta_info['reconstruction_update']
        recorded = read(Path(request['decision_files'][str(step)]))
        assert [decision_key(r) for r in batch.non_tensor_batch['shop_decision']] == [decision_key(r) for r in recorded]
        lengths = batch.batch['attention_mask'].sum(-1).reshape(4, -1).sum(-1).tolist()
        metrics[f'{logging_prefix}/dp_max_tokens'] = max(lengths)
        metrics[f'{logging_prefix}/dp_min_tokens'] = min(lengths)

    original_after = native.after_update

    def after(step, metrics):
        if documented_resume and step > 42:
            return original_after(step, metrics)
        run = Path(os.environ['SHOP_GRPO_RUN'])
        observed = read_rows(run / 'weight_updates.jsonl')[-1]
        expected = request['expected_weights'][str(step)]
        passed = observed['all_ranks_equal'] and all(observed[k] == expected[k] for k in ['before', 'after'])
        output = run / 'recovery_0042/continued_checks' if documented_resume else run
        save(output / f'verified_update_{step}.json', {'update': step, 'passed': passed,
             'authorized_non_bitwise_resume': documented_resume, 'observed': observed, 'expected': expected})
        if documented_resume:
            assert observed['all_ranks_equal'] and observed['finite']
            if step == 41:
                assert observed['before'] == expected['before'], 'Resume must start at the exact checkpoint40 weights'
        else:
            assert passed, f'Weight reconstruction mismatch at update {step}; main run unchanged'
        original_after(step, metrics)

    ray_trainer.adjust_batch = adjust
    # native.install compiles fit into a copied globals dictionary. Updating
    # the module alone does not change the function actually executing fit.
    replace_fit_adjust(ray_trainer.RayPPOTrainer.fit, original_adjust, adjust)
    ray_trainer.RayPPOTrainer._balance_batch = balance
    native.after_update = after


def replace_fit_adjust(fit, original, replacement):
    assert fit.__globals__['adjust_batch'] is original
    fit.__globals__['adjust_batch'] = replacement


def promote(run, attempt):
    for step in [41, 42]:
        assert read(attempt / f'verified_update_{step}.json')['passed']
    source = attempt / 'training/checkpoints/global_step_42'
    assert (source / 'data.pt').exists()
    assert read(source / 'actor/replicated_checkpoint.json')['all_rank_fingerprints_equal']
    for kind in ['model', 'optim', 'extra_state']:
        for rank in range(4):
            assert (source / f'actor/{kind}_world_size_4_rank_{rank}.pt').exists()
    target = run / 'training/checkpoints/global_step_42'
    assert not target.exists(), 'Never overwrite a checkpoint'
    shutil.copytree(source, target, copy_function=os.link)
    (run / 'training/checkpoints/latest_checkpointed_iteration.txt').write_text('42')
    save(run / 'recovery_0042/status.json', {'status': 'complete', 'time': utc(), 'checkpoint': str(target),
         'original_updates_preserved': [41, 42], 'reconstructed_weight_hashes_match': True,
         'raw_samples_regenerated': 0, 'attempt': str(attempt),
         'limitation': 'Original optimizer state at 41/42 was unavailable for independent comparison.'})
    from .parallel import prune_full_checkpoints
    prune_full_checkpoints(run, keep=2)


def resume(run):
    from .budget import execute
    from .train import settings, SOURCE_VERL, MIGRATION
    request = prepare(run)
    if read(run / 'recovery_0042/status.json')['status'] == 'complete':
        from .campaign import main
        return main()
    assert not (attempt_path(run) / 'weight_updates.jsonl').exists(), 'Inspect any previous attempt before repeating reconstruction'
    attempt = prepare_workspace(run, request)
    cfg = settings(attempt, updates=42, resume=run / 'training/checkpoints/global_step_40')
    cfg['trainer.test_freq'] = 0
    env = dict(os.environ, VERL_AGENT_SOURCE=str(SOURCE_VERL), SHOP_GRPO_RUN=str(attempt),
               SHOP_RECONSTRUCT_UPDATES='1', FORKAUDIT_VERL_OUTPUT=str(attempt / 'training'),
               FORKAUDIT_MODEL_SEED=str(SEED), FORKAUDIT_ROLLOUT_SESSION='1',
               PYTHONPATH=f'{ROOT}:{MIGRATION}:{SOURCE_VERL}', CUDA_VISIBLE_DEVICES='0,1,2,3',
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               VLLM_ATTENTION_BACKEND='FLASH_ATTN', HYDRA_FULL_ERROR='1',
               RAY_TMPDIR='/tmp/shop_grpo_recovery_0042', SHOP_COMPLETED_UPDATES='40')
    args = [f'{k}=' + ('null' if v is None else json.dumps(v, separators=(',', ':')) if isinstance(v, (list, bool)) else str(v)) for k, v in cfg.items()]
    save(run / 'recovery_0042/launch.json', {'settings': cfg, 'code_sha256': {p.name: digest_file(p) for p in HERE.glob('*.py')}})
    save(run / 'recovery_0042/status.json', {'status': 'reconstructing', 'time': utc(), 'attempt': str(attempt)})
    try:
        execute(run, 'recover_0042', [GPU_PYTHON, '-m', __package__ + '.launch', *args], env)
        promote(run, attempt)
    except BaseException:
        import traceback
        save(run / 'recovery_0042/status.json', {'status': 'failed', 'time': utc(), 'attempt': str(attempt), 'error': traceback.format_exc()})
        raise
    from .campaign import main
    main()


def prepare_authorized_continuation(run):
    """Archive failed-tail history before the explicitly approved checkpoint reset."""
    folder = run / 'recovery_0042'
    authorization = folder / 'continuation_authorization.json'
    assert not authorization.exists(), 'Do not reset a continuation that has already started'
    assert read(folder / 'attempt_02/verified_update_41.json')['passed'] is False
    assert not list((run / 'active_gpu_sessions').glob('*.json'))
    assert [r['update'] for r in read_rows(run / 'training_metrics.jsonl')] == list(range(1, 43))
    archive = folder / 'pre_continuation_archive'
    assert not archive.exists()
    archive.mkdir()
    manifest = {}
    for name in ['training_metrics.jsonl', 'weight_updates.jsonl', 'training_signal.jsonl',
                 'advantage_checks.jsonl', 'alignment_live/metrics.jsonl']:
        path = run / name
        if not path.exists():
            continue
        rows = read_rows(path)
        kept = rows[:40] if name == 'weight_updates.jsonl' else [r for r in rows if r['update'] <= 40]
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest[name] = {'sha256': digest_file(path), 'archived_path': str(destination),
                          'original_rows': len(rows), 'retained_canonical_rows': len(kept)}
        path.rename(destination)
        for row in kept:
            append(path, row)
        assert digest_file(destination) == manifest[name]['sha256']
    request = read(folder / 'request.json')
    for step in [41, 42]:
        for path in (run / 'alignment_live').glob(f'step_{step:04d}_*'):
            dest = archive / 'alignment_live' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            manifest[str(path.relative_to(run))] = {'sha256': digest_file(path), 'archived_path': str(dest)}
            path.rename(dest)
        path = run / f'training/rollouts/{step}.jsonl'
        if path.exists():
            dest = archive / 'training/rollouts' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            manifest[str(path.relative_to(run))] = {'sha256': digest_file(path), 'archived_path': str(dest)}
            path.rename(dest)
        request['decision_files'][str(step)] = str(archive / 'alignment_live' / Path(request['decision_files'][str(step)]).name)
    request.update(researcher_approved_non_bitwise_resume=True,
                   authorization_text='从第40步恢复并继续训练（推荐）', authorization_time=utc(),
                   mismatch_action='Retain and report mismatch; researcher approved restarting from checkpoint40 with saved batches.',
                   method='Documented checkpoint resume with original cached samples and original padding/order for 41/42; no claim of exact original-state reconstruction.')
    save(folder / 'continued_replay_request.json', request)
    save(archive / 'manifest.json', manifest)
    save(authorization, {'researcher_approved': True, 'time': utc(), 'from_update': 40,
         'canonical_logs_prepared': True, 'archived_completed_updates': [41, 42],
         'original_samples_preserved': True, 'reuse_cached_sampling_updates': [41, 42, 43],
         'non_bitwise_resume': True, 'archive': str(archive),
         'note': 'Updates41/42 are replayed from checkpoint40, not duplicated on top of checkpoint42. Original generation evidence for all three batches is retained; later two batches originated from the pre-failure model lineage.'})


if __name__ == '__main__':
    prepare(DEFAULT_RUN.resolve())
