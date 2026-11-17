"""Read-only CPU incident verification; run in pinned verl Python from repo root."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from verl.protocol import DataProto

ROOT = Path.cwd()
RUN = ROOT / '.runtime/runs/shop_grpo_forced_20261117_01'
OUT = ROOT / 'exploratory_shop_audit/grpo_forced_audit/evidence/watchdog_1794926457'
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()
def read(path):
    return json.loads(path.read_text())

# Reuse the recorded fingerprint function without importing GPU-install hooks.
source = ROOT / 'exploratory_ctha/verl_agent_migration/checkpoint_evidence.py'
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'digest')
scope = {'torch': torch, 'hashlib': hashlib, 'json': json}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
digest = scope['digest']
checkpoint = RUN / 'training/checkpoints/global_step_40'
actor = checkpoint / 'actor'
manifest = {}
for path in sorted(checkpoint.rglob('*')):
    if path.is_file():
        # Hash hardlinked rank aliases only once.
        st = path.stat()
        cached = next((v for v in manifest.values() if v['inode'] == st.st_ino), None)
        manifest[str(path.relative_to(RUN))] = {'bytes': st.st_size, 'inode': st.st_ino,
                                               'sha256': cached['sha256'] if cached else sha(path)}
expected = read(actor / 'live_fingerprint_rank_0.json')
for rank in range(4):
    assert read(actor / f'live_fingerprint_rank_{rank}.json') == expected
    receipt = read(RUN / f'training/restore_rank_{rank}.json')
    assert receipt['passed'] and receipt['actual'] == expected
    assert Path(receipt['source']).resolve() == actor.resolve()
    assert (actor / f'extra_state_world_size_4_rank_{rank}.pt').stat().st_size > 0
optimizer = torch.load(actor / 'optim_world_size_4_rank_0.pt', map_location='cpu', weights_only=False)
assert digest(optimizer) == expected['optimizer_sha256']
del optimizer
extra = torch.load(actor / 'extra_state_world_size_4_rank_0.pt', map_location='cpu', weights_only=False)
assert digest(extra['lr_scheduler']) == expected['scheduler_sha256']
loader = torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False)
updates = rows(RUN / 'training_metrics.jsonl')
weights = rows(RUN / 'weight_updates.jsonl')
assert [r['update'] for r in updates] == list(range(1,43))
assert len(weights) == 42 and all(r['finite'] for r in weights)
assert all(r['all_ranks_equal'] for r in weights[-2:])
assert all(a['after'] == b['before'] for a,b in zip(weights, weights[1:]))
checks = []
for number in (40,41,42):
    paths = list((RUN / 'native_batch_cache').glob(f'train_{number:04d}_*.pkl'))
    assert len(paths) == 1
    path = paths[0]
    data = DataProto.load_from_disk(str(path))
    stem = RUN / f'rollouts/train_{number:04d}_sample'
    episodes = rows(Path(str(stem) + '_episodes.jsonl'))
    steps = rows(Path(str(stem) + '_steps.jsonl'))
    tokens = rows(Path(str(stem) + '_tokens.jsonl'))
    assert len(episodes) == len({e['trajectory_id'] for e in episodes}) == 128
    assert len(steps) == len(tokens) == len(data) == sum(e['length'] for e in episodes)
    by_token = {(r['trajectory_id'],r['step']):r for r in tokens}
    by_step = {(r['trajectory_id'],r['step']):r for r in steps}
    assert len(by_token) == len(by_step) == len(data)
    for i, text in enumerate(data.non_tensor_batch['shop_decision']):
        decision = json.loads(str(text)); key = (decision['trajectory_id'],decision['step'])
        mask = data.batch['attention_mask'][i].bool()
        response_len = data.batch['responses'].shape[-1]
        assert data.batch['responses'][i][mask[-response_len:]].tolist() == by_token[key]['response_token_ids']
        assert data.batch['prompts'][i][mask[:-response_len]].tolist() == by_step[key]['input']['token_ids']
    for e in episodes:
        ss = sorted((s for s in steps if s['trajectory_id'] == e['trajectory_id']),key=lambda s:s['step'])
        assert [s['step'] for s in ss] == list(range(e['length'])) and ss[-1]['done']
    for p in (path, Path(str(stem)+'_episodes.jsonl'), Path(str(stem)+'_steps.jsonl'), Path(str(stem)+'_tokens.jsonl')):
        manifest[str(p.relative_to(RUN))] = {'bytes':p.stat().st_size,'sha256':sha(p)}
    checks.append({'update':number+1,'rows':len(data),'episodes':len(episodes),
                   'padding_to_16':(-len(data))%16,'cache':str(path.relative_to(RUN)),
                   'tokens_match':True,'all_episodes_terminal':True})
assert not list((RUN/'training/checkpoints').glob('global_step_4[12]'))
assert not any(r['update'] == 43 for r in rows(RUN/'advantage_checks.jsonl'))
result = {'latest_complete_checkpoint':40, 'latest_completed_update':42,
          'uncheckpointed_completed_updates':[41,42], 'update43':'sampling complete; failed before response_mask/reward/logprob/advantage/optimizer',
          'four_rank_save_and_prior_restore_fingerprints_equal':True,
          'optimizer_and_scheduler_cpu_hashes_match':True, 'dataloader_loadable':True,
          'cached_samples':checks,'weight_update_chain_intact':True,
          'training_restart':False,'recovery_blocker':'No checkpoint of updates 41/42; exact optimizer replay equivalence has not been established. Do not silently roll back to 40.',
          'failed_phase_gpu_hours':2.4982053213649325,
          'total_recorded_gpu_hours':sum(r.get('gpu_hours',0) for r in rows(RUN/'cost_ledger.jsonl') if r['event']=='end')}
(OUT/'saved_state_checks.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(result,indent=2))
