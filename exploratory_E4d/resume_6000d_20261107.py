"""Explicit same-model replacement-GPU resume; original experiment is frozen."""
import argparse, fcntl, importlib, os, subprocess, sys, time, traceback
from pathlib import Path
import run_server as S
E, D, H, ROOT = S.E, S.D, S.H, S.ROOT
OLD = H / 'run'
R = H / 'run_resume_6000d_20261107'
S.R = R


def checked_plan():
    p = E.load(R / 'plan.json')
    old = E.load(OLD / 'plan.json')
    assert E.sha_file(OLD / 'plan.json') == p['original_plan_sha256']
    for name, sha in {**old['source'], **p['source']}.items():
        assert E.sha_file(Path(name)) == sha, name
    import importlib.metadata as metadata
    assert all(metadata.version(k) == v for k, v in old['versions'].items())
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    assert inventory == p['gpu_inventory'], 'Device changed again; stop for review'
    assert E.sha_file(R / 'preserved_manifest.json') == p['preserved_manifest_sha256']
    return p


def env_for(i):
    p = checked_plan()
    return dict(os.environ, CUDA_VISIBLE_DEVICES=p['gpus'][i],
                E4D_EXECUTION_PLAN_SHA256=E.sha_file(R / 'plan.json'),
                M0_MODEL=p['model'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                VLLM_WORKER_MULTIPROC_METHOD='spawn')


S.checked_plan = checked_plan
S.env_for = env_for


def preserved():
    p = checked_plan()
    manifest = E.load(R / 'preserved_manifest.json')
    for path, sha in manifest.items():
        assert E.sha_file(ROOT / path) == sha, 'Old data changed: ' + path
    old = E.load(OLD / 'plan.json')
    for path, sha in old['protected_inputs'].items():
        assert E.sha_file(Path(path)) == sha, path
    E.locked(); E.frozen_first_manifest(E.load(E.LOCK))
    return p, manifest


def lora_smoke(i):
    plan = checked_plan()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpus'][i]
    from e4_policy import LoraPolicy
    pts = sorted(E.zpoints(E.locked()), key=lambda x: x['point_id'])
    chosen = [pts[0], pts[-1]]
    prompts = [p['prompt'] for p in chosen]
    seeds = [D.SCREEN_OFFSET + 987654 + k for k in range(2)]
    pol = LoraPolicy(); outputs = {}
    try:
        for arm, seed in [('base', None)] + [(a, s) for a in D.ALL_ARMS for s in D.SEEDS]:
            if arm == 'base': pol.set_adapter(None, 0)
            else: pol.set_adapter(str(E.ADAPTER_DIR / f'{arm}_s{seed}'), 1 + D.ALL_ARMS.index(arm) * 3 + seed)
            assert (pol.lora is None) == (arm == 'base')
            outputs[f'{arm}_{seed}'] = pol.sample(prompts, seeds, logprobs=False)
    finally:
        pol.close()
    E.save_json(R / f'lora_smoke_{i}.json', dict(gpu=plan['gpus'][i],
        point_ids=[p['point_id'] for p in chosen], prompts_sha256=D.digest(prompts),
        seeds=seeds, outputs=outputs, execution_plan_sha256=E.sha_file(R / 'plan.json')))


def dual(phase):
    children = []
    for i in range(2):
        with (R / f'{phase}_{i}_driver.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--worker', phase, '--gpu', str(i)], cwd=ROOT, env=env_for(i),
                stdout=log, stderr=subprocess.STDOUT))
    E.save_json(R / f'{phase}_launch.json', dict(pids=[p.pid for p in children], time=time.time()))
    S.state(phase, children=[p.pid for p in children])
    while any(p.poll() is None for p in children):
        assert all(p.poll() in (None, 0) for p in children), 'One worker failed; other worker preserved'
        time.sleep(10)
    assert all(p.returncode == 0 for p in children)


def check_smokes():
    old = [E.load(OLD / f'smoke_{i}.json') for i in range(2)]
    new = [E.load(R / f'smoke_{i}.json') for i in range(2)]
    assert all(x['passed'] for x in old + new)
    assert all(x['prompts_sha256'] == old[0]['prompts_sha256'] and x['outputs'] == old[0]['outputs'] for x in old + new)
    a, b = [E.load(R / f'lora_smoke_{i}.json') for i in range(2)]
    assert a['prompts_sha256'] == b['prompts_sha256'] and a['seeds'] == b['seeds']
    assert len(a['outputs']) == 13 and a['outputs'] == b['outputs'], 'Base/LoRA cross-card outputs differ'
    E.save_json(R / 'smoke_verified.json', dict(passed=True, old_base_equal=True,
        current_base_and_12_adapters_equal=True, prompts=2,
        files={str(p.relative_to(ROOT)):E.sha_file(p) for p in
               [OLD/f'smoke_{i}.json' for i in range(2)] +
               [R/f'{kind}_{i}.json' for kind in ('smoke','lora_smoke') for i in range(2)]},
        limitation='Finite exact generation checks do not guarantee equality on all possible inputs.'))


def execution_valid(x, path, point, plan, manifest):
    rel = str(path.relative_to(ROOT))
    old = E.load(OLD / 'plan.json')
    shard = int(D.digest('E4d-gpu-shard:' + point['point_id']), 16) % 2
    ex = x['provenance']['execution']
    if rel in manifest:
        assert E.sha_file(path) == manifest[rel]
        expected_uuid, expected_sha = old['gpus'][shard], E.sha_file(OLD / 'plan.json')
    else:
        expected_uuid, expected_sha = plan['gpus'][shard], E.sha_file(R / 'plan.json')
    assert ex['gpu_uuid'] == expected_uuid and ex['execution_plan_sha256'] == expected_sha
    assert ex['source_sha256'] == E.sha_file(H / 'e4d.py')


def verify():
    plan, manifest = preserved(); check_smokes()
    lock = E.locked(); pts = E.zpoints(lock)
    replay = importlib.import_module('11_replay'); raw = {}
    def check(path, p, u, offset):
        x = E.load(path); E.verify_target_log(x)
        assert len(x['continuations']) == 1
        c = x['continuations'][0]
        assert c['rep'] == u and not c['refeed_div'] and len(c['actions']) == len(c['valids'])
        assert c['len'] == p['turn'] + len(c['actions']) - 1 and c['len'] <= 50
        assert len(c['seeds']) == len(c['actions']) - 1
        assert c['seeds'] == [replay.seed_of(p['seed'], p['turn'], 1, u*100+p['turn']+k)+offset for k in range(len(c['seeds']))]
        assert x['factual_outcomes'] == [int(c['won'])]
        execution_valid(x, path, p, plan, manifest)
        raw[str(path.relative_to(ROOT))] = E.sha_file(path)
        return x
    for p in pts:
        for arm, cond in [('base','00')] + [(a,c) for a in D.ALL_ARMS for c in ('01','10','11')]:
            E.result_prob(arm, cond, p, lock)
            for u in range(24):
                x = check(D.run_path(E.DATA, arm, cond, p['point_id'], u), p, u, D.FORMAL_OFFSET)
                expected = lock['adapters'][f'{arm}_s{u//8}'] if cond in ('01','11') else None
                assert x['provenance']['adapter_sha256'] == expected
    by = {p['point_id']: p for p in pts}
    for arm in ('base',) + D.PRIMARY_ARMS:
        E.read_probe_policy(arm, lock)
        for pid in lock['probe_ids']:
            for j in range(5):
                for u in range(8): check(E.probe_path(arm,pid,j,u), by[pid], u, D.PROBE_OFFSET)
    assert len(raw) == 312*len(pts) + 3*5*8*len(lock['probe_ids'])
    E.save_json(R / 'raw_manifest.json', raw)
    E.save_json(R / 'verification.json', dict(passed=True, states=len(pts), games=lock['n_games_Z3'],
        records=len(raw), probe_points=len(lock['probe_ids']), preserved_files=len(manifest),
        raw_manifest_sha256=E.sha_file(R/'raw_manifest.json'),
        primary_sha256=E.sha_file(H/'e4d.json'), secondary_sha256=E.sha_file(H/'e4d_armprobe.json'),
        primary_reading=E.load(H/'e4d.json').get('reading'),
        secondary_reading=E.load(H/'e4d_armprobe.json')['reading_secondary'],
        hardware_replacement_disclosed=True, paired_seeds_and_shards_verified=True,
        original_plan_sha256=plan['original_plan_sha256'], execution_plan_sha256=E.sha_file(R/'plan.json')))


def main():
    with (R/'driver.lock').open('a') as lock, (OLD/'driver.lock').open('a') as oldlock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(oldlock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            S.state('checking_preserved_inputs'); preserved()
            if not (R/'smoke_verified.json').exists():
                dual('smoke'); dual('lora_smoke'); check_smokes()
            else: check_smokes()
            dual('formal')
            S.state('analyzing')
            for cmd in ('analyze', 'probe-analyze'): S.execute(cmd, [H/'e4d.py', cmd])
            verify(); S.state('completed_verified')
            S.wake('【E4d换卡续跑完成】请检查 '+str(R/'verification.json')+' 与原主/次报告，独立核验并仅发布credit-grok。当前双6000D的设备更换已登记，旧5622条保留。hindsight H80032B仍在49点断点且未达204点检查，不得因E4d完成提前关机；请处理剩余队列。')
        except Exception:
            error = traceback.format_exc(); (R/'failure.log').write_text(error); S.state('failed', error=error)
            S.wake('【E4d换卡续跑异常】请读 '+str(R/'failure.log')+' 并尽力修复。保留另一正常工作进程、全部旧记录、冻结协议及门槛；不得改旧plan或静默混设备。H80032B尚未完成，不得关机。')
            raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('smoke','lora_smoke','formal'));p.add_argument('--gpu',type=int,choices=(0,1));p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.worker == 'smoke': S.smoke(a.gpu)
    elif a.worker == 'lora_smoke': lora_smoke(a.gpu)
    elif a.worker == 'formal':
        check_smokes();S.worker('formal',a.gpu)
    elif a.verify: verify()
    else: main()
