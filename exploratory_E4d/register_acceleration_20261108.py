"""One-time plan registration, exact diagnostic gate, immutable checkpoint."""
from pathlib import Path
import time
import resume_devices_20261108 as M
E,H,ROOT=M.E,M.H,M.ROOT
R=H/'run_accelerated_20261108';F=H/'speedcheck_20261108'

def main():
    assert not (R/'plan.json').exists(),'Plan is immutable'
    M.check_smokes();previous,manifest=M.B.preserved()
    baseline=E.load(F/'baseline200.json');fast=E.load(F/'fast400.json')
    small=E.load(F/'async_graph.json');oldsmall=E.load(F/'baseline.json')
    ref=E.load(Path(previous['reference']))
    assert baseline['results']==fast['results'] and len(fast['results'])==800
    assert baseline['workload_sha256']==fast['workload_sha256']
    assert baseline['engine_changes']==M.N.ENGINE_CHANGES
    assert fast['environment']==M.N.ENV_CHANGES
    assert small['results']==oldsmall['results'] and small['smoke']==ref['outputs']
    assert all(small['checks'].values()) and small['passed']
    for key in ('enforce_eager','enable_prefix_caching','async_scheduling'):
        assert fast['engine_changes'][key]==small['engine_changes'][key]
    assert fast['source_sha256']==baseline['source_sha256']==E.sha_file(F/'benchmark_large.py')
    R.mkdir(exist_ok=True)
    for path in sorted((E.DATA/'results').rglob('*.json')):
        manifest[str(path.relative_to(ROOT))]=E.sha_file(path)
    E.save_json(R/'preserved_manifest.json',manifest)
    gate=dict(passed=True,large_exact_responses=800,small_exact_responses=128,old_exact_anchors=26,
        baseline_seconds=baseline['total_seconds'],fast_seconds=fast['total_seconds'],
        offline_speedup=baseline['total_seconds']/fast['total_seconds'],
        limitation='Finite exact checks; static batches are not end-to-end replay ETA.')
    E.save_json(R/'smoke_verified.json',gate)
    E.save_json(R/'handoff.json',E.load(F/'handoff.json'))
    p=dict(previous)
    p.update(time=time.time(),engine_changes=fast['engine_changes'],workers_per_gpu=24,max_concurrent_per_gpu=400,
        parent_plans={str(d/'plan.json'):E.sha_file(d/'plan.json') for d in [M.PREV,M.R]},
        preserved_manifest_sha256=E.sha_file(R/'preserved_manifest.json'),
        smoke_verification_sha256=E.sha_file(R/'smoke_verified.json'),
        benchmark_files={str(F/n):E.sha_file(F/n) for n in ['baseline.json','async_graph.json','baseline200.json','fast400.json','workload.json']},
        reused_records=len(list((E.DATA/'results').rglob('*.json'))),no_shutdown=True)
    for n in ['accelerate_20261108.py','register_acceleration_20261108.py','test_accelerate_20261108.py','ACCELERATION_20261108.txt']:
        p['source'][str(H/n)]=E.sha_file(H/n)
    for n in ['benchmark.py','benchmark_async.py','benchmark_large.py']:
        p['source'][str(F/n)]=E.sha_file(F/n)
    p['source'][str(R/'handoff.json')]=E.sha_file(R/'handoff.json')
    E.save_json(R/'plan.json',p)
    print(gate,'snapshot',p['reused_records'],flush=True)

if __name__=='__main__':main()
