"""Register task placement using counts/timing only; never read success outcomes."""
import time
from pathlib import Path
import accelerate_20261108 as A
E,D,H,ROOT=A.E,A.D,A.H,A.ROOT
R=H/'run_balanced_20261109'

def assign(jobs,initial):
    loads=list(initial);out={}
    for job in sorted(jobs,key=lambda j:(-j['estimated_seconds'],j['key'])):
        gpu=min(range(2),key=lambda i:(loads[i],i));out[job['key']]=gpu;loads[gpu]+=job['estimated_seconds']
    return out,loads

def main():
    assert not (R/'plan.json').exists(),'Immutable registration already exists'
    old=A.checked_plan();A.check_smokes();R.mkdir(exist_ok=True)
    pts=E.zpoints(E.load(E.LOCK));probe_ids=E.load(E.LOCK)['probe_ids']
    shard=lambda p:int(D.digest('E4d-gpu-shard:'+p['point_id']),16)%2
    rates={0:dict(base=2.1663768988554226,lora=2.5916902304373),1:dict(base=2.030922129543291,lora=2.4690387793299227)}
    tasks={}
    for g in [0,1]:
        for arm,c in [('base','00')]+[(a,c) for a in D.ALL_ARMS for c in ('01','10','11')]:
            key=f'formal:{arm}:{c}:{g}'
            remaining=sum(not D.run_path(E.DATA,arm,c,p['point_id'],u).exists() for p in pts if shard(p)==g for u in range(24))
            tasks[key]=dict(key=key,phase='formal',arm=arm,cond=c,shard=g,remaining=remaining,
                estimated_seconds=remaining*rates[g]['lora' if c in ('01','11') else 'base'],
                args=['run','--arm',arm,'--cond',c,'--workers','24','--max_concurrent','400','--shard-count','2','--shard-index',str(g)])
        for arm in ('base',)+D.PRIMARY_ARMS:
            key=f'probe:{arm}:{g}'
            remaining=sum(not E.probe_path(arm,p['point_id'],j,u).exists() for p in pts if shard(p)==g and p['point_id'] in probe_ids for j in range(5) for u in range(8))
            tasks[key]=dict(key=key,phase='probe',arm=arm,shard=g,probe_ids=probe_ids,remaining=remaining,
                estimated_seconds=remaining*rates[g]['base' if arm=='base' else 'lora'],
                args=['armprobe','--policy-id',arm,'--workers','24','--max_concurrent','400','--shard-count','2','--shard-index',str(g)])
    status=E.load(A.R/'status.json');workers=status['children'];active={};science={};processes={}
    for g in [0,1]:
        s=E.load(A.R/f'formal_{g}_status.json');assert s['pid']==workers[g] and s['stage']=='running'
        q=E.load(A.R/(s['task']+f'_gpu{g}_process.json'));args=q['args'];assert '--science' in args and 'run' in args
        arm=args[args.index('--arm')+1];cond=args[args.index('--cond')+1];key=f'formal:{arm}:{cond}:{g}'
        active[str(g)]=tasks[key];p=A.proc(q['pid']);assert p and p['state']!='Z'
        science[str(g)]=dict(pid=q['pid'],start=p['start'],args=args,log=str(A.R/(s['task']+f'_gpu{g}.log')))
    for pid in [status['pid'],*workers]:
        p=A.proc(pid);assert p and p['state']!='Z';processes[str(pid)]={**p,'args':Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()}
    receipt=dict(controller=status['pid'],workers=workers,science=science,processes=processes,time=time.time())
    E.save_json(R/'handoff.json',receipt)
    pending=[j for key,j in tasks.items() if j['remaining'] and key not in {j['key'] for j in active.values()}]
    initial=[active[str(g)]['estimated_seconds'] for g in [0,1]]
    assigned,loads=assign(pending,initial)
    for job in pending:job['gpu']=assigned[job['key']]
    queues={str(g):[j['key'] for j in sorted(pending,key=lambda j:(j['phase']!='formal',-j['estimated_seconds'],j['key'])) if j['gpu']==g] for g in [0,1]}
    before=[initial[g]+sum(j['estimated_seconds'] for j in pending if j['shard']==g) for g in [0,1]]
    manifest=E.load(A.R/'preserved_manifest.json')
    for p in (E.DATA/'results').rglob('*.json'):manifest[str(p.relative_to(ROOT))]=E.sha_file(p)
    E.save_json(R/'preserved_manifest.json',manifest)
    p=dict(old);p['source']=dict(old['source']);p['parent_plans']=dict(old['parent_plans'])
    p['parent_plans'][str(A.R/'plan.json')]=E.sha_file(A.R/'plan.json')
    p.update(time=time.time(),jobs={j['key']:j for j in pending},active_jobs=active,queues=queues,
        handoff_sha256=E.sha_file(R/'handoff.json'),preserved_manifest_sha256=E.sha_file(R/'preserved_manifest.json'),
        estimated_before_hours=[x/3600 for x in before],estimated_balanced_hours=[x/3600 for x in loads],
        no_shutdown=False,scheduling_only=True,assignment_uses_outcomes=False,
        registration_caveat='Work estimates use existing throughput and completed-file counts only; physical GPU allocation changes, logical shards and all scientific parameters do not.')
    for n in ['balance_20261109.py','register_balance_20261109.py','test_balance_20261109.py','BALANCE_20261109.txt']:
        p['source'][str(H/n)]=E.sha_file(H/n)
    E.save_json(R/'plan.json',p)
    print(dict(before_hours=p['estimated_before_hours'],balanced_hours=p['estimated_balanced_hours'],jobs=len(pending),moved=[k for k,j in p['jobs'].items() if j['gpu']!=j['shard']]),flush=True)

if __name__=='__main__':main()
