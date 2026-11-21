"""Four independent native engines with immutable task ownership for serial Stage E."""
import os
import shutil
import time

from .b_collect import native_batches
from .d_data import PARENT
from .d_eval import episodes
from .d_behavior import ALL_REQUIREMENTS
from .e2512_routing import ROUTING, owners
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, read_rows, save


def snapshot_dir(path):
    path.mkdir(parents=True,exist_ok=True)
    link=path/'snapshots'
    if not link.exists():
        link.symlink_to(PARENT/'snapshots',target_is_directory=True)


def checkpoint(l,k,arm):
    if k==0:
        return 'base','B@0'
    cp=l.state(k,arm)/'adapter'
    signature=digest_file(cp/'adapter_model.safetensors')
    assert signature==read(l.folder(k,arm)/'optimizer.json')['safetensors_sha256']
    return cp,signature


def native_rank(rank):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu = visible.split(',')[rank].strip() if visible else str(rank)
    os.environ['SHOP_GPU_IDS'] = gpu
    os.environ['SHOP_GPU_ID'] = gpu


def collect(l,k,arm,rank):
    native_rank(rank)
    source=l.root/'shared_first' if k==1 else l.folder(k,arm)
    p=source/'native'/f'rank{rank}'
    if (p/'complete.json').exists():
        return
    snapshot_dir(p)
    cp,signature=checkpoint(l,k-1,arm)
    tasks=[t for t in read(l.folder(k,arm).parent/'tasks.json') if owners()[t['goal_index']]==rank]
    assert len(tasks)==8
    goals=read(PARENT/'snapshots/goals_frozen.json')
    files={rep:p/f'rollout_{rep}.pending' for rep in range(2)}
    for tmp in files.values():
        tmp.unlink(missing_ok=True)
    start=time.monotonic()
    runner=Runner(p,cp,parallel_start=True)
    try:
        native_batches(runner,tasks,goals,files,stream=f'T_opd_D_{k}',include_legal=True)
    finally:
        runner.close()
    signatures=[]
    for rep,tmp in files.items():
        out=p/f'rollout_{rep}.jsonl'
        tmp.replace(out)
        signatures.append(digest_file(out))
    save(p/'complete.json',dict(rank=rank,goals=[v['goal_index'] for v in tasks],checkpoint=signature,
         routing_sha=digest_file(ROUTING),rollout_sha256=signatures,seconds=time.monotonic()-start,
         startup=runner.startup_timing,finished=time.time()))


def merge_tapes(l,k,arm):
    dst=l.folder(k,arm)
    base=l.root/'shared_first' if k==1 else dst
    _,signature=checkpoint(l,k-1,arm)
    if not (base/'collection.json').exists():
        tasks=read(dst.parent/'tasks.json')
        all_goals={v['goal_index'] for v in tasks}
        legal=total=0;signatures=[]
        for rep in range(2):
            out=base/f'rollout_{rep}.jsonl'
            tmp=out.with_suffix('.pending');done=set()
            with tmp.open('wb') as merged:
                for rank in range(4):
                    p=base/'native'/f'rank{rank}';meta=read(p/'complete.json')
                    assert meta['checkpoint']==signature and meta['routing_sha']==digest_file(ROUTING)
                    assert set(meta['goals'])=={g for g in all_goals if owners()[g]==rank}
                    part=p/f'rollout_{rep}.jsonl'
                    assert digest_file(part)==meta['rollout_sha256'][rep]
                    for x in read_rows(part):
                        assert owners()[x['goal']]==rank and x['rep']==rep and x['goal'] not in done
                        legal+=int(x['legal']);total+=1
                        if x['done']:done.add(x['goal'])
                    with part.open('rb') as f:
                        shutil.copyfileobj(f,merged)
            assert done==all_goals
            tmp.replace(out)
            signatures.append(digest_file(out))
        save(base/'collection.json',dict(checkpoint=signature,rollout_sha256=signatures,
             routing_sha=digest_file(ROUTING),legal=legal,records=total,legal_rate=legal/total))
    meta=read(base/'collection.json')
    assert meta['checkpoint']==signature and meta['routing_sha']==digest_file(ROUTING)
    if k==1:
        for i,h in enumerate(meta['rollout_sha256']):
            first=base/f'rollout_{i}.jsonl';dest=dst/first.name
            assert digest_file(first)==h
            if not dest.exists() and not dest.with_suffix('.jsonl.gz').exists():
                dest.hardlink_to(first)
            if dest.exists():assert digest_file(dest)==h
        save(dst/'collection.json',meta)


def evaluate(l,k,arm,rank):
    native_rank(rank)
    root=l.root/'evaluation'/f'{k:03}'/arm/f'rank{rank}'
    if (root/'complete.json').exists():return
    snapshot_dir(root)
    os.environ['OPD_ENTROPY_TAPE']=str(root/f'eval_entropy_{rank}.bin')
    cp,signature=checkpoint(l,k,arm)
    split=read(PARENT/'asin_split_manifest.json')
    controller=Runner(root,cp,parallel_start=True)
    count=0
    try:
        for pool in ('V','T_opd_eval'):
            tasks=[t['goal_index'] for t in split[pool] if owners()[t['goal_index']]==rank]
            assert len(tasks)==50
            jobs=[(pool,g,rep) for g in tasks for rep in range(5)]
            sampled={}
            for i in range(0,len(jobs),32):
                f=root/f'episodes_{pool}_{i:04}.json';part=jobs[i:i+32]
                if not f.exists():
                    if os.environ.get('OPD_MIGRATION_GROUP'):
                        from .e2514_search import observer
                        logs,record=observer()
                        rows=episodes(controller,part,action_observer=record)
                        save(root/f'search_{pool}_{i:04}.json',logs)
                        save(f,rows)
                    else:save(f,episodes(controller,part))
                rows=read(f)
                assert {(e['split'],e['goal'],e['rep']) for e in rows}==set(part) and len(rows)==len(part)
                count+=len(rows)
                if pool=='T_opd_eval':
                    for e in rows:
                        if e['rep']<4:
                            sampled[e['goal'],e['rep']]=int(e['reason'] in ALL_REQUIREMENTS)
            if pool=='T_opd_eval':
                assert len(sampled)==200
                save(root/'gate_T_opd_eval_sample.json',dict(arm=arm,step=k,rank=rank,split=pool,decode='sample',
                     routing_sha=digest_file(ROUTING),checkpoint=signature,
                     trials=[[g,rep,win] for (g,rep),win in sorted(sampled.items())]))
    finally:
        controller.close()
    assert count==500
    save(root/'complete.json',dict(arm=arm,k=k,rank=rank,count=count,checkpoint=signature,routing_sha=digest_file(ROUTING)))
