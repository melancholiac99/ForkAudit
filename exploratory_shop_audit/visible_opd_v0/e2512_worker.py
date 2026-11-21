"""Use the full machine for the current serial arm with the signed task placement."""
import os
import sys
import time
from .e25_paths import RUN, SERIAL
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def target_shards(l,k,a):
    from .e2512_routing import owners
    tasks=read(l.folder(k,a).parent/'tasks.json')
    return [[((SERIAL,k,a),t['goal_index']) for t in tasks if owners()[t['goal_index']]==rank] for rank in range(4)]


def native_targets(l, k, arm, rank):
    from . import d_targets
    from .e2512_routing import ROUTING
    path = l.folder(k, arm)
    native = path / 'native' / f'rank{rank}'
    receipt = read(native / 'complete.json')
    goals = {g for _, g in target_shards(l, k, arm)[rank]}
    assert set(receipt['goals']) == goals and receipt['rank'] == rank
    assert receipt['routing_sha'] == digest_file(ROUTING)
    assert receipt['checkpoint'] == read(l.folder(k-1, arm) / 'optimizer.json')['safetensors_sha256']
    part = path / 'target_parts' / str(rank)
    start = time.time()
    d_targets.build(k, arm, layout=l, goals=goals, output=part,
                    rollout_root=native, rollout_hashes=receipt['rollout_sha256'])
    meta = read(part / 'target_manifest.json')
    assert set(meta['goals']) == goals and meta['native_rollout_sha256'] == receipt['rollout_sha256']
    assert meta['scorer_sha'] == read(l.folder(k-1, arm) / 'optimizer.json')['after_sha']
    if not (part / 'pipeline_timing.json').exists():
        save(part / 'pipeline_timing.json', dict(start=start, end=time.time(), rank=rank))


def collected_and_scored(k, arm, rank):
    # The parent verifies each source tape and the four target files during the merges.
    path = RUN/SERIAL/'pilot'/f'{k:02}'/arm
    part = path/'target_parts'/str(rank)
    if not (part/'pipeline_timing.json').is_file():
        return False
    native = read(path/'native'/f'rank{rank}'/'complete.json')
    target = read(part/'target_manifest.json')
    previous = read(path.parents[1]/f'{k-1:02}'/arm/'optimizer.json')
    assert native['rank']==rank and native['checkpoint']==previous['safetensors_sha256']
    assert target['scorer_sha']==previous['after_sha']
    assert set(native['goals'])==set(target['goals']) and len(target['goals'])==8
    assert native['rollout_sha256']==target['native_rollout_sha256']
    assert native['routing_sha']==read(RUN/SERIAL/'settings.json')['routing_sha']
    return True


def evaluation_complete(k, arm, rank):
    root=RUN/SERIAL
    file=root/'evaluation'/f'{k:03}'/arm/f'rank{rank}'/'complete.json'
    if not file.exists():
        return False
    meta=read(file)
    assert (meta['k'],meta['arm'],meta['rank'],meta['count'])==(k,arm,rank,500)
    assert meta['routing_sha']==read(root/'settings.json')['routing_sha']
    assert meta['checkpoint']==read(root/'pilot'/f'{k:02}'/arm/'optimizer.json')['safetensors_sha256']
    return True


def can_prepare_next(k):
    # The next actor and projector are known here only between monitoring points.
    return 7 <= k < 100 and k % 10 != 0


def main():
    phase,step,arm=sys.argv[1:]
    k=int(step);rank=int(os.environ['LOCAL_RANK'])
    if phase in ('collect','targets') and k>=7 and collected_and_scored(k,arm,rank):
        return
    if phase=='eval' and k>=10 and evaluation_complete(k,arm,rank):
        return
    from .e25_layout import Layout
    from .e25_probe import configure
    from .e2512_native import collect, evaluate
    rank=configure();l=Layout(SERIAL)
    if phase=='collect':
        collect(l,k,arm,rank)
        if k>=7:
            native_targets(l,k,arm,rank)
    elif phase=='eval':evaluate(l,k,arm,rank)
    elif phase=='targets':
        if k>=7:
            native_targets(l,k,arm,rank)
        else:
            from .e2511_targets import score_shard
            score_shard(rank,target_shards(l,k,arm))
    elif phase=='train':
        import torch
        from . import d_train
        path=l.folder(k,arm)
        retry=any(path.glob('retained_oom_rank*.json'))
        try:
            d_train.train(k,arms=[arm],folder=l.folder,state_dir=l.state,snapshot=l.snapshot,load_batch=l.batch,
                          rates={**l.config['rates'],'P-frozen':l.config['rates']['P']},compare_original=False,rank_local_activations=(k>=7 and not retry),
                          checkpoint_only=retry)
        except torch.cuda.OutOfMemoryError:
            if k>=7 and not retry:
                save(path/f'retained_oom_rank{rank}.json',dict(rank=rank,update=k,arm=arm))
            raise
    elif phase=='per_update':
        from .e25_measure import per_update
        started=time.time()
        per_update(rank,l,k,[arm])
        timing=l.folder(k,arm)/f'fp32_timing_{rank}.json'
        if not timing.exists():
            save(timing,dict(rank=rank,update=k,start=started,end=time.time()))
        if can_prepare_next(k):
            # A fresh process releases the FP32 actor and its CUDA context before vLLM starts.
            print(f'Rank {rank} finished FP32 update {k}; preparing {arm} update {k+1}',flush=True)
            os.execv(sys.executable, [sys.executable,'-m',__package__+'.e2512_worker','collect',str(k+1),arm])
    elif phase=='common':
        from .e25_measure import common
        common(rank,l,k,[arm])
        if k>=10:
            # Both a continuing arm and a stopping arm require this same endpoint evaluation.
            print(f'Rank {rank} finished the common probe; evaluating {arm}@{k}',flush=True)
            os.execv(sys.executable,[sys.executable,'-m',__package__+'.e2512_worker','eval',str(k),arm])
    elif phase.startswith('refresh_'):
        from . import e25_refresh
        assert arm=='P' and k%10==0
        dst=l.root/'refresh'/f'{k:03}';dst.mkdir(parents=True,exist_ok=True)
        if phase=='refresh_score':e25_refresh.score(rank,SERIAL,k,l.config['window_updates'],dst)
        elif phase=='refresh_fit':e25_refresh.fit(rank,dst)
        else:raise ValueError(phase)
    else:raise ValueError(phase)


if __name__=='__main__':
    main()
