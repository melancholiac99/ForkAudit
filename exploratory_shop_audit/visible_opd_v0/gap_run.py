"""Run R-gap then P-gap; isolate all outputs and defer V until both arm endpoints are frozen."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .e25_layout import Layout, RoutedBatch
from .e2512_native import checkpoint, collect, merge_tapes, native_rank, snapshot_dir
from .e2512_routing import owners
from .gap_eval import run_batch
from .gap_preflight import multi
from .gap_score import PUBLIC, RUN, input_tapes, split_tasks
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


class GapLayout(Layout):
    def __init__(self):
        self.name='gap26'
        self.root=RUN/'formal'

    def batch(self,k,arm):
        b=RoutedBatch(k,arm,self.folder(k,arm))
        return b,b.q

    def projector(self,version):
        return self.root/'refits'/f'{version:02}' if version else RUN/'initial'


def training_stats(l,k,arm):
    from .e25_refresh import tape
    from exploratory_shop_audit.env_worker import parse_action_fixed
    import torch
    p=l.folder(k,arm)
    batch,_=l.batch(k,arm)
    assert digest_file(p/'strength.pt')==read(p/'target_manifest.json')['strength_sha256']
    strength=sum(w*float(v.double().sum()) for w,v in zip(batch.weights,torch.load(p/'strength.pt',weights_only=True),strict=True))
    queries={};episodes=set()
    for src in input_tapes(p):
        for r in tape(src):
            key=r['goal'],r['rep']
            episodes.add(key)
            if key not in queries and r['legal'] and parse_action_fixed(r['action'])[0]=='search':
                queries[key]=r['classes'].count('search')
    assert len(episodes)==64
    col=read(p/'collection.json')
    result=dict(update=k,arm=arm,legal=col['legal'],actions=col['records'],legal_rate=col['legal_rate'],
                strength_kl=strength,searches=len(queries),query_tokens=sum(queries.values()),
                missed_searches=len(episodes)-len(queries))
    save(p/'monitor.json',result)
    return result


def window_decision(rows,baseline,prior):
    legal=sum(a['legal'] for a in rows)/sum(a['actions'] for a in rows)
    n=sum(a['searches'] for a in rows)
    length=sum(a['query_tokens'] for a in rows)/n if n else None
    flags=dict(legal=baseline-legal>=.1-1e-12,short_search=length is not None and length<15)
    persist=[name for name in flags if flags[name] and prior.get(name,False)]
    return dict(legal_rate=legal,baseline=baseline,mean_first_query_tokens=length,with_query=n,
                missing_queries=sum(a['missed_searches'] for a in rows),flags=flags,stop_reasons=persist)


def native_eval(l,k,arm,rank,pool,mode):
    from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
    if pool=='V':
        assert all((l.root/'endpoints'/f'{a}.json').exists() for a in ('R-gap','P-gap'))
    native_rank(rank)
    folder=l.root/'evaluation'/pool/mode/f'{arm}_{k:02}'/f'rank{rank}'
    snapshot_dir(folder)
    anchor_dir=l.root/'pages'/pool/f'rank{rank}'
    cp,sha=checkpoint(l,k,arm)
    complete=folder/'complete.json'
    if complete.exists():
        assert read(complete)['checkpoint']==sha
        return
    goals=[t['goal_index'] for t in split_tasks()[pool] if owners()[t['goal_index']]==rank]
    assert len(goals)==50
    engine=Runner(folder,cp,parallel_start=True)
    try:
        for rep in range(5):
            for start in range(0,50,16):
                out=folder/f'part_{rep}_{start:02}.json'
                if not out.exists():
                    save(out,run_batch(engine,pool,goals[start:start+16],rep,anchor_dir,mode,arm=='B' and rep==4 and mode=='free'))
                rows=read(out)
                assert sorted(r['goal'] for r in rows)==sorted(goals[start:start+16]) and all(r['rep']==rep for r in rows)
    finally:
        engine.close()
    save(complete,dict(checkpoint=sha,goals=goals,pool=pool,mode=mode,replicates=5))


def common_probe(l,rank,k,arm):
    import torch
    from .c_targets import actor,response
    from .e25_probe import pooled,full_metrics
    from .e25_paths import RUN as OLD
    from .s21_data import SmokeBatch
    b=SmokeBatch();model=actor(load_on_rank=True).eval()
    l.snapshot(model,k,arm)
    chunks=pooled(b,rank,owners())
    src=OLD/'formal_2512'/'base_logits'/str(torch.bfloat16)/f'rank_{rank}.pt'
    base=torch.load(src,weights_only=True,mmap=True)
    assert set(base)=={ix[0] for ix in chunks}
    w=kl=0.
    with torch.no_grad():
        for ix in chunks:
            i=ix[0];r=b.rows[i];_,logits=response(model,r['prompt'],r['answer'])
            v=full_metrics(base[i].cuda(),logits)[:,0]
            for j in ix:
                w+=b.weights[j]*len(v);kl+=b.weights[j]*float(v.sum())
    save(l.folder(k,arm)/f'probe_{rank}.json',dict(mass=w,kl_sum=kl,reference_sha=digest_file(src)))


def worker(phase,k,arm,extra):
    from .e25_probe import configure
    from .gap_targets import score_rank
    rank=configure();l=GapLayout()
    if phase=='collect':
        collect(l,k,arm,rank,view_modes=('none','full'))
    elif phase=='targets':
        score_rank(l,k,arm,rank)
    elif phase=='train':
        from .d_train import train
        train(k,arms=[arm],folder=l.folder,state_dir=l.state,snapshot=l.snapshot,load_batch=l.batch,
              rates={arm:1e-6},compare_original=False,rank_local_activations=True,
              checkpoint_only=(l.folder(k,arm)/'checkpoint_only_retry').exists(),audit_nominations=(k==2))
    elif phase=='probe':
        common_probe(l,rank,k,arm)
    elif phase=='eval':
        native_eval(l,k,arm,rank,*extra)
    elif phase=='mass':
        from .gap_clicks import masses
        from .c_targets import actor
        from exploratory_shop_audit.grpo_forced_audit.common import MODEL
        from transformers import AutoTokenizer
        pool,=extra
        if pool=='V':assert all((l.root/'endpoints'/f'{a}.json').exists() for a in ('R-gap','P-gap'))
        goals=[t['goal_index'] for t in split_tasks()[pool] if owners()[t['goal_index']]==rank]
        src=l.root/'pages'/pool/f'rank{rank}'
        dest=l.root/'mass'/pool/f'{arm}_{k:02}'/f'rank{rank}.json'
        model=actor(load_on_rank=True).eval()
        if k:l.snapshot(model,k,arm)
        save(dest,masses(model,AutoTokenizer.from_pretrained(MODEL),src,pool,goals,read(src/'qualify.json')))
    elif phase=='refresh_score':
        from .gap_score import score_inputs
        from .c_targets import actor
        model=actor(load_on_rank=True).eval();l.snapshot(model,k,arm)
        future={t['asin'] for step in range(k+1,k+6) for t in read(l.folder(step,arm).parent/'tasks.json')}
        paths=[p for s in range(max(1,k-19),k+1) for p in input_tapes(l.folder(s,arm))]
        score_inputs(rank,k,paths,future,model,l.projector(k))
    elif phase=='refresh_fit':
        from .gap_fit import fit
        fit(l.projector(k),rank,l.projector(k-5))
    else:
        raise ValueError(phase)


def launch(l,phase,k,arm,*extra):
    tag='_'.join([arm,f'{k:02}',phase,*extra])
    out=l.root/'stages'/f'{tag}.json'
    if out.exists():return
    log=l.root/'logs'/f'{tag}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=multi('gap_run','worker',phase,str(k),arm,*extra)
    print(tag,flush=True)
    start=time.monotonic()
    with log.open('a') as f:
        proc=subprocess.run(cmd,stdout=f,stderr=f)
        if proc.returncode and phase=='train' and 'OutOfMemoryError' in log.read_text():
            (l.folder(k,arm)/'checkpoint_only_retry').touch()
            proc=subprocess.run(cmd,stdout=f,stderr=f)
        proc.check_returncode()
    save(out,dict(phase=phase,arm=arm,k=k,seconds=time.monotonic()-start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def retain(l,k,arm):
    # Remove only newly generated, replaceable intermediates inside the section 26 run.
    if k<2:return
    from shutil import rmtree
    assert (l.state(k,arm)/'adam.pt').is_file()
    current=read(l.folder(k,arm)/'optimizer.json')
    assert digest_file(l.state(k,arm)/'adam.pt')==current['adam_sha256']
    prev=l.state(k-1,arm)
    (prev/'adam.pt').unlink(missing_ok=True)
    if (k-1)%5 and (prev/'adapter').exists():
        rmtree(prev/'adapter')
    if k>5 and (k-1)%5==0:
        old=k-6
        if old>0 and old%10 and (l.state(old,arm)/'adapter').exists():
            rmtree(l.state(old,arm)/'adapter')



def archive_rollout(l,k,arm):
    import gzip
    import hashlib
    import shutil
    path=l.folder(k,arm)
    # Step 1 remains the shared signed B source for both arms.
    if k==1:return
    signature=read(path/'collection.json')['rollout_sha256']
    for rep,sha in enumerate(signature):
        src=path/f'rollout_{rep}.jsonl'
        gz=src.with_suffix('.jsonl.gz')
        if not gz.exists():
            temp=gz.with_suffix('.tmp')
            with src.open('rb') as stream,gzip.open(temp,'wb',compresslevel=2) as out:
                shutil.copyfileobj(stream,out)
            with gzip.open(temp,'rb') as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
            temp.replace(gz)
        src.unlink(missing_ok=True)
        for r in range(4):
            base=path/'native'/f'rank{r}'
            part=base/src.name
            if part.exists():
                assert digest_file(part)==read(base/'complete.json')['rollout_sha256'][rep]
                part.unlink()
    for r in range(4):
        (path/'scored'/str(r)/'rows.pt').unlink(missing_ok=True)


def train_arm(l,arm):
    from .gap_targets import merge
    from .gap_fit import calibrate
    endpoint=l.root/'endpoints'/f'{arm}.json'
    if endpoint.exists():return read(endpoint)['update']
    for k in range(1,51):
        p=l.folder(k,arm)
        if not (p/'monitor.json').exists():
            if arm=='P-gap' and k>1 and (k-1)%5==0:
                v=k-1;proj=l.projector(v)
                launch(l,'refresh_score',v,arm)
                launch(l,'refresh_fit',v,arm)
                if not (proj/'calibration.json').exists():calibrate(proj)
                if (proj/'calibration.json').exists():
                    for f in proj.glob('features_*.pt'):f.unlink()
            launch(l,'collect',k,arm);merge_tapes(l,k,arm)
            launch(l,'targets',k,arm)
            if not (p/'target_manifest.json').exists():merge(l,k,arm)
            launch(l,'train',k,arm)
            training_stats(l,k,arm)
            retain(l,k,arm)
            archive_rollout(l,k,arm)
        stopped=False
        if k%5==0:
            prev=l.folder(k-5,arm)/'window.json'
            decision=window_decision([read(l.folder(j,arm)/'monitor.json') for j in range(k-4,k+1)],
                                    read(l.root/'shared_first/collection.json')['legal_rate'],
                                    read(prev)['flags'] if k>5 else {})
            save(p/'window.json',decision)
            stopped=bool(decision['stop_reasons'])
        if k%10==0:
            launch(l,'probe',k,arm)
        terminal=stopped or k==50
        if k in (10,20,30,50) or terminal:
            launch(l,'eval',k,arm,'T_opd_eval','free')
        if terminal:
            save(endpoint,dict(arm=arm,update=k,early_stop=stopped,reason=decision['stop_reasons'],
                               adapter_sha=read(p/'optimizer.json')['safetensors_sha256']))
            return k


def main():
    l=GapLayout();l.root.mkdir(parents=True,exist_ok=True)
    with (l.root/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(PUBLIC/'initialization_check.json')['passed'] and read(PUBLIC/'preflight_budget.json')['budget_gpu_hours_25pct']>0
        assert (l.root/'launch_authorization.json').is_file()
        arms=('R-gap','P-gap')
        l.initialize(50,arms=arms)
        endpoints={a:train_arm(l,a) for a in arms}
        for pool in ('T_opd_eval','V'):
            launch(l,'eval',0,'B',pool,'free')
            for a,k in endpoints.items():
                if pool=='V':
                    for j in sorted({v for v in (20,30,50) if v<=k}|{k}):
                        launch(l,'eval',j,a,pool,'free')
            for rank in range(4):
                src=l.root/'pages'/pool/f'rank{rank}'
                goals=[t['goal_index'] for t in split_tasks()[pool] if owners()[t['goal_index']]==rank]
                if not (src/'qualify.json').exists():
                    # Native scoring dependency lives in the CPU environment.
                    import subprocess
                    from exploratory_shop_audit.grpo_forced_audit.common import CPU_PYTHON
                    cmd=[str(CPU_PYTHON),'-m','exploratory_shop_audit.visible_opd_v0.gap_qualify',str(src),pool,*map(str,goals)]
                    subprocess.run(cmd,check=True)
            for a,k in {'B':0,**endpoints}.items():
                launch(l,'eval',k,a,pool,'page')
                launch(l,'mass',k,a,pool)
        from .gap_report import report
        report(l,endpoints)


if __name__=='__main__':
    if sys.argv[1:2]==['worker']:
        worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5:])
    else:
        main()
