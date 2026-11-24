"""Replay §26 with one execution-only change, the first of successive commands."""
import argparse
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import re

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import time
from .e2512_native import checkpoint, snapshot_dir
from .e2512_routing import owners
from .gap_cpu_actions import classify
from .gap_eval import run_batch, SEEDS
from .gap_run import GapLayout
from .gap_score import RUN, PUBLIC, split_tasks
from exploratory_shop_audit.grpo_forced_audit.common import SOURCE, digest_file, read, save
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.grpo_forced_audit.checks import environment

ROOT = RUN/'first_command_replay'
DEST = PUBLIC/'first_command_replay'
MODELS = [('B',0),('R-gap',30),('R-gap',45),('P-gap',30),('P-gap',50)]
POOLS = ('T_opd_eval','V')
VERB = r'[A-Za-z_]\w*\['
FIRST = re.compile(r'(?m)^[ \t]*(?:Action:[ \t]*)?('+VERB+')',re.ASCII)
NEXT = re.compile(r'\s*(?:Action:[ \t]*)?('+VERB+')',re.ASCII)
LATER = re.compile(r'(?m)^[ \t]*(?:Action:[ \t]*)?('+VERB+')',re.ASCII)


def end_command(text,opening):
    depth=1
    for j in range(opening,len(text)):
        depth+=(text[j]=='[')-(text[j]==']')
        if not depth:
            return j+1


def first_command(text,native):
    message=text.replace('\\n','\n')
    first=FIRST.search(message)
    if first:
        end=end_command(message,first.end(1))
        if end is not None:
            later=NEXT.match(message,end) or LATER.search(message,end)
            if later and end_command(message,later.end(1)) is not None:
                return message[first.start(1):end],True
    return native,False


@lru_cache(maxsize=1)
def catalog():
    src=SOURCE/'community_products.json'
    return {str(v['value']).lower() for p in read(src) for axis in (p.get('customization_options') or {}).values()
            for v in axis if isinstance(v,dict) and v.get('value') is not None}


class FirstCommand:
    select=staticmethod(first_command)

    def __init__(self,goals):
        self.gold={r['goal_index']:r['asin'].lower() for r in goals}
        self.options=catalog()
        assert len(self.options)>100

    def event(self,g,state,text,action,cut,legal):
        event=dict(text=text,action=action,truncated=cut,legal=bool(legal))
        if not legal:
            public=state['public']
            clicks=[x for x in state['available']['clickables'] if x.lower()!='search']
            row=dict(action=action,page=public['page_type'],legal_clicks=clicks)
            label,_,detail=classify(row,self.gold[g],public,self.options)
            event.update(illegal_type=label,detail=detail)
        return event


def jobs():
    for arm,k in MODELS:
        for pool in POOLS:
            for mode in (('free','page') if k in (0,45,50) else ('free',)):
                for rank in range(4):
                    yield arm,k,pool,mode,rank


class ReplayEnvironments:
    """Independent native sessions, preserving the order of model requests and observations."""

    def __init__(self, workers, threads):
        self.workers = workers
        self.threads = threads
        self.owners = {}

    def call_many(self, requests):
        buckets = [[] for _ in self.workers]
        for i, request in enumerate(requests):
            key = request['key']
            if request['cmd'] == 'open':
                if key in self.owners:
                    raise ValueError(f'Duplicate active environment {key}')
                self.owners[key] = i % len(buckets)
            buckets[self.owners[key]].append((i, request))
        pending = [(group, self.threads.submit(worker.call, cmd='batch', requests=[r for _, r in group]))
                   for worker, group in zip(self.workers, buckets) if group]
        results = [None]*len(requests)
        for group, future in pending:
            for (i, request), response in zip(group, future.result(), strict=True):
                results[i] = response
                if request['cmd'] == 'close':
                    del self.owners[request['key']]
        return results

    def close(self):
        try:
            list(self.threads.map(lambda worker: worker.close(), self.workers))
        finally:
            self.threads.shutdown()


def create_runner(folder, checkpoint):
    # The independent environment initializations overlap vLLM initialization.
    threads = ThreadPoolExecutor(max_workers=4)
    futures = [threads.submit(environment, folder, f'replay_env_{i}') for i in (1, 2, 3)]
    runner = None
    try:
        runner = Runner(folder, checkpoint, parallel_start=True)
        extras = [f.result() for f in futures]
        if any(e.ready['tables'] != runner.engine.ready['tables'] for e in extras):
            raise ValueError('Frozen environment table mismatch')
        runner.engine = ReplayEnvironments([runner.engine, *extras], threads)
        return runner
    except BaseException:
        if runner is not None:
            runner.close()
        for future in futures:
            if future.exception() is None:
                future.result().close()
        threads.shutdown()
        raise


@contextmanager
def record_generated_tokens(runner):
    stream = sha256()
    model = runner.model
    original = model.call

    def measured(requests):
        rows = original(requests)
        for row in rows:
            stream.update((' '.join(map(str, row['token_ids']))+'\n').encode('ascii'))
        return rows

    model.call = measured
    try:
        yield stream
    finally:
        model.call = original


class ReplaySession:
    """Keep a single frozen model loaded; verify warm replay against a cold native batch."""

    def __init__(self):
        self.runner = None
        self.checkpoint = None
        self.reference = None
        self.verified = False
        self.verified_tokens = None
        self.disable_reuse = False

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.runner = None

    def reset(self):
        models = self.runner.model.models
        if len(models) != 1 or models[0].call(cmd='reset_replay_cache') != 'reset':
            raise RuntimeError('Replay cache reset failed')

    def acquire(self, folder, checkpoint):
        warm = self.runner is not None and self.checkpoint == checkpoint and not self.disable_reuse
        if warm:
            try:
                self.reset()
                if not self.verified:
                    if self.reference is None:
                        raise ValueError('Missing cold reference')
                    args, rule, original, tokens = self.reference
                    with record_generated_tokens(self.runner) as repeated_tokens:
                        repeated = run_batch(self.runner, *args, intervention=rule)
                    if repeated != original or repeated_tokens.hexdigest() != tokens:
                        raise ValueError('Warm replay produced a different native trajectory')
                    self.reset()
                    self.verified = True
                    self.verified_tokens = tokens
                    self.reference = None
            except (ValueError, RuntimeError, TimeoutError) as error:
                print('Cold-start fallback:', str(error), flush=True)
                self.disable_reuse = True
                warm = False
        if not warm:
            self.close()
            self.checkpoint = checkpoint
            self.runner = create_runner(folder, checkpoint)
            self.verified = False
            self.verified_tokens = None
            self.reference = None
        return self.runner, warm

    def evaluate_batch(self, args, rule):
        if not self.disable_reuse and not self.verified and self.reference is None:
            with record_generated_tokens(self.runner) as tokens:
                values = run_batch(self.runner, *args, intervention=rule)
            self.reference = (args, rule, values, tokens.hexdigest())
            return values
        return run_batch(self.runner, *args, intervention=rule)


def run(l,arm,k,pool,mode,rank,session):
    folder=ROOT/pool/mode/f'{arm}_{k:02}'/f'rank{rank}'
    snapshot_dir(folder)
    base=l.root/'evaluation'/pool/mode/f'{arm}_{k:02}'/f'rank{rank}'
    archived=read(base/'complete.json')
    tasks=split_tasks()[pool]
    goals=[t['goal_index'] for t in tasks if owners()[t['goal_index']]==rank]
    cp,signature=checkpoint(l,k,arm)
    # The archive reports the logical model hash, optimizer.json also pins the adapter file.
    assert goals==archived['goals'] and len(goals)==50 and archived['replicates']==5
    if k:
        opt=read(l.folder(k,arm)/'optimizer.json')
        assert archived['checkpoint'] in (signature,opt['after_sha'])
    else:
        assert archived['checkpoint']==signature
    if (folder/'complete.json').is_file():
        assert read(folder/'complete.json')['checkpoint']==signature
        return
    print('start',arm,k,pool,mode,rank,flush=True)
    rule=FirstCommand(tasks)
    start=time.monotonic()
    runner,warm=session.acquire(folder,cp)
    setup_time=time.monotonic()-start
    parts={}
    for rep in range(5):
        for offset in range(0,50,16):
            subset=goals[offset:offset+16]
            out=folder/f'part_{rep}_{offset:02}.json'
            if not out.exists():
                args=(pool,subset,rep,l.root/'pages'/pool/f'rank{rank}',mode)
                values=session.evaluate_batch(args,rule)
                save(out,values)
            values=read(out)
            assert len(values)==len(subset) and {r['goal'] for r in values}==set(subset) and all(r['rep']==rep for r in values)
            parts[out.name]=digest_file(out)
    save(folder/'complete.json',dict(checkpoint=signature,original_checkpoint=archived['checkpoint'],
         archive_sha256=digest_file(base/'complete.json'),pool=pool,mode=mode,goals=goals,seeds=SEEDS,
         logical_rank=rank,parts=parts,model_setup_seconds=setup_time,
         environment_workers=4,warm_model=warm,warm_verified=session.verified,validation_tokens_sha256=session.verified_tokens,total_seconds=time.monotonic()-start))


def main(gpu,lanes):
    from .e25_probe import configure
    os.environ.update(LOCAL_RANK=str(gpu),SHOP_GPU_ID=str(gpu),SHOP_GPU_IDS=str(gpu))
    assert configure()==gpu
    ROOT.mkdir(exist_ok=True)
    l=GapLayout()
    with (ROOT/f'worker{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        session=ReplaySession()
        try:
            for i,job in enumerate(jobs()):
                if i%lanes==gpu:
                    run(l,*job,session)
        finally:
            session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--lanes',type=int,default=2)
    args=p.parse_args()
    main(args.gpu,args.lanes)
