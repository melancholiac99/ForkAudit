"""Replay §26 with one execution-only change, the first of successive commands."""
import argparse
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import re

from .e2512_native import checkpoint, snapshot_dir
from .e2512_routing import owners
from .gap_cpu_actions import classify
from .gap_eval import run_batch, SEEDS
from .gap_run import GapLayout
from .gap_score import RUN, PUBLIC, split_tasks
from exploratory_shop_audit.grpo_forced_audit.common import SOURCE, digest_file, read, save
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner

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


def run(l,arm,k,pool,mode,rank):
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
    runner=Runner(folder,cp,parallel_start=True)
    parts={}
    try:
        for rep in range(5):
            for offset in range(0,50,16):
                subset=goals[offset:offset+16]
                out=folder/f'part_{rep}_{offset:02}.json'
                if not out.exists():
                    values=run_batch(runner,pool,subset,rep,l.root/'pages'/pool/f'rank{rank}',mode,intervention=rule)
                    save(out,values)
                values=read(out)
                assert len(values)==len(subset) and {r['goal'] for r in values}==set(subset) and all(r['rep']==rep for r in values)
                parts[out.name]=digest_file(out)
    finally:
        runner.close()
    save(folder/'complete.json',dict(checkpoint=signature,original_checkpoint=archived['checkpoint'],
         archive_sha256=digest_file(base/'complete.json'),pool=pool,mode=mode,goals=goals,seeds=SEEDS,
         logical_rank=rank,parts=parts))


def main(gpu,lanes):
    os.environ.update(SHOP_GPU_ID=str(gpu),SHOP_GPU_IDS=str(gpu))
    ROOT.mkdir(exist_ok=True)
    l=GapLayout()
    with (ROOT/f'worker{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i,job in enumerate(jobs()):
            if i%lanes==gpu:
                run(l,*job)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--lanes',type=int,default=2)
    args=p.parse_args()
    main(args.gpu,args.lanes)
