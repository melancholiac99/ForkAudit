"""Reconstruct only public native states for the frozen smoke tapes, without GPU."""
import argparse
from pathlib import Path
import os
import time
from transformers import AutoTokenizer
from exploratory_shop_audit.grpo_forced_audit.common import read_rows, MODEL, append, save, digest_file
from exploratory_shop_audit.grpo_forced_audit.checks import environment
from exploratory_shop_audit.runtime import make_input
from .parser import categories

def repair(run):
    os.umask(0o077)
    dest=run/'parser_repair.jsonl'
    tmp=run/f'parser_partial_{time.time_ns()}.jsonl'
    if dest.exists():raise FileExistsError(dest)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    engine=environment(run,'cpu_parser_replay')
    try:
        for b in sorted(run.glob('smoke_batch_*')):
            all_rows=read_rows(b/'steps.jsonl')
            eps=[e for e in read_rows(b/'episodes.jsonl') if e['mode']=='none']
            for ep in eps:
                i,r=ep['goal'],ep['rep'];key=f'vc_{i}_none_{r}'
                state=engine.call(cmd='open',key=key,goal_index=i)
                history=[]
                for row in (x for x in all_rows if x['mode']=='none' and x['goal']==i and x['rep']==r):
                    native=make_input(tok,state['task'],history,state['observation'])
                    assert native['token_ids']==row['prompt_ids'] and row['step']==len(history)
                    append(tmp,dict(goal=i,rep=r,step=row['step'],classes=categories(tok,row['response_ids'],state['public'])))
                    history.append((state['observation'],row['text']))
                    outcome=engine.call(cmd='step',key=key,action=row['action'])
                    state=outcome['state']
                assert outcome['status']['done'] and outcome['status']['termination_reason']==ep['reason']
                engine.call(cmd='close',key=key)
    finally:engine.close()
    tmp.rename(dest)
    save(run/'parser_repair_manifest.json',dict(original_parser=digest_file(Path(__file__).with_name('packets.py')),new_parser=digest_file(Path(__file__).with_name('parser.py')),rows=len(read_rows(dest)),sidecar_sha=digest_file(dest),validation='reconstructed exact native input IDs at each step and matched every terminal reason',additional_gpu_hours=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);repair(p.parse_args().run)
