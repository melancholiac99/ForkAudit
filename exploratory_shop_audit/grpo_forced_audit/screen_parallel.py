import argparse
import subprocess
import sys
from .common import *
from .cpu import screen

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--shard',type=int);a=p.parse_args()
    if a.shard is not None:screen(a.run,a.shard,12);return
    processes=[]
    for i in range(12):
        log=(a.run/f'screen_{i}.log').open('a')
        processes.append((subprocess.Popen([sys.executable,'-m',__package__+'.screen_parallel','--run',str(a.run),'--shard',str(i)],stdout=log,stderr=log,cwd=ROOT),log))
    codes=[]
    for proc,log in processes:codes.append(proc.wait());log.close()
    assert codes==[0]*12,codes
    rows=read_rows(a.run/'screening_completed_before_parallel_resize.jsonl')+sum([read_rows(a.run/f'script_screening12_{i:02d}.jsonl') for i in range(12)],[])
    assert len(rows)==len({r['task_id'] for r in rows})
    with (a.run/'script_screening.jsonl').open('w') as f:
        for r in sorted(rows,key=lambda r:int(r['task_id'].split(':')[1])):f.write(json.dumps(r,ensure_ascii=False)+'\n')
    from collections import Counter
    save(a.run/'screening_summary.json',{'tasks':len(rows),'labels':dict(Counter(r['label'] for r in rows))})
    from .cpu import split
    split(a.run)

if __name__=='__main__':main()
