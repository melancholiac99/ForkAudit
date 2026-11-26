import sys
import time
from common_io import *
ids=[25*i+12 for i in range(20)]
ans={}
for arm in ('B','R-cond@100'):
    rows=[g for i in ids for g in read(RAW/arm/f'{i}.json')]
    assert len(rows)==40 and all(0<g['tokens']<=16384 and 0<=g['insertions']<=8 for g in rows)
    ans[arm]=dict(answers=len(rows),insertions=int(sum(g['insertions'] for g in rows)))
save(HERE/'pilot.json',dict(wall_seconds=time.time()-int(sys.argv[1]),questions=20,rows=ids,arms=ans))
