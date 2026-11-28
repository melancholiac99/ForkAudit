"""Add four independently seeded B generations to the exact first 100-step stream."""
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import contexts as old
from contexts import ROOT,MODEL,read,save,sha
HERE=Path(__file__).resolve().parent
WORK=old.WORK.parent/'contrast-v2'
WORK.mkdir(exist_ok=True)
def sample(lane):
    old.SEED=20261128
    old.WORK=WORK
    old.sample(lane)

def combine():
    sys.path.insert(0,str(ROOT/'source/eval'))
    from evaluate_math import extract_boxed_answer,grade_answer
    def load(path):
        return {row['idx']:row['answers'] for f in path.glob('batch-*.json') for row in read(f)}
    before,extra=load(old.WORK),load(WORK)
    maps={}
    covered=[0,0]
    for j in read(WORK/'manifest.json'):
        idx=j['idx']
        assert len(before[idx])==len(extra[idx])==4
        pos=neg=None
        for n,v in enumerate(before[idx]+extra[idx],1):
            text=v['text']
            gold=j['Answer'] or extract_boxed_answer(j['solution'])
            # An ungraded proof is ineligible until an answer has an actual grade.
            if gold and r'\boxed' in text:
                starts=[m.start() for m in re.finditer(re.escape(r'\boxed'),text)]
                box=next((b for a,z in reversed(list(zip(starts,starts[1:]+[len(text)]))) if (b:=extract_boxed_answer(text[a:z]))),None)
                if box:
                    correct=bool(grade_answer(box,str(gold)))
                    if correct and pos is None:pos=text
                    if not correct and neg is None:neg=text
            if n in (4,8):covered[n//4-1]+=int(bool(pos and neg))
        maps[j['problem']]=[pos,neg]
    save(WORK/'pairs.json',maps)
    save(HERE/'coverage.json',dict(tasks=len(maps),four=covered[0],eight=covered[1],pairs_sha256=sha(WORK/'pairs.json'),seed=20261128))
if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':
        save(WORK/'manifest.json',read(old.WORK/'manifest.json')[:3200])
    elif action=='sample':sample(int(sys.argv[2]))
    else:combine()
