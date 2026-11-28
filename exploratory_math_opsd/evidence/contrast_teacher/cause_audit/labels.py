"""Verify supervision selection from original archived positives/negatives."""
import json,re
from collections import Counter
from pathlib import Path
from behavior import extract,audit,D
W=Path('/mnt/data2/ForkAudit/contrast')
jobs=json.loads((W/'manifest.json').read_text())
chosen=json.loads((W/'negative.json').read_text())
raw={r['idx']:r['answers'] for path in sorted(W.glob('batch-*.json')) for r in json.loads(path.read_text())}
proofs=json.loads((D.parent/'proof_review.json').read_text())
out=Counter();flags=[]
for j in jobs[:3200]:
    t=chosen[j['problem']]
    if t is None:continue
    out['active']+=1
    rows=raw[j['idx']];sample=next(g for g in rows if g['text']==t)
    if str(j['idx']) in proofs:
        out['proof_review']+=1;continue
    assert sample['correct'] is False
    pos=extract(j['solution']);gold=str(j['Answer'] or pos)
    neg_ans=extract(t)
    if not neg_ans:
        starts=[m.start() for m in re.finditer(re.escape(r'\boxed'),t)]
        neg_ans=next(a for a in reversed([extract(t[a:b]) for a,b in zip(starts,starts[1:]+[len(t)])]) if a)
    assert neg_ans
    nright=bool(audit.grade(neg_ans,gold));pright=bool(audit.grade(pos,gold))
    out['negative_matches_gold']+=nright;out['positive_box_mismatch_or_missing']+=not pright
    if nright or not pright:flags.append(dict(idx=j['idx'],gold=gold,negative=neg_ans,positive=pos))
(D/'labels.json').write_text(json.dumps(dict(counts=out,flagged=flags),indent=1,ensure_ascii=False)+'\n')
