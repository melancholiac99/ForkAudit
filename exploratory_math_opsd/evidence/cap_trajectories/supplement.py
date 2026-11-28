"""Resolve archive-only comparison controls and attach manual semantic checks."""
import bisect
import json
import re
from collections import Counter
from cap_audit import HERE, MODELS, TOKENIZER, org, segments, dump

reviews=json.loads((HERE/'review_evidence.json').read_text())
tok=org.Tokenizer.from_file(str(TOKENIZER))
manual=[]
for ds in ('aime24','nonthinking'):
    base,h=org.source(ds,'B')
    arms={m:org.source(ds,m)[0] for m in MODELS}
    arms['B']=base
    for v in reviews:
        bench,m,qid,j=v['id']
        if bench!=ds:
            continue
        qi=v['question_index']
        q=arms[m]['results'][qi]
        assert q['problem']==v['problem']
        c=v['control']
        if c is None:
            b=next(x for x in base['results'] if x['problem']==v['problem'])
            for sample,g in enumerate(b['generations']):
                if g['finish_reason']=='stop':
                    c=dict(model='B',sample=sample,tokens=g['generated_tokens'],correct=g['correct'],end=g['full_generation'][-380:],source_sha256=h)
                    v['control']=c
                    break
        if c:
            cq=next(x for x in arms[c['model']]['results'] if x['problem']==v['problem'])
            gen=cq['generations'][c['sample']]
            enc=tok.encode(gen['full_generation'],add_special_tokens=False)
            c['quarters']=segments(gen['full_generation'],enc)
        if (ds,m,qid,j)==('aime24','R-lambda15@100',77,7):
            t=q['generations'][j]['full_generation'];end=t.index('1 * 1 + 100 * 6 = 1 + 600 = 601.')+len('1 * 1 + 100 * 6 = 1 + 600 = 601')
            ends=[b for _,b in tok.encode(t,add_special_tokens=False).offsets]
            n=bisect.bisect_left(ends,end)+1
            manual.append(dict(id=v['id'],question_index=qi,token=n,remaining=len(ends)-n,snippet=t[end-480:end+240]))
dump(HERE/'review_evidence.json',reviews)
dump(HERE/'manual_correct.json',manual)
