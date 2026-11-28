"""Validate production drafts and produce a compact, per-record human review queue."""
import json
import re
from check import HERE,DATA,read,save,verify
from expand import pool,done

def collect():
    jobs={r['idx']:r for r in pool()}
    previous=done();drafts={}
    for path in DATA.glob('drafts-*.json'):
        for v in read(path):
            assert v['idx'] not in drafts
            drafts[v['idx']]=v['draft']
    assert set(drafts)==set(jobs)-set(previous)
    attempts={}
    for i,raw in drafts.items():
        try:
            start=raw.index('{')
            obj=json.JSONDecoder().raw_decode(raw[start:])[0]
            evidence=obj['evidence'].strip()
            if not evidence or evidence not in jobs[i]['negative']:raise ValueError('B evidence not found verbatim')
            edits=obj['edits']
            if not isinstance(edits,list) or not edits:raise ValueError('no edits')
            s=jobs[i]['solution']
            for old,new in edits:
                if not old or old not in s:raise ValueError('edit not found verbatim')
                s=s.replace(old,new)
            attempts[str(i)]=dict(text=s,error=obj['error'],evidence=evidence,edits=edits)
        except (ValueError,KeyError,TypeError) as e:
            attempts[str(i)]={'reason':str(e)}
    result=verify([jobs[i] for i in drafts],attempts)
    save(DATA/'validated.json',result)
    pending=[r for r in result if r['qualified']]
    save(DATA/'review.json',[dict(idx=r['idx'],problem=jobs[r['idx']]['problem'],error=r['error'],evidence=r['evidence'],edits=r['edits']) for r in pending])
    print('Eligible for semantic review',len(pending))
if __name__=='__main__':collect()
