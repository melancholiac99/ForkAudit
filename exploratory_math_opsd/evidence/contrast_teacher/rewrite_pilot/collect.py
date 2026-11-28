"""Validate production drafts and produce a compact, per-record human review queue."""
import json
import re
from check import HERE,DATA,read,save,verify,boxes,norm
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
    # Recover equivalent drafts that could not be represented as machine-readable
    # substring edits. Preserve the archive's original blank lines and indentation.
    full={}
    for path in DATA.glob('full-*.json'):
        for v in read(path): full[v['idx']]=v['text']
    assert set(full)==set(drafts)
    extra={}
    for i,model_text in full.items():
        target=boxes(jobs[i]['negative'])[-1]
        original=jobs[i]['solution']
        generated=model_text.strip().splitlines()
        if generated and generated[0].startswith('```') and generated[-1]=='```':generated=generated[1:-1]
        lines=[line.strip() for line in generated if line.strip()]
        if len(lines)!=sum(bool(line.strip()) for line in original.splitlines()):continue
        it=iter(lines)
        candidate=''.join(line if not line.strip() else line.replace(line.strip(),next(it),1) for line in original.splitlines(keepends=True))
        final=boxes(candidate)
        if not final or norm(final[-1])!=norm(target):continue
        extra[str(i)]=dict(text=candidate,error='Pending independent comparison to the archived B error',evidence=target)
    result=verify([jobs[i] for i in drafts],attempts)
    second=verify([jobs[int(i)] for i in extra],extra)
    alternates={r['idx']:r for r in second if r['qualified']}
    result=[r if r['qualified'] else alternates.get(r['idx'],r) for r in result]
    save(DATA/'validated.json',result)
    pending=[r for r in result if r['qualified']]
    from difflib import unified_diff
    save(DATA/'review.json',[dict(idx=r['idx'],problem=jobs[r['idx']]['problem'],error=r['error'],evidence=r['evidence'],B=jobs[r['idx']]['negative'],diff='\n'.join(unified_diff(jobs[r['idx']]['solution'].splitlines(),r['text'].splitlines(),n=1,lineterm=''))) for r in pending])
    print('Eligible for semantic review',len(pending))
if __name__=='__main__':collect()
