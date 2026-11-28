"""Compare identical submitted edits and recover full drafts skipped by the old line prefilter."""
from check import HERE,DATA,read,save,headings,verify,boxes,norm
from expand import pool
from collections import Counter
def main():
    jobs={j['idx']:j for j in pool()}
    counts={};updates={};pending=[]
    def assess(rows,name,rewrite_file=None):
        n=Counter();out=[]
        for r in rows:
            x=dict(r); old=jobs[r['idx']]['solution'];txt=r.get('text')
            if txt is not None:
                checks=dict(r['checks'])
                checks['lines']=10*abs(len(txt.splitlines())-len(old.splitlines()))<=len(old.splitlines())
                checks['headings']=len(headings(old))==len(headings(txt))
                x.update(checks=checks,qualified=all(checks.values()))
                if x['qualified'] and not r['qualified']:updates[str(r['idx'])]=x
            n.update(total=1,before=int(r['qualified']),after=int(x['qualified']));out.append(x)
        counts[name]=dict(n)
        if rewrite_file:save(rewrite_file,out)
    assess(read(HERE/'pilot.json'),'pilot')
    assess(read(HERE/'diagnostic_edits.json'),'diagnostic')
    for p in sorted((DATA/'editor').glob('checked-*.json')):
        assess(read(p),p.stem,p)
    base=read(DATA/'validated-base.json')
    old_sem=read(DATA/'semantic-base.json')
    known={r['idx'] for r in base if r['qualified']}
    for path in DATA.glob('full-*.json'):
        for v in read(path):
            i=v['idx']
            if i in known:continue
            j=jobs[i];orig=j['solution'];lines=v['text'].strip().splitlines()
            if lines and lines[0].strip().rstrip(':').casefold() in {'reference to copy and minimally edit','modified reference','edited reference','rewritten reference'}:lines=lines[1:]
            while lines and not lines[0].strip():lines.pop(0)
            if len(lines)>1 and lines[0].startswith('```') and lines[-1]=='```':lines=lines[1:-1]
            text='\n'.join(lines)
            if orig.startswith('\n'):text='\n'+text
            if orig.endswith('\n'):text+='\n'
            if 10*abs(len(text.splitlines())-len(orig.splitlines()))>len(orig.splitlines()) or len(headings(orig))!=len(headings(text)):continue
            b=boxes(text)
            if not b or norm(b[-1])!=norm(boxes(j['negative'])[-1]):continue
            rr=verify([j],{str(i):dict(text=text,error='pending review')})[0]
            if rr['qualified']:pending.append(rr)
    counts['local_model']=dict(tasks=len(base),old_auto=len(known),old_review=sum(v['pass'] for v in old_sem.values()),new_auto_pending=len(pending))
    save(DATA/'format_relaxed_pending.json',pending)
    save(DATA/'format_relaxed_updates.json',updates)
    save(HERE/'line_recheck.json',counts)
if __name__=='__main__':main()
