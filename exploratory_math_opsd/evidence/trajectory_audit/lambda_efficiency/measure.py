"""Offline archive metrics. Set OMP/BLAS threads to one per worker."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
import ast,bisect,gzip,hashlib,json,re,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from pydivsufsort import longest_previous_factor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'cap_trajectories'))
import cap_audit as cap
from cap_audit import org
from audit import answer
ARMS=['B','R-lambda0@100','R-lambda05@100','R@100','R-lambda15@100']
BUDGET={'aime24':[4096,8192,12288,16384,20480,24576,32768,38912],'nonthinking':[1024,2048,4096,6144,8192,12288,16384]}
func=next(x for x in ast.parse(cap.GRADER.read_text()).body if getattr(x,'name','')=='extract_boxed_answer')
ns={}
exec(compile(ast.Module(body=[func],type_ignores=[]),str(cap.GRADER),'exec'),ns)
official=ns['extract_boxed_answer']
PATTERNS={
 'sentence':re.compile(r'[^.!?\n]+(?:[.!?]+|\n|$)'),
 'paragraph':re.compile(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)'),
 'expression':re.compile(r'\$\$[\s\S]{1,700}?\$\$|\$[^$\n]{1,350}\$|\\\([\s\S]{1,350}?\\\)|\\\[[\s\S]{1,700}?\\\]'),
 'equation':re.compile(r'[^\n=$]{1,90}=[^\n=$]{1,90}'),
 'calculation':re.compile(r'[-+]?\d+(?:\.\d+)?\s*[-+*/×÷^]\s*[-+]?\d+(?:\.\d+)?\s*=\s*[-+]?\d+(?:\.\d+)?')}
MARKS={'reflect':re.compile(r'\b(?:wait|hmm|but|alternatively|let me (?:double[- ]check|check|verify))\b',re.I),
 'role':re.compile(cap.ROLE.pattern+r"|(?:^|\n)[ \t]*(?:assistant|human|user|system)(?:'s|’s) (?:thought process|response)\s*:",re.I),'control':re.compile(r'</?think>|<\|[^<>\n]{1,35}\|>'),
 'double_control':re.compile(r'(</?think>|<\|[^<>\n]{1,35}\|>)\s*\1'),
 'format_run':re.compile(r'\*{5,}|\x60{5,}|[$]{4,}|\\\\\\\\'),
 'offtopic_cue':re.compile(r'\bas an AI|\blanguage model|\bsubscribe to|\bfollow me on|\byoutube channel',re.I)}
def events(t,ends,gold):
    boxes=[]
    for m in re.finditer(r'\\boxed\s*\{',t):
        b=org.box(t[m.start():])
        if b:boxes.append((m.start()+b[0],b[1]))
    exp=boxes.copy()
    for m in org.CUE.finditer(t):
        v=org.VALUE.match(t,m.end())
        if v:
            val=next(x for x in v.groups() if x is not None).strip()
            if val:exp.append((v.end(),val))
    exp=sorted(set(exp))
    primary=org.first(t,org.box(t))
    if primary:exp=[primary]+[v for v in exp if v[0]>primary[0]]
    def encode(seq):
        return [(bisect.bisect_left(ends,p)+1,val,bool(cap.grade(val,gold))) for p,val in seq]
    return encode(exp),encode(sorted(set(boxes)))

def trajectory(t,g,q,ds):
    enc=tok.encode(t,add_special_tokens=False);ids=np.array(enc.ids,np.int32);off=enc.offsets
    n=len(ids);ends=[v for _,v in off]
    lpf=longest_previous_factor(ids) if n>1 else np.zeros(n,int)
    slices=[(0,n),(0,n//3),(n//3,2*n//3),(2*n//3,n),(max(0,n-1024),n)]
    seg=[dict(n=b-a,longest=int(np.minimum(lpf[a:b],b-np.arange(a,b)).max(initial=0))) for a,b in slices]
    for width in (4,8,16):
        starts=np.flatnonzero(lpf>=width); diff=np.zeros(n+1,int)
        diff[starts]+=1;diff[starts+width]-=1
        cum=np.r_[0,np.cumsum(diff[:-1].cumsum()>0)]
        for d,(a,b) in zip(seg,slices):d['cov'+str(width)]=int(cum[b]-cum[a])
    def save(key,positions):
        positions.sort()
        for d,(a,b) in zip(seg,slices):
            d[key]=bisect.bisect_left(positions,b)-bisect.bisect_left(positions,a)
    def at(end):return min(n-1,bisect.bisect_left(ends,end))
    for key,pat in PATTERNS.items():
        seen=set();allpos=[];reps=[]
        for m in pat.finditer(t):
            val=' '.join(m.group().split())
            if not val:continue
            pos=at(m.end());allpos.append(pos)
            if val in seen:reps.append(pos)
            seen.add(val)
        save(key,allpos);save('rep_'+key,reps)
    for key,pat in MARKS.items():save(key,[at(m.end()) for m in pat.finditer(t)])
    last=None;ls=[]
    for m in cap.LANGUAGE.finditer(t):
        if last and last!=m.lastgroup:ls.append(at(m.end()))
        last=m.lastgroup
    save('lang_switch',ls)
    prev=None;changes=[]
    for word in cap.SCRIPT.finditer(t):
        v=ord(word.group()[0])
        script=0 if v<128 else 1 if v<0x530 else 2 if v<0x700 else 3 if v<0x3100 else 4 if v<0xa000 else 5
        if prev is not None and prev!=script:changes.append(at(word.end()))
        prev=script
    save('script_switch',changes)
    exp,box=events(t,ends,q['ground_truth'])
    def pack(seq):
        cs=[z[2] for z in seq];vals=[z[1] for z in seq];wrong=[v for v,c in zip(vals,cs) if not c]
        return dict(first=seq[0] if seq else None,first_right=next((v[0] for v in seq if v[2]),None),
                    wr=sum(not a and b for a,b in zip(cs,cs[1:])),rw=sum(a and not b for a,b in zip(cs,cs[1:])),
                    changes=sum(a!=b for a,b in zip(vals,vals[1:])),repeated_wrong=len(wrong)-len(set(wrong)),count=len(seq),
                    final_matches_prior=g['predicted_answer'] in vals[:-1],last=seq[-1] if seq else None)
    truncated=[]
    for budget in BUDGET[ds]:
        if budget>=g['generated_tokens'] or budget>=n:truncated.append(bool(g['correct']))
        else:
            pred=official(t[:ends[budget-1]])
            truncated.append(bool(pred and cap.grade(pred,q['ground_truth'])))
    extracted=official(t)
    assert (extracted or None)==(g['predicted_answer'] if answer(g) is not None else None) or (not extracted and not g['formatted'])
    return dict(reencoded=n,tokens=g['generated_tokens'],cap=g['finish_reason']=='length',missing=answer(g) is None,
                correct=g['correct'],explicit=pack(exp),boxed=pack(box),official_budget=truncated,segments=seg,
                end_think='</think>' in t,formatted=g['formatted'])
def init():
    global tok
    tok=org.Tokenizer.from_file(str(cap.TOKENIZER))
def cell(job):
    ds,arm=job;x,h=org.source(ds,arm)
    assert x['seed']==1001 and x['val_n']==(12 if ds=='aime24' else 2)
    dest=HERE/(ds+'_'+arm+'.jsonl.gz');tmp=dest.with_suffix('.tmp')
    with gzip.open(tmp,'wt',compresslevel=3) as f:
        for i,q in enumerate(x['results']):
            for j,g in enumerate(q['generations']):
                r=trajectory(g['full_generation'],g,q,ds)
                r.update(q=i,j=j,hash=hashlib.sha256(q['problem'].encode()).hexdigest(),archive=h)
                f.write(json.dumps(r,separators=(',',':'))+'\n')
    tmp.replace(dest);print(ds,arm,flush=True)
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=6,initializer=init) as pool:
        list(pool.map(cell,[(d,m) for d in BUDGET for m in ARMS]))
