"""CPU-only, paired measurements from the original decoded answer archives."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import re,sys,json,hashlib,bisect
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
OUT=ROOT/'run/reasoning-organization-final'
OUT.mkdir(exist_ok=True)
SEED=20261124
SETS=('nonthinking','thinking','aime24','aime25','hmmt25')
ARMS=('B','R@100')
WAIT=re.compile(r'\bwait\b',re.I)
VERIFY=re.compile(r'\b(?:double[- ]check|check|verify|verification|recheck|verifying)\b',re.I)
REFLECT=re.compile(r'^\W*(?:wait|hmm|but|alternatively)\b',re.I)
CUE=re.compile(r'\b(?:(?:the|my|our)\s+(?:final\s+)?|final\s+)answer\s*(?:is\b|(?:should|would|must)\s+be\b|[:=])\s*(?:[:=]\s*)?',re.I)
VALUE=re.compile(r'\$\$(.+?)\$\$|\$(?!\$)([^$\n]+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]|([+-]?\d+(?:\.\d+)?(?:/\d+)?)(?![\w/^+*%-]|\.\d|\s*[/^+*=])(?=[\s,;!?)\n]|\.(?:\s|$)|$)',re.S)
def box(text):
    for m in re.finditer(r'\\boxed\s*\{',text):
        depth=1
        for j in range(m.end(),len(text)):
            c=text[j]
            if c=='{':depth+=1
            if c=='}':depth-=1
            if depth==0:
                val=text[m.end():j].strip()
                if val:return j+1,val
                break
    return None
def first(text,b):
    for m in CUE.finditer(text):
        if b and b[0]<=m.end():break
        v=VALUE.match(text,m.end())
        if v:
            val=next(x for x in v.groups() if x is not None).strip()
            if val and (not b or v.end()<b[0]):return v.end(),val
    return b
def source(ds,arm):
    if ds in SETS[:2]:
        d=ROOT/'run/math500-long-seed1001-n2'/f'{ds}-{arm}'
        p=d/'answers.json';h=json.loads((d/'receipt.json').read_text())['answers_sha256']
    else:
        d=ROOT/'run/evaluation'/arm/ds
        p=d/'1001.json';h=json.loads((d/'1001.receipt.json').read_text())['sha256']
    s=p.read_bytes();assert hashlib.sha256(s).hexdigest()==h
    return json.loads(s),h
def init():
    global tok,grade
    tok=Tokenizer.from_file(str(ROOT/'models/Qwen3-1.7B/tokenizer.json'))
    sys.path.insert(0,str(ROOT/'source/eval'))
    from evaluate_math import grade_answer
    grade=lru_cache(65536)(grade_answer)
def chunk(task):
    ds,arm,lane=task
    dest=OUT/f'{ds}-{arm}-{lane}.json'
    if dest.exists():return
    data,h=source(ds,arm)
    assert data['seed']==1001 and data['enable_thinking']==(ds!='nonthinking')
    rows=[]
    for qi in range(lane,len(data['results']),4):
        q=data['results'][qi]
        for j,g in enumerate(q['generations']):
            txt=g['full_generation']; encoded=tok.encode(txt,add_special_tokens=False)
            offsets=encoded.offsets;n=len(offsets);ends=[e for _,e in offsets]
            rng=np.random.default_rng([SEED,SETS.index(ds),ARMS.index(arm),qi,j])
            loc=[bisect.bisect_right(ends,m.start()) for m in WAIT.finditer(txt)]
            pairs=max(0,len(loc)-1); near=int(np.sum(np.diff(loc)<=64))
            null=0.
            if pairs:
                for _ in range(128):
                    null+=np.sum(np.diff(np.sort(rng.choice(n,len(loc),replace=False)))<=64)/128
            paras=[v.strip() for v in re.split(r'\n[ \t]*\n+',txt) if v.strip()]
            checks=[bool(VERIFY.search(' '.join(re.findall(r"[A-Za-z]+",v)[:18]))) for v in paras]
            more=nextpart=terminal=0
            for k,c in enumerate(checks):
                if not c:continue
                if k+1==len(checks):terminal+=1
                elif checks[k+1] or REFLECT.match(paras[k+1]):more+=1
                else:nextpart+=1
            row=dict(question=hashlib.sha256(q['problem'].encode()).hexdigest(),sample=j,n=n,native=g['generated_tokens'],wait=len(loc),pairs=pairs,near=near,null=null,more=more,other=nextpart,end=terminal,final=int(g['correct']))
            if ds!='nonthinking':
                b=box(txt)
                for rule,item in [('explicit',first(txt,b)),('boxed',b)]:
                    prefix=n if not item else min(n,bisect.bisect_left(ends,item[0])+1)
                    right=(bool(g['correct']) if item and item[1]==g['predicted_answer'] else bool(grade(item[1],q['ground_truth'])) if item else False)
                    row[rule]=[prefix,n-prefix,int(bool(item)),int(bool(item) and not right),int(bool(item) and not right and g['correct']),int(right and not g['correct'])]
            rows.append(row)
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(dict(sha=h,rows=rows)));tmp.replace(dest)
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=6,initializer=init) as pool:
        list(pool.map(chunk,[(d,a,k) for d in SETS for a in ARMS for k in range(4)],chunksize=1))
