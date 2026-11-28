"""Score submitted edits without inspecting forward outputs. Production inputs stay read-only."""
import hashlib
import json
import re
import sys
from pathlib import Path
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
DATA=Path('/mnt/data2/ForkAudit/contrast-rewrite')
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
sys.path.insert(0,str(ROOT/'source/eval'))
from evaluate_math import extract_boxed_answer,grade_answer
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n')
def boxes(text):
    parts=re.split(r'(?=\\boxed\b)',text)
    return [x for part in parts if (x:=extract_boxed_answer(part)) and x.strip()]
def norm(s):
    s=re.sub(r'\\(?:text|textbf|mathbf)\{([^{}]*)\}',r'\1',s)
    return re.sub(r'\s+','',s).casefold()
def present(answer,text):
    # Whitespace-insensitive math, with explicit identifier/number boundaries for short answers.
    clean=norm(answer)
    if re.fullmatch(r'[\w.+-]+',clean):
        return bool(re.search(r'(?<![\w.])'+re.escape(clean)+r'(?![\w.])',text,re.I))
    return bool(clean and clean in norm(text))
def distance(a,b):
    row=list(range(len(b)+1))
    for i,t in enumerate(a,1):
        new=[i]
        for j,u in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(t!=u)))
        row=new
    return row[-1]
def headings(text):return re.findall(r'^\s{0,3}#{1,6}[ \t]+[^\n]*',text,re.M)
def verify(pool,manual):
    tok=Tokenizer.from_file(str(ROOT/'models/Qwen3-1.7B/tokenizer.json'))
    results=[]
    for r in pool:
        k=r['idx']; entry=manual[str(k)]
        old=r['solution']; h=hashlib.sha256(old.encode()).hexdigest()
        item=dict(idx=k,reference_sha256=h,qualified=False,**entry)
        if 'text' in entry:
            text=entry['text'];ref=boxes(old); bb=boxes(r['negative']); new=boxes(text)
            gold=r['Answer'] or ref[-1]
            target=bb[-1]; got=new[-1] if new else ''
            a=tok.encode(old,add_special_tokens=False).ids;b=tok.encode(text,add_special_tokens=False).ids
            dist=distance(a,b)
            checks=dict(box_match=bool(got) and (norm(target)==norm(got) or bool(grade_answer(got,target))),
                        wrong=bool(got) and not grade_answer(got,str(gold)),gold_absent=all(not present(str(v),text) for v in (gold,ref[-1] if ref else gold)),
                        edit_limit=dist<=len(a)/4,headings=headings(old)==headings(text),lines=len(old.splitlines())==len(text.splitlines()))
            item.update(qualified=all(checks.values()),checks=checks,edit_tokens=dist,reference_tokens=len(a),B_box=target,new_box=got)
        results.append(item)
    return results
def run(stage):
    path='pilot' if stage=='pilot' else 'diag_pool'
    pool=read(DATA/(path+'.json'))
    all_manual={}
    for f in DATA.glob('manual-*.json'):all_manual.update(read(f))
    results=verify(pool,all_manual)
    save(HERE/(stage+'.json'),results)
    print(stage,len(results),sum(r['qualified'] for r in results),flush=True)
if __name__=='__main__':run(sys.argv[1])
