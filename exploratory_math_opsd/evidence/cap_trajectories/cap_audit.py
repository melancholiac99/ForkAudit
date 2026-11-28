"""Offline cap audit. No checkpoint or inference dependency."""
import ast
import bisect
import hashlib
import json
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from math_verify import parse, verify

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'reasoning_organization'))
import analyze as org
sys.path.insert(0, str(HERE.parent / 'trajectory_audit'))
from audit import answer as archived_answer

SEED = 20261124
MODELS = ('R@100', 'R-lambda15@100')
TOKENIZER = org.ROOT / 'models/Qwen3-1.7B/tokenizer.json'
GRADER = org.ROOT / 'source/eval/evaluate_math.py'
func = next(n for n in ast.parse(GRADER.read_text()).body if getattr(n, 'name', '') == 'grade_answer')
namespace = dict(parse=parse, verify=verify)
exec(compile(ast.Module(body=[func], type_ignores=[]), str(GRADER), 'exec'), namespace)
grade = lru_cache(32768)(namespace['grade_answer'])
REFLECT = re.compile(r'\b(?:wait|but|hmm|alternatively)\b', re.I)
MATH = re.compile(r'\\[A-Za-z]+|(?<=[\w)])\s*[=+*/^<>−×÷]\s*(?=[\w(\\])')
ROLE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>|(?:^|\n)[ \t]*(?:assistant|user|system|human)\s*:", re.I)
SCRIPT = re.compile(r"[A-Za-z]+|[\u3400-\u9fff]+|[\u0400-\u052f]+|[\uac00-\ud7af]+|[\u3040-\u30ff]+|[\u0600-\u06ff]+")


LANGUAGE = re.compile(r"(?P<en>\b(?:therefore|thus|let us|let me|we have|we get|the answer)\b)|(?P<es>\b(?:de aquí|por lo tanto|entonces|por tanto)\b)|(?P<vi>\b(?:ta có|tuy nhiên|để tính|vì vậy)\b)|(?P<fr>\b(?:donc|nous avons|ainsi)\b)|(?P<zh>所以|因此|我们|因为)|(?P<ru>следовательно|таким образом)",re.I)


def segments(txt, enc):
    seen = set()
    paras = set()
    previous = None
    last_language = None
    out = []
    ids = enc.ids
    for k in range(4):
        lo, hi = k*len(ids)//4, (k+1)*len(ids)//4
        start, end = enc.offsets[lo][0], enc.offsets[hi-1][1]
        sub = txt[start:end]
        rep = 0
        for i in range(max(4, lo), hi):
            gram = tuple(ids[i-4:i+1])
            rep += gram in seen
            seen.add(gram)
        switches = 0
        for word in SCRIPT.findall(sub):
            s = ord(word[0])
            group = 'latin' if s<128 else 'cyrillic' if s<0x530 else 'arabic' if s<0x700 else 'kana' if s<0x3100 else 'han' if s<0xa000 else 'hangul'
            switches += previous is not None and group != previous
            previous = group
        language_switch = 0
        # Conservative identifiable vocabulary. Track across quarter boundaries.
        for mark in LANGUAGE.finditer(sub):
            lang = mark.lastgroup
            language_switch += last_language is not None and lang != last_language
            last_language = lang
        out.append(dict(language_switch=language_switch,tokens=hi-lo, repeated_5gram=rep, grams=hi-max(4,lo),
                        paragraphs=0, repeated_paragraphs=0, math=len(MATH.findall(sub)),
                        reflection=len(REFLECT.findall(sub)), role=len(ROLE.findall(sub)), script_switch=switches))
    ends = [e for _,e in enc.offsets]
    for m in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',txt):
        p = ' '.join(m.group().split())
        k = min(3, 4*bisect.bisect_left(ends,m.end())//len(ids))
        out[k]['paragraphs'] += 1
        out[k]['repeated_paragraphs'] += p in paras
        paras.add(p)
    return out


def candidates(text):
    offset=0
    while offset<len(text):
        sub=text[offset:]
        item=org.first(sub, org.box(sub))
        if not item:
            break
        end,value=item
        offset+=end
        yield offset,value


def dump(path,obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1)+'\n')


def main():
    tok = org.Tokenizer.from_file(str(TOKENIZER))
    totals = {}
    capped = []
    excerpts = []
    for ds in ('aime24','nonthinking'):
        archives = {m:org.source(ds,m) for m in MODELS}
        assert [q['problem'] for q in archives[MODELS[0]][0]['results']]==[q['problem'] for q in archives[MODELS[1]][0]['results']]
        for arm in MODELS:
            data,h=archives[arm]
            rows=data['results']
            groups={'stop':[], 'length':[]}
            allcaps=[(i,j) for i,q in enumerate(rows) for j,g in enumerate(q['generations']) if g['finish_reason']=='length']
            selected=set(allcaps if ds=='aime24' else random.Random(SEED).sample(allcaps,8))
            print(ds, arm,flush=True)
            for i,q in enumerate(rows):
                for j,g in enumerate(q['generations']):
                    reason=g['finish_reason']
                    n=g['generated_tokens']
                    groups[reason].append(n)
                    if reason!='length':
                        continue
                    text=g['full_generation']
                    enc=tok.encode(text, add_special_tokens=False)
                    ends=[e for _,e in enc.offsets]
                    def snippet(pos):
                        return dict(char=pos,token=bisect.bisect_left(ends,pos)+1,text=text[max(0,pos-280):pos+180])
                    items=list(candidates(text))
                    hit=next(((pos,v) for pos,v in items if grade(v,q['ground_truth'])),None)
                    assert bool(grade(g['predicted_answer'],q['ground_truth']))==bool(g['correct']) if archived_answer(g) is not None else not g['correct']
                    row=dict(question_index=i,ds=ds,arm=arm,question=q['problem_id'],sample=j,problem=q['problem'],gold=q['ground_truth'],tokens=n,
                             reencoded=len(ends),final=g['predicted_answer'],correct=g['correct'],missing=archived_answer(g) is None,
                             explicit_count=len(items),first_explicit=items[0] if items else None,first_correct=None,
                             quarters=segments(text,enc),reviewed=(i,j) in selected)
                    if hit:
                        pos,val=hit
                        row['first_correct']=dict(answer=val,**snippet(pos),remaining_reencoded=len(ends)-bisect.bisect_left(ends,pos)-1)
                    capped.append(row)
                    if (i,j) in selected:
                        controls=[]
                        for model in (arm,*[m for m in MODELS if m!=arm]):
                            for index,other in enumerate(archives[model][0]['results'][i]['generations']):
                                if other['finish_reason']=='stop':
                                    controls.append(dict(model=model,sample=index,correct=other['correct'],tokens=other['generated_tokens'],
                                                         end=other['full_generation'][-380:]))
                            if controls:
                                break
                        char_bounds=[0,*[bisect.bisect_left(ends,k*len(text)//4)+1 for k in (1,2,3)],len(ends)]
                        excerpts.append(dict(question_index=i,char_quarter_token_bounds=char_bounds,
                                            midpoint_evidence=[snippet(int(v*len(text))) for v in (.125,.375,.625,.875)],id=[ds,arm,q['problem_id'],j],problem=q['problem'],gold=q['ground_truth'],
                                            samples=[snippet(ends[min(len(ends)-1,int(t*len(ends)))]) for t in (.05,.25,.5,.75,.95,.9999)],
                                            first_correct=row['first_correct'],control=(controls or [None])[0]))
            totals[ds+' '+arm]=dict(sha256=h,groups={r:dict(count=len(ns),total=sum(ns),min=min(ns),p25=float(np.percentile(ns,25)),median=float(np.median(ns)),mean=float(np.mean(ns)),p75=float(np.percentile(ns,75)),p95=float(np.percentile(ns,95)),max=max(ns)) for r,ns in groups.items()})
    dump(HERE/'capped.json',capped)
    dump(HERE/'review_evidence.json',excerpts)
    dump(HERE/'lengths.json',totals)


if __name__=='__main__':
    main()
