"""CPU post-hoc audit of archived MATH-500 answers."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'reasoning_organization'))
import analyze as org

SEED = 20261124
BOUNDARY = 'R-clip-boundary@100'
MODELS = (BOUNDARY, 'R@100', 'B', 'R-noclip@100')
METRICS = ('first', 'wrong_to_correct', 'correct_to_wrong', 'final', 'coverage')
BOX = re.compile(r'\\boxed\s*\{')


def archive(arm, budget, thinking=False):
    key = ('thinking-' if thinking else 'nonthinking-')+arm
    folder = org.ROOT/'run'/('math500-trajectory-k20' if budget==4096 else 'math500-long-seed1001-n2')/(arm if budget==4096 else key)
    raw = (folder/'answers.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.loads((folder/'receipt.json').read_text())
    assert digest == receipt['output_sha256' if budget==4096 else 'answers_sha256']
    data = json.loads(raw)
    assert data['seed']==1001 and data['max_new_tokens']==budget and data['enable_thinking']==thinking
    assert len(data['results'])==500
    assert all(len(q['generations'])==(4 if budget==4096 else 2) for q in data['results'])
    return data['results'], digest


def boxes(text):
    values=[]
    for m in BOX.finditer(text):
        depth=1
        for end in range(m.end(),len(text)):
            depth += (text[end]=='{')-(text[end]=='}')
            if depth==0:
                val=text[m.end():end].strip()
                if val: values.append(val)
                break
    return values


def summary(draws):
    return [round(float(draws[0]),6), *np.quantile(draws[1:],[.025,.975]).round(6).tolist()]


def compare(budget, w):
    models={}; hashes={}; questions=None
    profile=Counter(); distribution=Counter()
    for arm in MODELS:
        data,hashes[arm]=archive(arm,budget)
        ids=[q['problem'] for q in data]
        if questions is None: questions=ids
        assert len(set(ids))==500 and ids==questions
        rows=[]
        for q in data:
            samples=[]
            for g in q['generations']:
                txt=g['full_generation']; final=int(g['correct'])
                b=org.box(txt); e=org.first(txt,b)
                def correct(val):
                    return int(final if val==g['predicted_answer'] else org.grade(val,q['ground_truth'])) if val is not None else 0
                first=correct(e[1] if e else None)
                samples.append([first,(1-first)*final,first*(1-final),final,int(bool(e))])
                if arm==BOUNDARY:
                    vals=boxes(txt)
                    assert (vals[0] if vals else None)==(b[1] if b else None)
                    distribution[len(vals)]+=1
                    profile['total']+=1
                    profile['open']+=int('<think>' in txt)
                    profile['close']+=int('</think>' in txt)
                    profile['either']+=int('<think>' in txt or '</think>' in txt)
                    profile['contains_boxed_command']+=int('\\boxed' in txt)
                    profile['contains_complete_box']+=int(bool(vals))
                    profile['first_box_correct']+=correct(vals[0] if vals else None)
                    if len(vals)>=2:
                        profile['multiple']+=1
                        different=vals[0]!=vals[-1]
                        profile['different_text']+=int(different)
                        profile['different_by_grader']+=int(different and not org.grade(vals[-1],vals[0]))
            rows.append(np.mean(samples,axis=0))
        models[arm]=w@np.array(rows)*100
        np.testing.assert_allclose(models[arm][:,0]+models[arm][:,1]-models[arm][:,2],models[arm][:,3],atol=1e-10)
        if arm==BOUNDARY and budget==16384:
            idxs=np.random.default_rng(SEED).choice(1000,10,replace=False)
            lines=['# R-clip-boundary@100 非思考 16k 原文样例','',
                   '按归档题序、采样序展平，以 NumPy default_rng(20261124) 无放回均匀抽 10 条。字符按 Unicode 码点计。']
            for i in idxs:
                qi,gi=divmod(int(i),2);q=data[qi];g=q['generations'][gi];txt=g['full_generation']
                lines+=['',f"## 题序 {qi+1}，采样 {gi+1}，{g['generated_tokens']} token，最终判分 {int(g['correct'])}",'','开头 300 字符','','~~~~text',txt[:300],'~~~~','','结尾 300 字符','','~~~~text',txt[-300:],'~~~~']
            (HERE/'SAMPLES.md').write_text('\n'.join(lines)+'\n')
    def table(values):return {key:summary(values[:,i]) for i,key in enumerate(METRICS)}
    return dict(hashes=hashes,profile=dict(profile),box_counts=dict(sorted(distribution.items())),
        models={a:table(x) for a,x in models.items()},
        paired={a:table(models[BOUNDARY]-models[a]) for a in MODELS[1:]})


def truncated(w):
    from evaluate_math import extract_boxed_answer
    data,digest=archive('B',32768,True)
    rows=[]; count=Counter(); count_diffs=[]
    for q in data:
        samples=[]
        for g in q['generations']:
            text=g['full_generation']; n=g['generated_tokens']
            if n>16384:
                count['cut']+=1
                encoded=org.tok.encode(text,add_special_tokens=False)
                count_diffs.append(len(encoded.ids)-n)
                text=org.tok.decode(encoded.ids[:16384],skip_special_tokens=False)
            initial=int(org.grade(extract_boxed_answer(text),q['ground_truth']))
            if n<=16384: assert initial==int(g['correct'])
            else:
                count['correct_after_cut']+=initial
                count['lost_correct']+=int(g['correct'] and not initial)
                count['gained_correct']+=int(initial and not g['correct'])
            samples.append([initial,int(g['correct'])])
        rows.append(np.mean(samples,axis=0))
    a=w@np.array(rows)*100
    return dict(sha256=digest,accuracy16k=summary(a[:,0]),accuracy32k=summary(a[:,1]),
                counts=dict(count),retokenized_minus_native_length=dict(Counter(count_diffs)))


def main():
    org.init()
    rng=np.random.default_rng(SEED)
    w=np.vstack([np.ones(500),rng.multinomial(500,np.full(500,1/500),10000)])/500
    report={str(b):compare(b,w) for b in (4096,16384)}
    report['thinking_B']=truncated(w)
    (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
