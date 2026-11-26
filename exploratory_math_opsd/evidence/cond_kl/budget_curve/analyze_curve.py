"""Post-hoc CPU analysis using receipt-verified generated answers."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'boundary_answers'))
from audit import archive, org

BOUNDARY = 'R-clip-boundary@100'
ARMS = ('B','T0@100','R@100','R-noclip@100','R-cond@100',BOUNDARY,'R-clip-interior@100')
LAMBDA_ZERO = 'R-lambda0@100'
CONTROL = ('B','B thinking','R@100')
CURVE = ('B','B thinking','R@100','R-noclip@100','T0@100',BOUNDARY,LAMBDA_ZERO)
BUDGETS = (2048,4096,6144,8192,12288,16384)


def describe(rows):
    arrays=np.array(rows)
    result={}
    for label, mask in [('present',arrays[:,0]==1),('absent',arrays[:,0]==0)]:
        n=int(mask.sum())
        result[label]=dict(count=n,accuracy_pct=float(100*arrays[mask,1].mean()) if n else None,
                           tokens=float(arrays[mask,2].mean()) if n else None)
    result['prevalence_pct']=float(100*arrays[:,0].mean())
    return result


def run():
    org.init()
    from evaluate_math import extract_boxed_answer
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(org.ROOT/'models/Qwen3-1.7B/tokenizer.json'))
    w=np.random.default_rng(20261124).multinomial(500,np.full(500,1/500),10000)/500
    def paired(x):
        return [round(float(100*x.mean()),6),*np.quantile(100*(w@x),[.025,.975]).round(6).tolist()]
    result=dict(marker={},sha256={},curve={},paired={},validation={},counts={})
    estimates={}; actual={}; order=None
    for budget in (4096,16384,32768):
        for arm in {4096: ARMS, 16384: (*ARMS, LAMBDA_ZERO), 32768: ('B thinking',)}[budget]:
            if budget==4096 and arm=='R-cond@100' and not (org.ROOT/'run/math500-trajectory-k20/R-cond@100/receipt.json').exists():
                result['marker'][f'{arm}/4096']=None
                continue
            thinking=arm=='B thinking'
            data,digest=archive('B' if thinking else arm,budget,thinking)
            ids=[r['problem'] for r in data]
            assert len(set(ids))==500 and (order is None or order==ids)
            order=ids
            key=f'{arm}/{budget}'
            result['sha256'][key]=digest
            strata=[];proxy=[];per_question=[]; final=[];cut_counts=np.zeros(6,dtype=int)
            for question in data:
                scores=[]; originals=[]
                for g in question['generations']:
                    text=g['full_generation']; n=g['generated_tokens']; original=int(g['correct'])
                    originals.append(original)
                    strata.append(('</think>' in text,original,n))
                    if budget==4096:continue
                    needs_curve=arm in CURVE
                    budgets=BUDGETS if needs_curve else ((4096,) if arm=='R-cond@100' else ())
                    tokens=None; row=[]
                    for i,b in enumerate(budgets):
                        prefix=text
                        if n>b:
                            if tokens is None: tokens=tok.encode(text,add_special_tokens=False).ids
                            prefix=tok.decode(tokens[:b],skip_special_tokens=False)
                            cut_counts[i]+=1
                        answer=extract_boxed_answer(prefix)
                        score=int(org.grade(answer,question['ground_truth']))
                        if n<=b:assert score==original
                        row.append(score)
                        if arm=='R-cond@100':
                            proxy.append(('</think>' in prefix,score,min(n,b)))
                    if needs_curve:scores.append(row)
                if scores:per_question.append(np.mean(scores,axis=0))
                final.append(np.mean(originals))
            if not thinking: result['marker'][key]=describe(strata)
            if proxy:result['R-cond_4k_reconstructed_from_16k']=describe(proxy)
            if per_question:
                estimates[arm]=np.array(per_question)
                result['curve'][arm]=(100*estimates[arm].mean(0)).round(6).tolist()
                result['counts'][arm]=cut_counts.tolist()
            if budget==4096:actual[arm]=np.array(final)
    result['lambda0_paired']={}
    for j,b in enumerate(BUDGETS):
        result['lambda0_paired'][b]={a:paired(estimates[LAMBDA_ZERO][:,j]-estimates[a][:,j])
                                   for a in ('R@100',BOUNDARY,'B thinking')}
        result['paired'][b]={a:paired(estimates[BOUNDARY][:,j]-estimates[a][:,j]) for a in CONTROL}
    for a in ('R@100','R-noclip@100',BOUNDARY):
        v=estimates[a][:,1]
        result['validation'][a]=dict(actual_4k_pct=round(100*actual[a].mean(),6),reconstructed_4k_pct=round(100*v.mean(),6),
                                     reconstructed_minus_actual_pp=paired(v-actual[a]))
    result['budgets']=BUDGETS
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
