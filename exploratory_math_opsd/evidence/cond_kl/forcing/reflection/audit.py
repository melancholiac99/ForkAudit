"""Independent reduction from raw responses and probability arrays for the two reflection diagnostics."""
import hashlib
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from settings import *
sys.path.insert(0,str(ROOT/'source/eval'))
from evaluate_math import grade_answer, extract_boxed_answer
from functools import lru_cache

def run():
    labels=read(RAW/'labels.json')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    joined=[tok.encode(s,add_special_tokens=False) for w in ('Wait','But','Alternatively','Hmm') for s in (w,' '+w)]
    assert all(len(x)==1 for x in joined)
    ids=sorted(x[0] for x in joined)
    assert len(set(ids))==8 and ids==read(HERE/'tokens.json')['included_ids']
    wait=set(x[0] for x in joined[:2])
    archives=read(HERE.parents[2]/'internalization/inputs.json')
    adapters={}
    for arm in ('P','T0','R','R-noclip','R-cond'):
        ck=ROOT/'run'/arm/'checkpoint-100/adapter_model.safetensors'
        digest=hashlib.sha256(ck.read_bytes()).hexdigest()
        expected=(archives['adapters'][arm+'@100'] if arm in ('P','T0','R') else read(ck.parent.parent/'complete.json')['adapters']['100'])
        assert digest==expected
        adapters[arm]=digest
    N=[];sums=[];right=[];used=0;diffs=[];fork_q=[];nroll=0;sent=0;para=0
    correct=0;wrong=0;missing=0;digit_bang=0;sample=[]
    for lane in range(4):
        path=ROOT/'run/P-projector'/f'rollouts-{lane}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==archives['archives'][path.name]
        for idx,sp,t,y in torch.load(path,weights_only=True):
            meta=labels[str(idx)]
            tag=meta['correct']
            answer=extract_boxed_answer(tok.decode(y,skip_special_tokens=True))
            if meta['gold']:
                assert grade_answer(answer,meta['gold'])==tag
                correct+=int(tag);wrong+=int(not tag)
            else:missing+=1;assert tag is None
            bounds=meta['boundaries'];offset=[j for j,g in bounds]
            assert offset==sorted(set(offset)) and all(0<j<len(y) for j in offset)
            sent+=sum(g=='sentence' for _,g in bounds);para+=sum(g=='paragraph' for _,g in bounds)
            for j,g in bounds:
                if g=='sentence':
                    suffix=tok.decode(y[max(0,j-5):j]).rstrip()
                    digit_bang+=int(len(suffix)>1 and suffix[-1]=='!' and suffix[-2].isdigit())
            with np.load(RAW/f'score-{idx}.npz') as f:
                p=f['p']; first=f['first'];sites=f['sites'].tolist()
            assert p.shape==(len(offset),6,8) and np.isfinite(p).all() and (p>=0).all() and (p.sum(-1)<=1+1e-12).all()
            if tag is not None:
                N.append(len(offset));sums.append(p.sum(axis=(0,2)));right.append(tag)
            for k,site in enumerate(sites):
                assert [site,'sentence'] in bounds
                rows=read(RAW/f'fork-{idx}-{site}.json')
                assert len(rows)==8 and len({(r['arm'],r['rep']) for r in rows})==8
                vals=[[],[]]
                for r in rows:
                    branch,rep=r['arm'],r['rep'];generated=r['token_ids']
                    assert r['first']==first[k,branch,rep]==generated[0]
                    assert (generated[0] in wait)==(branch==0)
                    assert r['remaining']==len(generated)<=1024-site
                    # Re-score a deterministic spread of archived continuations from their complete token strings.
                    if (idx+site+rep)%31==0:
                        pred=extract_boxed_answer(tok.decode(y[:site]+generated,skip_special_tokens=True))
                        assert pred==r['predicted'] and grade_answer(pred,meta['gold'])==r['correct']
                        used+=1
                    vals[branch].append([r['remaining'],r['correct']])
                diffs.append(np.mean(vals[0],axis=0)-np.mean(vals[1],axis=0))
                fork_q.append(idx);nroll+=8
    right=np.array(right);N=np.array(N);sums=np.array(sums)
    avg=[sums[right==t].sum(0)/N[right==t].sum() for t in (False,True)]
    lr=np.log(avg[0])-np.log(avg[1]);d=lr-lr[0]
    r=read(HERE/'targeting.json')['position/all/both']['models']
    max_err=max(abs(d[i]-r[m]['combined']['log_ratio_minus_B']['mean']) for i,m in enumerate(MODELS))
    assert max_err<1e-10
    fx=np.mean(diffs,axis=0)
    ref=read(HERE/'forks.json')['metrics']['Wait_minus_nonWait/equal']
    assert abs(fx[0]-ref['remaining_length']['mean'])<1e-10 and abs(fx[1]-ref['accuracy']['mean'])<1e-10
    save(HERE/'audit.json',dict(correct=correct,wrong=wrong,missing=missing,sentence=sent,paragraph=para,eligible_positions=int(N.sum()),digit_followed_by_bang_sentence_boundaries=digit_bang,adapters=adapters,eight_token_ids=ids,independent_targeting_max_abs_error=max_err,independent_fork_deltas=fx.tolist(),fork_questions=len(set(fork_q)),sites=len(diffs),continuations=nroll,continuations_regraded=used))
if __name__=='__main__':run()
