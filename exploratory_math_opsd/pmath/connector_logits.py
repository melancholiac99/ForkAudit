"""Signed held-out logit components conditional on a literal token's presence in C."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from common import MODEL, WORK, json_save
from fit import split

WORDS=('Wait','But','Hmm','Alternatively','Actually','Let me check','verify','double-check')
COMPONENTS=('P','residual','R')
OUT=Path(__file__).resolve().parents[1]/'evidence/pmath/connector_logits'

def main():
    tok=AutoTokenizer.from_pretrained(MODEL)
    vocab=tok.get_vocab()
    forms={};groups={};ids=[]
    for word in WORDS:
        groups[word]=[]
        for text in (word,' '+word):
            tid=vocab.get(text.replace(' ','Ġ'))
            pieces=tok.encode(text,add_special_tokens=False)
            if tid is not None:
                assert tok.decode([tid],clean_up_tokenization_spaces=False)==text
                groups[word].append(len(ids));ids.append(tid)
            forms[text]=dict(id=tid,tokenization=pieces,pieces=[tok.decode([i]) for i in pieces])
    assert len(set(ids))==len(ids)
    folds=split()
    scale=float(torch.load(WORK/'projector.pt',weights_only=True)['scale'])
    head=torch.load(WORK/'head.pt',weights_only=True)[ids].double()
    states=[]
    for k in range(4):
        st=torch.load(WORK/f'fit-{k}.pt',weights_only=True)
        states.append((st['A'].double(),head@st['B'].double().T))
    lookup=torch.full((len(tok)+1024,),-1,dtype=torch.long)
    lookup[ids]=torch.arange(len(ids))
    totals=np.zeros((1500,len(ids),7))
    positions=0
    for qi,(idx,k) in enumerate(sorted(folds.items())):
        x=torch.load(WORK/f'features-{idx}.pt',weights_only=True)
        matches=lookup[x['ids'].long()]
        pos,col=((matches>=0)&x['valid']).nonzero(as_tuple=True)
        positions+=len(x['h'])
        if not pos.numel():continue
        fi=matches[pos,col]
        A,vec=states[k]
        p=scale*((x['h'][pos].double()@A.T)*vec[fi]).sum(-1)
        r=x['d'][pos,col].double()
        comps=torch.stack((p,r-p,r),1).numpy()
        np.add.at(totals[qi],fi.numpy(),np.column_stack((np.ones(len(pos)),comps,comps<0)))
    assert positions==1185192
    sets={w:indices for w,indices in groups.items()}
    for i,tid in enumerate(ids):sets['id:'+str(tid)]=[i]
    names=list(sets)
    data=np.stack([totals[:,ix,:].sum(1) for ix in sets.values()],axis=1)
    flat=data.reshape(1500,-1);rng=np.random.default_rng(20261124);stats=[]
    for _ in range(100):
        draws=rng.integers(1500,size=(100,1500))
        weights=np.stack([np.bincount(d,minlength=1500) for d in draws])
        batch=(weights@flat).reshape(100,len(names),7)
        stats.append(np.divide(batch[:,:,1:],batch[:,:,:1],out=np.full_like(batch[:,:,1:],np.nan),where=batch[:,:,:1]>0))
    sample=np.concatenate(stats)
    res={}
    for j,name in enumerate(names):
        vals=data[:,j].sum(0);n=int(vals[0])
        rec=dict(matches=n,questions=int((data[:,j,0]>0).sum()))
        if n:
            valid=np.isfinite(sample[:,j,0]);ci=np.percentile(sample[valid,j,:],[2.5,97.5],axis=0)
            rec['bootstrap_nonempty']=int(valid.sum())
            for a,c in enumerate(COMPONENTS):
                rec[c]=dict(mean_logit=vals[1+a]/n,mean_ci95=ci[:,a].tolist(),
                            negative_fraction=vals[4+a]/n,negative_ci95=ci[:,3+a].tolist())
        res[name]=rec
    OUT.mkdir(exist_ok=True,parents=True)
    json_save(OUT/'results.json',dict(scale=scale,positions=positions,questions=1500,forms=forms,
        resampling=dict(unit='whole question',draws=10000,seed=20261124),results=res))
    md=['# 候选集内的折外 logit 修正','','1,500 题，1,185,192 个回答位置。按词表精确拼写（区分大小写）查找无前导空格和一个前导空格的形式。候选并集去重；每个匹配的“位置、token ID”计一次。P 为 λf，残差为 R−P，R 为归档的、按候选集内学生概率中心化的老师−学生 logit 差。用对应留出折的因子和训练时的 λ。负号指严格小于零。','',
        '整题有放回重抽 10,000 次（种子 20261124），每次以抽中题目的匹配数为分母，报告百分位区间；某次重抽没有匹配时该次无定义。','']
    def table(entries):
        md.extend(['|词形|位置–ID 数 / 题数|分量|平均 logit [95% 区间]|负值比例 [95% 区间]|','|---|---:|---|---:|---:|'])
        for label,key in entries:
            v=res[key]
            if not v['matches']:
                md.append(f'|{label}|0 / 0|—|NA|NA|');continue
            for c in COMPONENTS:
                x=v[c];lo,hi=x['mean_ci95'];a,b=x['negative_ci95']
                md.append(f'|{label}|{v["matches"]:,} / {v["questions"]:,}|{c}|{x["mean_logit"]:+.4f} [{lo:+.4f}, {hi:+.4f}]|{x["negative_fraction"]:.2%} [{a:.2%}, {b:.2%}]|')
    table([(w,w) for w in WORDS])
    md+=['','## 分 token ID','','引号内保留前导空格。','']
    table([(f'`{json.dumps(t)}` ({v["id"]})','id:'+str(v['id'])) for t,v in forms.items() if v['id'] is not None])
    md+=['','## 无单一词表 ID 的形式','']
    for t,v in forms.items():
        if v['id'] is None:md.append(f'- `{json.dumps(t)}` → `{json.dumps(v["pieces"])}`')
    (OUT/'README.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':main()
