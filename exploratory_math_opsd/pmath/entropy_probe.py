"""Stratify archived held-out logit predictions by full-vocabulary base entropy (CPU)."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from common import MODEL, ROOT, WORK, json_save
from fit import split

OUT=ROOT/'run/entropy-probe-v1'
REPORT=Path(__file__).resolve().parents[1]/'evidence/entropy_probe'
WORDS=('Wait','But','Alternatively','Hmm','Actually',r'\boxed')
TEMP=1.1


def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def controls():
    tok=AutoTokenizer.from_pretrained(MODEL)
    eos=json.loads((MODEL/'generation_config.json').read_text())['eos_token_id']
    vocab=tok.get_vocab()
    forms={}
    for word in WORDS:
        for form in (word,' '+word):
            idx=vocab.get(form.replace(' ','Ġ'))
            if idx is not None:
                assert tok.decode([idx],clean_up_tokenization_spaces=False)==form
            forms[form]=dict(id=idx,encoded=tok.encode(form,add_special_tokens=False))
    return forms,eos


@torch.inference_mode()
def score(lane):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(exist_ok=True,parents=True)
    folds=split()
    scale=load(WORK/'projector.pt')['scale']
    head=load(WORK/'head.pt').float()
    forms,eos=controls()
    control=torch.zeros(len(head),dtype=torch.bool)
    control[eos+[v['id'] for v in forms.values() if v['id'] is not None]]=True
    factors=[]
    for fold in range(4):
        p=load(WORK/f'fit-{fold}.pt')
        factors.append((p['A'],F.linear(head,p['B'])))
    for idx in sorted(folds)[lane::4]:
        dest=OUT/f'q-{idx}.npy'
        if dest.exists():continue
        x=load(WORK/f'features-{idx}.pt')
        ht=load(ROOT/f'run/internalization-v1/teacher/{idx}.pt')
        A,V=factors[folds[idx]]
        chunks=[]
        for start in range(0,len(ht),128):
            sl=slice(start,start+128)
            h=x['h'][sl].float()
            s=F.linear(h,head).bfloat16().float()
            t=F.linear(ht[sl].float(),head).bfloat16().float()
            lp=F.log_softmax(s/TEMP,dim=1)
            lt=F.log_softmax(t/TEMP,dim=1)
            entropy=-(lp.exp()*lp).sum(1,dtype=torch.float64)
            prob=lt.exp()
            gap=(prob*(lt-lp)).sum(1,dtype=torch.float64)
            top=s.argmax(1)
            ids=x['ids'][sl].long()
            valid=x['valid'][sl]
            left=F.linear(h,A)
            predicted=scale*torch.einsum('pr,pcr->pc',left,V[ids])
            d=x['d'][sl]
            w=valid.double()/valid.sum(1,keepdim=True)
            label_mean=(w*d).sum(1)
            label_second=(w*d.double().square()).sum(1)
            mse=(w*(d-predicted).double().square()).sum(1)
            # Rres: subtract on deduplicated teacher/student top-32 union.
            res=t.scatter_add(1,ids,-predicted.masked_fill(~valid,0))
            lr=F.log_softmax(res/TEMP,dim=1)
            rres=(lr.exp()*(lr-lp)).sum(1,dtype=torch.float64)
            # U: same teacher, compared to the P target applied on student top-64.
            top64=s.topk(64).indices
            adj=scale*torch.einsum('pr,pcr->pc',left,V[top64])
            projected=s.scatter_add(1,top64,adj)
            u=(prob*(lt-F.log_softmax(projected/TEMP,dim=1))).sum(1,dtype=torch.float64)
            chunks.append(torch.stack([entropy,gap,u,rres,label_mean,label_second,mse,control[top]],dim=1))
        values=torch.cat(chunks).numpy()
        assert np.isfinite(values).all() and len(values)==len(ht)
        tmp=dest.with_suffix('.pending.npy')
        np.save(tmp,values)
        tmp.replace(dest)


def ratios(t):
    n,g,u,r,first,second,err,ct=np.moveaxis(t,-1,0)
    var=second/n-(first/n)**2
    return np.stack([n/n[...,-1:],g/g[...,-1:],1-(err/n)/var,u/u[...,-1:],r/r[...,-1:],ct/n],axis=-1)


def summarize():
    ids=sorted(split())
    data=[np.load(OUT/f'q-{i}.npy') for i in ids]
    ent=np.concatenate([a[:,0] for a in data])
    assert len(ent)==1185192
    cuts=np.quantile(ent,np.arange(1,10)/10)
    parts=[]
    for a in data:
        bins=np.searchsorted(cuts,a[:,0],side='right')
        sums=np.stack([np.bincount(bins,weights=v,minlength=10) for v in [np.ones(len(a)),*a[:,1:].T]],axis=1)
        parts.append(np.vstack([sums,sums[-2:].sum(0),sums.sum(0)]))
    parts=np.stack(parts)
    ans=ratios(parts.sum(0))
    rng=np.random.default_rng(20261124)
    draws=[]
    for _ in range(100):
        rows=rng.integers(1500,size=(100,1500))
        w=np.stack([np.bincount(ix,minlength=1500) for ix in rows])
        draws.append(ratios((w@parts.reshape(1500,-1)).reshape(100,12,8)))
    lo,hi=np.percentile(np.concatenate(draws),[2.5,97.5],axis=0)
    fields=['position_share','R_KL_share','OOF_R2','U_KL_share','Rres_KL_share','control_top1_rate']
    names=[f'D{i}' for i in range(1,11)]+['top20','all']
    totals=parts[:,-1].sum(0)
    report=dict(questions=1500,positions=len(ent),temperature=TEMP,entropy_edges_nats=cuts.tolist(),
        scale=load(WORK/'projector.pt')['scale'],token_forms=controls()[0],eos_ids=controls()[1],
        net_kl_means=dict(G=totals[1]/totals[0],U=totals[2]/totals[0],Rres=totals[3]/totals[0]),
        inference='95% percentile CI, whole questions, fixed full-probe cutpoints, 10000 draws, seed 20261124',
        tied_top1='lowest vocabulary ID', pred_weighting='equal position and uniform valid candidates within each position; calibrated held-out prediction; weighted within-bin variance around bin mean',
        bins={n:{f:dict(value=v,ci95=[l,h]) for f,v,l,h in zip(fields,ans[j],lo[j],hi[j])} for j,n in enumerate(names)})
    assert np.allclose(ans[:10,[0,1,3,4]].sum(0),1)
    assert np.allclose(ans[10,[0,1,3,4]],ans[8:10,[0,1,3,4]].sum(0))
    REPORT.mkdir(exist_ok=True,parents=True)
    json_save(REPORT/'results.json',report)
    md=['# 学生熵分层（固定拟合探针，题外投影器）','',
        '1500 题，1,185,192 个回答位置。全词表单位为 nats，温度 1.1。按整题聚类重抽 10,000 次，种子 20261124，使用全探针固定切点。位置等权，各位置的有效候选等权。R² 使用校准后的题外预测和各组内的加权标签方差。各 KL 均为净前向 KL。U 表示老师相对投影目标；Rres 表示并集内扣除修正后的残差老师相对学生。控制词精确匹配单个 token 及其带一个前导空格的形式，包含生成配置的两个 EOS ID。多 token 拼写不代表一个 top-1 候选。并列最大取最小词表 ID。','',
        '下表单位为 %；每格给出点估计与按题聚类的 95% 区间。','',
        '|熵组|位置占比|R KL 占比|解释比例|U KL 占比|Rres KL 占比|控制词 top-1 比例|','|---|---:|---:|---:|---:|---:|---:|']
    for j,n in enumerate(names):
        md.append('| '+n+' | '+' | '.join(f'{100*v:.2f} [{100*l:.2f}, {100*h:.2f}]' for v,l,h in zip(ans[j],lo[j],hi[j]))+' |')
    (REPORT/'README.md').write_text('\n'.join(md)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('lane',choices=['0','1','2','3','report'])
    args=parser.parse_args()
    if args.lane=='report':summarize()
    else:score(int(args.lane))
