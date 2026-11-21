"""Nested ASIN-fold forward-KL comparison on signed T_cal inputs only; CPU."""
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,xlogy
import torch
from .b_fit import load_data, weights
from .c_targets import STAGE_B
from .e25_probe import RUN,PUBLIC,SEED
from .e25_projector import FitData
from exploratory_shop_audit.grpo_forced_audit.common import read,save,digest_file

def divergence(s,q,mask,pred,scale):
    z=np.where(mask,s+scale*pred,-np.inf)
    logp=np.where(mask,z-logsumexp(z,axis=-1,keepdims=True),0.)
    return (xlogy(q,q)-q*logp).sum(-1)

def shrink(indices,s,q,mask,pred,meta,w):
    out={}
    for c in sorted(set(meta['cls'])):
        i=indices[meta['cls'][indices]==c]
        if not len(i):out[c]=0.;continue
        def loss(x):return np.average(divergence(s[i],q[i],mask[i],pred[i],x),weights=w[i])
        opt=minimize_scalar(loss,bounds=(0.,1.),method='bounded',options={'xatol':1e-8})
        out[c]=float(min((0.,opt.x,1.),key=loss))
    return out

def main():
    os.umask(0o077);dst=RUN/'offline';dst.mkdir(parents=True,exist_ok=True)
    fmt=read(STAGE_B/'section19/format_split_report.json')
    save(PUBLIC/'format_predictable_share.json',dict(definition='当前预测器（rank-8、512 题、Huber）的题外可预测份额',
        reference='relative OOF MSE improvement over strongest baseline',source_sha=digest_file(STAGE_B/'section19/format_split_report.json'),
        classes={c:dict(positions=v['positions'],tasks=v['tasks'],baseline=v['strongest'],**v['baselines'][v['strongest']])
            for c,v in fmt['groups'].items()},frozen_b=fmt['frozen_format_b']))
    ts,meta,_=load_data();h=FitData(ts,meta,torch.load(STAGE_B/'head.pt',weights_only=True,mmap=True),'cpu')
    s=h.s.double().numpy();q=h.target.double().numpy();q=q/q.sum(-1,keepdims=True);mask=h.mask.numpy()
    n=len(s);all_idx=np.arange(n);w=weights(meta,all_idx).astype(np.float64)
    fold=meta['fold'];cls=meta['cls'];shrinks={};oof=np.zeros_like(s);loss0=np.zeros(n);loss1=np.zeros(n)
    baseline=read(STAGE_B/'reports/calibration_strength_report.json')['classes']
    for f in range(4):
        path=dst/f'fold_{f}.npz';model_path=dst/f'kl_{f}.pt'
        tr=all_idx[fold!=f];va=all_idx[fold==f]
        def train_predict(learn,test):
            net=h.fit(learn,f,'kl')
            with torch.no_grad():
                out=np.concatenate([h.predict(net,b).numpy() for b in np.array_split(test,max(1,(len(test)+511)//512))])
            return net,out
        if not path.exists():
            inner=np.zeros_like(s)
            for j in range(4):
                if f!=j:
                    ix=tr[fold[tr]==j];_,inner[ix]=train_predict(tr[fold[tr]!=j],ix)
            cs=shrink(tr,s,q,mask,inner,meta,w);net,pred=train_predict(tr,va)
            torch.save(net.state_dict(),model_path);save(dst/f'shrink_{f}.json',cs)
            np.savez_compressed(path,va=va,pred=pred)
        v=np.load(path);assert np.array_equal(v['va'],va);oof[va]=v['pred'];cs=read(dst/f'shrink_{f}.json');shrinks[f]=cs
        base=np.load(STAGE_B/f'fold_{f}.npz');assert np.array_equal(base['idx'],va)
        sb=np.array([baseline[c]['b'] for c in cls[va]])[:,None];sk=np.array([cs[c] for c in cls[va]])[:,None]
        loss0[va]=divergence(s[va],q[va],mask[va],base['content'],sb)
        loss1[va]=divergence(s[va],q[va],mask[va],oof[va],sk)
        print('completed outer fold',f,flush=True)
    def stats(ix):
        gs,inverse=np.unique(meta['goal'][ix],return_inverse=True)
        denom=np.bincount(inverse,weights=w[ix])
        sums=np.bincount(inverse,weights=w[ix]*(loss0[ix]-loss1[ix]))
        draws=np.random.default_rng(SEED).multinomial(len(gs),np.full(len(gs),1/len(gs)),size=10000)
        ci=np.quantile((draws@sums)/(draws@denom),[.025,.975])
        return dict(tasks=len(gs),positions=len(ix),huber=np.average(loss0[ix],weights=w[ix]),distribution=np.average(loss1[ix],weights=w[ix]),
                    gain=sums.sum()/denom.sum(),ci95=ci.tolist())
    overall=stats(all_idx);prod=shrink(all_idx,s,q,mask,oof,meta,w)
    record=dict(estimand='task/category/position weighted held-out forward KL',outer_training_only_shrink=shrinks,
        deployment_oof_shrink=prod,global_result=overall,by_class={c:stats(all_idx[cls==c]) for c in set(cls)},
        p_kl_pilot_after_E=bool(overall['ci95'][0]>0),seed=SEED,
        weights_sha=[digest_file(dst/f'kl_{f}.pt') for f in range(4)])
    save(PUBLIC/'distribution_projector.json',record);save(dst/'complete.json',record)
if __name__=='__main__':main()
