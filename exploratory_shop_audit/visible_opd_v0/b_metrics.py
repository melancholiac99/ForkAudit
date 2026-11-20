"""Clustered held-out visibility, power and calibration statistics for stage B."""
from collections import Counter
import json,itertools
import numpy as np
from scipy.special import logsumexp
from .b_collect import RUN
from .b_fit import load_data,SEED,CLASSES
from exploratory_shop_audit.grpo_forced_audit.common import save

def summary(x):
    x=np.asarray(x);return dict(n=len(x),mean=float(x.mean()),quantiles=np.quantile(x,[.1,.25,.5,.75,.9]).tolist()) if len(x) else dict(n=0)
class Cluster:
    def __init__(self,groups):
        gs,self.inv,self.count=np.unique(groups,return_inverse=True,return_counts=True)
        self.draw=np.random.default_rng(SEED).integers(len(gs),size=(2000,len(gs)))
        self.n=len(gs)
    def agg(self,x):
        return np.bincount(self.inv,weights=x,minlength=self.n)/self.count
    def estimate(self,v):
        v=self.agg(v);return dict(point=v.mean().item(),ci95=np.quantile(v[self.draw].mean(1),[.025,.975]).tolist())
    def ratio(self,xx,xy,x,y):
        a=np.stack([self.agg(u) for u in [xx,xy,x,y]],axis=1)
        def calc(p):
            var=p[...,0]-p[...,2]**2
            return np.divide(p[...,1]-p[...,2]*p[...,3],var,out=np.full_like(var,np.nan),where=var>1e-10)
        raw=calc(a.mean(0));boot=calc(a[self.draw].mean(1));boot=boot[np.isfinite(boot)]
        return dict(raw=float(raw) if np.isfinite(raw) else None,ci95=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,bootstrap_valid=len(boot))

def conditional(s,delta,mask):
    z=np.where(mask,s+delta,-np.inf);lp=z-logsumexp(z,axis=1,keepdims=True)
    return np.exp(lp),np.where(mask,lp,0.)

def stats_label(label,data,meta,preds,subset):
    use=subset
    y=data['d' if label=='content' else 'old'][use]
    mask=data['mask'][use];n=mask.sum(1)
    avg=lambda z:(z*mask).sum(1)/n
    g=Cluster(meta['goal'][use]);p=preds[label][use]
    mse=avg((y-p)**2);base={};best=None
    for name,v in [('zero',0.),('rank_mean',preds[label+'_rank_mean'][use]),('token_only',preds[label+'_token_only'][use])]:
        baseline=avg((y-v)**2);effect=g.estimate(baseline-mse);b=g.agg(baseline).mean()
        a=g.agg(baseline-mse);bb=g.agg(baseline)
        rr=a[g.draw].mean(1)/np.maximum(bb[g.draw].mean(1),1e-15)
        base[name]=dict(mse=float(b),gain=effect,relative_gain=float(effect['point']/max(b,1e-15)),relative_ci95=np.quantile(rr,[.025,.975]).tolist())
        if best is None or b<base[best]['mse']:best=name
    st=[preds[label+'_stability'+str(k)][use] for k in range(3)]
    both=[];agree=[]
    for a,b in itertools.combinations(st,2):
        ok=mask&(abs(a)>.02)&(abs(b)>.02)
        both.append(ok.sum());agree.append(((a*b>0)&ok).sum())
    stability=sum(agree)/sum(both) if sum(both) else None
    slope=g.ratio(avg(p*p),avg(p*y),avg(p),avg(y))
    slope['b']=float(np.clip(slope['raw'],0,1)) if slope['raw'] is not None else 0.
    slope['clipped_ci95']=np.clip(slope['ci95'],0,1).tolist() if slope['ci95'] else None
    return dict(tasks=g.n,positions=len(y),mse=float(g.agg(mse).mean()),baselines=base,strongest=best,
                sign_agreement=float(g.agg(avg(np.sign(y)==np.sign(p))).mean()),stable_same_sign=stability,
                stability_evaluable_fraction=float(sum(both)/(3*n.sum())),calibration=slope)

def power(data,meta,preds,use):
    all_n=int(use.sum());use=use&preds['power_valid']
    if not use.any():return dict(status='UNDERPOWERED',positions=0,unmatched=all_n)
    g=Cluster(meta['goal'][use]);m=data['mask'][use];n=m.sum(1)
    avg=lambda z:(z*m).sum(1)/n
    truth=preds['planted'][use];delta=preds['synthetic'][use]-preds['null'][use]
    recovery=g.ratio(avg(truth**2),avg(truth*delta),avg(truth),avg(delta))
    syn=preds['shuffled'][use]+truth
    gains=g.estimate(avg((syn-preds['null'][use])**2-(syn-preds['synthetic'][use])**2))
    ok=recovery['raw'] is not None and recovery['raw']>=.5 and recovery['ci95'] and recovery['ci95'][0]>0
    sig=truth[m];noise=preds['shuffled'][use][m]
    return dict(status='POWER_OK' if ok else 'UNDERPOWERED',tasks=g.n,positions=len(n),unmatched=all_n-len(n),recovery=recovery,
                oof_mse_gain=gains,independent_component_variance_fraction=float(sig.var()/max(1e-15,sig.var()+noise.var())))

def strength(s,delta,mask):
    qs,ls=conditional(s,0,mask);qt,lt=conditional(s,delta,mask)
    return (qt*(lt-ls)).sum(1),abs(qt-qs).sum(1)

def match_scale(s,mask,d,reference,stop):
    quant=[.25,.5,.75,.9];target=np.quantile(reference,quant)
    best=None
    for k in range(1+round(stop*100)):
        scale=k/100
        kl,_=strength(s,scale*d,mask)
        v=np.quantile(kl,quant);err=abs(v-target)/np.maximum(target,1e-5)
        if best is None or err.mean()<best[0]:best=(err.mean(),scale,v,err)
    _,proxy=strength(s,best[1]*d,mask)
    return dict(scale=best[1],target_quantiles=target.tolist(),achieved_quantiles=best[2].tolist(),relative_errors=best[3].tolist(),status='STRENGTH_MISMATCH' if best[3].max()>.25 else 'MATCHED',gradient_l1=summary(proxy))

def p1_status(detected, powered):
    if detected:
        return 'PASS' if powered else 'PASS_LOW_POWER'
    return 'FAIL_POWERED' if powered else 'UNDERPOWERED'


def analyze():
    ts,meta,_=load_data();data={k:v.numpy().astype(np.float64) if k not in ('ids','mask') else v.numpy() for k,v in ts.items() if k!='h'}
    merged={};seen=[]
    for f in range(4):
        obj=np.load(RUN/f'fold_{f}.npz');idx=obj['idx'];assert (meta['fold'][idx]==f).all()
        seen.extend(idx.tolist())
        for k in obj.files:
            if k=='idx':continue
            a=obj[k]
            if k not in merged:merged[k]=np.zeros((len(meta['goal']),*a.shape[1:]),dtype=a.dtype)
            merged[k][idx]=a
    assert sorted(seen)==list(range(len(meta['goal'])))
    classes=meta['cls'];pages=meta['page']
    groups={'all':np.ones(len(classes),bool),**{c:classes==c for c in set(classes)}}
    groups.update({p+'/'+c:(pages==p)&(classes==c) for p,c in sorted(set(zip(pages,classes)))})
    projection={};old={};pw={};masses={};scales={};old_kl={}
    for name,use in groups.items():
        projection[name]=stats_label('content',data,meta,merged,use);old[name]=stats_label('original',data,meta,merged,use)
        pw[name]=power(data,meta,merged,use)
        v=projection[name];pos=v['baselines'][v['strongest']]['gain']['ci95'][0]>0
        v['P1']=('UNDERPOWERED' if pw[name]['status']!='POWER_OK' else ('PASS' if pos and (v['stable_same_sign'] or 0)>=.7 else 'FAIL_POWERED'))
        masses[name]={k:summary(data['mass_'+k][use]) for k in ('none','full','placeholder')}
        masses[name]['fraction_any_below_point25']=float((np.minimum.reduce([data['mass_'+k][use] for k in ('none','full','placeholder')])<.25).mean())
    for c in sorted(set(classes)):
        ii=classes==c;s=data['s'][ii];mask=data['mask'][ii];d=data['d'][ii];b=projection[c]['calibration']['b']
        p,grad=strength(s,b*merged['content'][ii],mask);raw,_=strength(s,d,mask)
        orig,_=strength(s,data['old'][ii],mask);op,_=strength(s,old[c]['calibration']['b']*merged['original'][ii],mask)
        old_kl[c]=dict(raw=summary(orig),projected=summary(op))
        scales[c]=dict(b=b,b_ci95=projection[c]['calibration']['clipped_ci95'],raw_kl=summary(raw),P_kl=summary(p),P_gradient_l1=summary(grad),
                       zero_change_fraction=float((np.where(mask,abs(b*merged['content'][ii]),0).max(1)<=1e-5).mean()),
                       beta=match_scale(s,mask,d,p,1),gamma=match_scale(s,mask,merged['null'][ii],p,4))
    directory=RUN/'reports';directory.mkdir(exist_ok=True)
    for filename,content in [('projection_report',dict(label='content',groups=projection)),('original_label_control',dict(label='sel versus student; report only',groups=old,class_KL=old_kl)),('power_injection_report',dict(groups=pw)),('candidate_mass_report',dict(groups=masses)),('calibration_strength_report',dict(classes=scales,quantiles=[.25,.50,.75,.90]))]:
        save(directory/(filename+'.json'),content)
    import pandas as pd
    df=pd.DataFrame(meta)
    for label in ('content','original','null','planted','synthetic'):
        df[label]=list(merged[label])
    df['candidate_ids']=list(data['ids']);df['candidate_mask']=list(data['mask']);df.to_parquet(RUN/'projection_oof.parquet')
    passed=[k for k in ('search','click','option') if k in projection and projection[k]['P1']=='PASS' and (projection[k]['calibration']['ci95'] or [0])[0]>0]
    save(directory/'P1_decision.json',dict(eligible_classes=passed,stage='B',actor_updates=0,positive_projected_pilot_supported=bool(passed),clusters='512 distinct tasks; one per ASIN; task and ASIN folds/intervals coincide'))
    save(directory/'p2_minima.json',__import__(__package__+'.section16',fromlist=['freeze_p2_minima']).freeze_p2_minima(Counter(classes),512))
if __name__=='__main__':analyze()
