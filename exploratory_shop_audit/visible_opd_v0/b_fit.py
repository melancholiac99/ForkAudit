"""Four ASIN-held-out rank-eight fits. Labels and hidden rows have independent roles."""
import argparse,os,time
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .b_collect import RUN,PARENT
from .projection import VisibleProjection
from exploratory_shop_audit.grpo_forced_audit.common import read,save,digest_value

SEED=20261120
CLASSES=['search','click','option','format','thought','unknown']
def load_data():
    split=read(PARENT/'asin_split_manifest.json')
    fold={r['goal_index']:split['cal_folds'][r['asin']] for r in split['T_cal']}
    ts=defaultdict(list);meta=defaultdict(list);actions=[]
    for file in sorted((RUN/'features').glob('*.pt')):
        v=torch.load(file,weights_only=True);n=len(v['classes'])
        for k in ('h','ids','mask','s','d','old','q','mass_none','mass_full','mass_placeholder'):ts[k].append(v[k])
        for k,val in [('fold',fold[v['goal']]),('goal',v['goal']),('step',v['step']),('page',v['page'])]:meta[k].extend([val]*n)
        meta['cls'].extend(v['classes'])
        if v['actions']:
            actions.append({k:v[k] for k in ('goal','step','page','actions','action_h','action_none','action_full','action_placeholder')})
    meta={k:np.array(v) for k,v in meta.items()}
    assert len(set(meta['goal']))==512 and len(fold)==512
    return {k:torch.cat(v) for k,v in ts.items()},meta,actions

def weights(meta,idx):
    # Equal tasks; equal observed categories per task; equal positions per category.
    g=meta['goal'][idx];cl=meta['cls'][idx]
    from collections import Counter
    freq=Counter(zip(g,cl));nc=Counter(k[0] for k in freq)
    return np.array([1/(nc[a]*freq[a,b]) for a,b in zip(g,cl)],dtype=np.float32)

def shuffle(raw,q,mask,meta,part,rng):
    result=np.zeros_like(raw);valid=np.zeros(len(raw),bool); remapped=0
    strata=defaultdict(lambda:defaultdict(list))
    for i in part:strata[meta['page'][i],meta['cls'][i]][meta['goal'][i]].append(i)
    for tasks in strata.values():
        if len(tasks)<2:continue
        gs=list(tasks);rng.shuffle(gs)
        for a,b in zip(gs,gs[1:]+gs[:1]):
            src=np.array(tasks[b]);dest=np.array(tasks[a])
            donors=rng.choice(src,len(dest),replace=len(src)<len(dest))
            for i,j in zip(dest,donors):
                x=raw[j,mask[j]];size=mask[i].sum()
                if len(x)!=size:remapped+=1
                chosen=x[np.rint(np.linspace(0,len(x)-1,size)).astype(int)]
                result[i,mask[i]]=chosen-(chosen*q[i,mask[i]]).sum()
                valid[i]=True
    return result,valid,remapped

def fit_fold(fold):
    start=time.monotonic()
    ts,meta,_=load_data();dev='cuda'
    h=F.normalize(ts.pop('h').to(dev).float());q=ts['q'].to(dev);mask=ts['mask'].to(dev)
    ids,inv=torch.unique(ts['ids'],return_inverse=True)
    head=torch.load(RUN/'head.pt',mmap=True,weights_only=True)
    e=F.normalize(head[ids].to(dev).float(),dim=-1);del head
    inv=inv.to(dev)
    def model_new(seed):
        torch.manual_seed(seed);return VisibleProjection(h.shape[1]).to(dev)
    def predict(net,ii):
        keys,loc=torch.unique(inv[ii],return_inverse=True)
        x=(net.state(h[ii])[:,None,:]*net.token(e[keys])[loc]).sum(-1)
        return x-(x*q[ii]).sum(-1,keepdim=True)
    tr=np.where(meta['fold']!=fold)[0];va=np.where(meta['fold']==fold)[0]
    path=RUN/'projector_versions';path.mkdir(exist_ok=True)
    def fit(y,train,seed,save_name=None):
        net=model_new(seed);optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        rng=np.random.default_rng(seed);ws=weights(meta,train);ws/=ws.mean()
        w=torch.zeros(len(h),device=dev);w[train]=torch.tensor(ws,device=dev)
        target=torch.as_tensor(y,dtype=torch.float,device=dev)
        for epoch in range(10):
            perm=rng.permutation(train)
            for sl in range(0,len(perm),256):
                ii=torch.tensor(perm[sl:sl+256],device=dev)
                err=F.huber_loss(predict(net,ii),target[ii],reduction='none',delta=1.)
                loss=(((err*mask[ii]).sum(-1)/mask[ii].sum(-1))*w[ii]).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
        if save_name:torch.save({k:v.cpu() for k,v in net.state_dict().items()},path/f'{save_name}_{fold}.pt')
        with torch.no_grad():
            return torch.cat([predict(net,torch.tensor(block,device=dev)).cpu() for block in np.array_split(va,max(1,(len(va)+511)//512))]).numpy()
    raw=ts['d'].numpy();valid=ts['mask'].numpy();qs=ts['q'].numpy()
    shuf,vtr,rem=shuffle(raw,qs,valid,meta,tr,np.random.default_rng(SEED+fold))
    sv,vva,remv=shuffle(raw,qs,valid,meta,va,np.random.default_rng(SEED+100+fold))
    shuf[va]=sv[va]
    with torch.no_grad():
        # Frozen planted factors share a fixed seed across folds.
        gen=model_new(SEED+900)
        planted=torch.cat([predict(gen,torch.arange(a,min(a+512,len(h)),device=dev)).cpu() for a in range(0,len(h),512)]).numpy()
    alpha=np.zeros(len(h),np.float32);inject=[]
    for page,cls in sorted(set(zip(meta['page'],meta['cls']))):
        group=(meta['page']==page)&(meta['cls']==cls)
        a=group & vtr; good=valid[a]
        if not good.any():continue
        x=shuf[a][good];y=planted[a][good]
        scale=x.std()/max(y.std(),1e-12)
        alpha[group]=scale
        inject.append(dict(page=page,category=cls,scale=float(scale),train_positions=int(a.sum())))
    planted*=alpha[:,None]
    same_seed=SEED+fold
    pred={}
    for label,y in [('content',raw),('original',ts['old'].numpy())]:
        pred[label]=fit(y,tr,same_seed,label)
        other=[i for i in range(4) if i!=fold]
        for j,ff in enumerate(other):
            pred[label+'_stability'+str(j)]=fit(y,np.where(meta['fold']==ff)[0],same_seed+ff+5)
    pred['null']=fit(shuf,np.where(vtr)[0],same_seed,'shuffled')
    pred['synthetic']=fit(shuf+planted,np.where(vtr)[0],same_seed)
    # Grouped, train-only simple baselines. Recenter both predictions on the recipient support.
    for label,y in [('content',raw),('original',ts['old'].numpy())]:
        ww=weights(meta,tr)[:,None]*valid[tr]/valid[tr].sum(1,keepdims=True)
        rank=(y[tr]*ww).sum(0)/ww.sum(0).clip(1e-12)
        token=ts['ids'].numpy()
        totals=np.bincount(token[tr].ravel(),weights=ww.ravel(),minlength=152064)
        sums=np.bincount(token[tr].ravel(),weights=(ww*y[tr]).ravel(),minlength=len(totals))
        for name,p in [('rank_mean',np.broadcast_to(rank,(len(va),9))),('token_only',(sums/totals.clip(1e-12))[token[va]])]:
            pred[label+'_'+name]=p-(p*qs[va]).sum(1,keepdims=True)
    np.savez(RUN/f'fold_{fold}.npz',idx=va,planted=planted[va],shuffled=shuf[va],power_valid=vva[va]&(alpha[va]>0),**pred)
    save(RUN/f'fit_{fold}.json',dict(fold=fold,train_tasks=len(set(meta['goal'][tr])),heldout_tasks=len(set(meta['goal'][va])),train_ids_sha256=digest_value(sorted(set(meta['goal'][tr].tolist()))),heldout_ids_sha256=digest_value(sorted(set(meta['goal'][va].tolist()))),donor_remapped=rem+remv,unmatched_train=int((~vtr[tr]).sum()),unmatched_test=int((~vva[va]).sum()),scales=inject,gpu_hours=(time.monotonic()-start)/3600))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',type=int);fit_fold(p.parse_args().fold)
