"""Nested ASIN-fold full-action ridge, using only student final-action hidden states."""
import time
import numpy as np
import torch
from .b_collect import RUN,PARENT
from .b_fit import load_data
from .b_metrics import Cluster
from exploratory_shop_audit.grpo_forced_audit.common import read,save,append

def ridge():
    start=time.monotonic();_,_,points=load_data();split=read(PARENT/'asin_split_manifest.json')
    folds={r['goal_index']:split['cal_folds'][r['asin']] for r in split['T_cal']}
    xx=[];yy=[];gg=[];ff=[];pg=[];pos=[];probs=[]
    for k,v in enumerate(points):
        n=len(v['actions'])
        if n<2:continue
        q=v['action_none'].softmax(0);d=v['action_full']-v['action_placeholder'];y=d-(d*q).sum()
        xx.append(v['action_h']);yy.append(y);probs.extend(q.tolist());gg.extend([v['goal']]*n);ff.extend([folds[v['goal']]]*n);pg.extend([v['page']]*n);pos.extend([k]*n)
    x=torch.cat(xx).cuda().double();y=torch.cat(yy).cuda().double()
    goal=np.array(gg);fold=np.array(ff);page=np.array(pg);pos=np.array(pos);q=np.array(probs)
    def centered(values,ids):
        # Each row set is a union of whole states.
        g=pos[ids];return values-np.bincount(g,weights=q[ids]*values,minlength=len(points))[g]
    def fit(train,test,lam):
        tr=np.where(train)[0];te=np.where(test)[0]
        # Give each task unit total weight and scale to average sample weight one.
        _,gx,cts=np.unique(goal[tr],return_inverse=True,return_counts=True)
        w=torch.from_numpy(1/cts[gx]).to(x.device);w=w/w.mean()
        a=x[tr];mean=(a*w[:,None]).sum(0)/w.sum()
        sd=(((a-mean).square()*w[:,None]).sum(0)/w.sum()).sqrt().clamp_min(1e-6)
        a=(a-mean)/sd;center_y=(y[tr]*w).sum()/w.sum()
        aw=a*w.sqrt()[:,None];bw=(y[tr]-center_y)*w.sqrt()
        gram=aw.T@aw;rhs=aw.T@bw;ev=(x[te]-mean)/sd
        out={}
        for l in lam:
            mat=gram.clone();mat.diagonal().add_(l)
            coef=torch.cholesky_solve(rhs[:,None],torch.linalg.cholesky(mat)).flatten()
            out[l]=centered((ev@coef+center_y).cpu().numpy(),te)
        return out
    grid=[.1,1.,10.,100.];oof=np.zeros(len(y));chosen=[]
    truth=y.cpu().numpy()
    for outer in range(4):
        losses={l:[] for l in grid}
        for inner in range(4):
            if inner==outer:continue
            v=fold==inner
            preds=fit((fold!=inner)&(fold!=outer),v,grid)
            clusters=Cluster(goal[v])
            for l in grid:losses[l].extend(clusters.agg((truth[v]-preds[l])**2))
        l=min(grid,key=lambda l:np.mean(losses[l]));chosen.append(l)
        oof[fold==outer]=fit(fold!=outer,fold==outer,[l])[l]
    result={}
    for name,v in [('all',np.ones(len(y),bool))]+[(p,page==p) for p in sorted(set(page))]:
        c=Cluster(goal[v]);err=(oof[v]-truth[v])**2;zero=truth[v]**2
        base=c.agg(zero).mean();mse=c.agg(err).mean()
        result[name]=dict(actions=int(v.sum()),states=len(set(pos[v])),tasks=c.n,mse=mse,zero_mse=base,relative_gain=float(1-mse/base) if base else None,
                          gain_ci=c.estimate(zero-err))
    out=RUN/'reports';out.mkdir(exist_ok=True)
    save(out/'capacity_action_ridge_report.json',dict(scope='read-only full rank; last hidden after last action token; train-fold standardized features',target='length-normalized sel minus matched placeholder; student-relative centered',regularization_by_fold=chosen,groups=result,singleton_states_excluded=sum(len(p['actions'])<2 for p in points)))
    append(RUN/'b_cost.jsonl',dict(stage='action_ridge',gpu_hours=(time.monotonic()-start)/3600))
if __name__=='__main__':ridge()
