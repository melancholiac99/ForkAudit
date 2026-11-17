"""Observe actual sampler/actor likelihoods; never change loss or PPO anchors."""
import json
from pathlib import Path
import time
import numpy as np


def ess_fraction(log_weights):
    x = np.asarray(log_weights, dtype=np.float64)
    if not len(x):return None
    w = np.exp(x - x.max())
    return float(w.sum()**2 / (len(w) * (w*w).sum()))


def summarize(actor, behavior, mask, advantages=None, trajectory_ids=None):
    actor, behavior = np.asarray(actor,dtype=np.float64), np.asarray(behavior,dtype=np.float64)
    mask = np.asarray(mask,dtype=bool)
    if actor.shape != behavior.shape or actor.shape != mask.shape:
        raise ValueError('Likelihood and mask shapes differ')
    delta = (actor-behavior)[mask]
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError('No finite active likelihood differences')
    ratio = np.exp(delta)
    gap = np.abs(np.exp(actor[mask])-np.exp(behavior[mask]))
    response_delta = np.where(mask, actor-behavior, 0).sum(-1)
    active = mask.any(-1)
    result = {'tokens':int(mask.sum()),'responses':int(active.sum()),
              'mean_abs_probability_difference':float(gap.mean()),
              'token_ratio_outside_08_12':float(((ratio<.8)|(ratio>1.2)).mean()),
              'token_ess_fraction':ess_fraction(delta),
              'response_ess_fraction':ess_fraction(response_delta[active]),
              # For x sampled from behavior mu and d=log(pi/mu), this is
              # exp(d)-1-d. The opposite orientation is not this estimator.
              'behavior_to_actor_k3_sample_estimate':float(np.expm1(delta).mean()-delta.mean()),
              'token_ratio_quantiles':{str(q):float(np.quantile(ratio,q)) for q in [0,.01,.5,.99,.999,1]},
              'abs_probability_difference_quantiles':{str(q):float(np.quantile(gap,q)) for q in [.5,.9,.95,.99,.999,1]},
              'abs_probability_difference_above':{str(t):float((gap>t).mean()) for t in [.01,.05,.1,.2,.5]}}
    if trajectory_ids is not None:
        sums={}
        for key,value,valid in zip(trajectory_ids,response_delta,active):
            if valid:sums[str(key)]=sums.get(str(key),0.)+float(value)
        result['trajectory_ess_fraction']=ess_fraction(list(sums.values()))
        result['trajectories']=len(sums)
    if advantages is not None:
        adv=np.asarray(advantages,dtype=np.float64)[mask]
        nz=adv!=0
        clipped=((adv>0)&(ratio>1.2))|((adv<0)&(ratio<.8))
        counterfactual=np.where(clipped,0.,adv*ratio)
        denom=np.linalg.norm(adv)*np.linalg.norm(counterfactual)
        result.update(nonzero_advantage_fraction=float(nz.mean()),
                      nonzero_advantage_ratio_outside_08_12=float(((ratio[nz]<.8)|(ratio[nz]>1.2)).mean()) if nz.any() else None,
                      counterfactual_initial_clipping_fraction=float(clipped[nz].mean()) if nz.any() else None,
                      surrogate_coefficient_cosine=float(adv@counterfactual/denom) if denom else None,
                      surrogate_coefficient_relative_l2=float(np.linalg.norm(counterfactual-adv)/np.linalg.norm(adv)) if nz.any() else None)
        result['counterfactual_note']='Behavior-anchor versus native-anchor coefficients before updates, using fixed advantages and .2 clip. Not parameter-gradient similarity, not observed reward impact; no correction applied.'
    return result


def capture(data, run, update):
    from .common import save, append, utc
    if 'rollout_log_probs' not in data.batch:return
    folder=Path(run)/'alignment_live';folder.mkdir(exist_ok=True)
    identity=f'step_{update:04d}_{time.time_ns()}'
    fields={dest:data.batch[source].detach().float().cpu().numpy()
            for dest,source in [('actor','old_log_probs'),('behavior','rollout_log_probs'),
                                ('mask','response_mask'),('advantages','advantages')]}
    decisions=[json.loads(str(x)) for x in data.non_tensor_batch['shop_decision']]
    path=folder/f'{identity}.npz'
    np.savez_compressed(path,**fields,responses=data.batch['responses'].detach().cpu().numpy())
    save(folder/f'{identity}_decisions.json',decisions)
    # Do not multiply trajectory weights by native DP padding duplicates.
    keep=[];seen=set()
    for i,row in enumerate(decisions):
        key=(row['trajectory_id'],row['step'])
        if key not in seen:keep.append(i);seen.add(key)
    unique={k:v[keep] for k,v in fields.items()}
    stats=summarize(**unique,trajectory_ids=[decisions[i]['trajectory_id'] for i in keep])
    append(folder/'metrics.jsonl',{'time':utc(),'update':update,'capture':identity,
                                 'saved_rows':len(decisions),'unique_rows':len(keep),**stats})
