"""Recover sampled-action target log ratios where the block anchor was retained."""
import argparse
import csv
import gzip
import io

import numpy as np
import torch

from exploratory_shop_audit.grpo_forced_audit.common import ROOT, digest_file, read, save
from .gap_cpu_packet import DRAWS, OUT, SEED, cluster_mean, log_probs

RUN = ROOT/'.runtime/runs/shop_gap26_20261123'
CACHE = RUN/'posthoc_cpu'
FORMAL = RUN/'formal'
CATEGORIES = ('all', 'search', 'click', 'option', 'format')


def upper_log_ratio(a, b, limit):
    # For p <= a, x=log(a/p) solves the increasing side of the Bernoulli KL.
    # Since -b log b <= a, (limit+a)/a brackets every feasible x.
    lo = np.zeros_like(a)
    hi = 1. + limit/a
    for _ in range(64):
        mid = (lo+hi)*.5
        value = a*mid - b*np.log1p((a/b)*(-np.expm1(-mid)))
        small = value < limit
        lo = np.where(small, mid, lo)
        hi = np.where(small, hi, mid)
    return hi


def sampled_bounds(q, index, strength):
    target = q[np.arange(len(q)), index]
    rest = q.copy()
    rest[np.arange(len(q)), index] = 0
    complement = rest.sum(1)
    mass = target+complement
    assert (target>0).all() and (complement>0).all() and np.max(abs(mass-1.))<3e-6
    a, b = target/mass, complement/mass
    # The archive stores a float32 KL of a float32 target (slightly unnormalized).
    pad = 1e-5 * (1+np.abs(strength))
    limit = np.maximum(0., (strength+pad)/mass-np.log(mass))
    hi = upper_log_ratio(a,b,limit)
    other_hi = upper_log_ratio(b,a,limit)
    lo = -np.log1p((b/a)*(-np.expm1(-other_hi)))
    return np.column_stack((lo+np.log(mass), hi+np.log(mass)))


def sampled_columns(ids, mask, answer):
    match = (ids == np.asarray(answer)[:, None]) & mask
    if not np.all(match.sum(1)==1):
        raise ValueError('The sampled token must occur once on the masked support')
    return match.argmax(1)


def archive_rows(path):
    m = read(path/'target_manifest.json')
    names = ('student','target','bf16_before','strength')
    data = []
    for name in names:
        f = path/(name+'.pt')
        assert digest_file(f)==m[name+'_sha256']
        data.append(torch.load(f, map_location='cpu', weights_only=True))
    student, targets, anchors, strengths = data
    assert len(set(map(len,data)))==1
    mask = np.vstack([r['mask'].numpy() for r in student])
    ids = np.vstack([r['ids'].numpy() for r in student])
    index = sampled_columns(ids, mask, np.concatenate([r['answer'] for r in student]))
    q = np.vstack([t.numpy() for t in targets]).astype(float)
    kls = np.concatenate([s.numpy() for s in strengths]).astype(float)
    assert np.all(q[~mask]==0)
    bounds = sampled_bounds(q,index,kls)
    block_start = m['block_sha']==m['scorer_sha']
    assert block_start == ((int(path.parent.name)-1)%5==0)
    exact = None
    if block_start:
        lp = log_probs(np.vstack([s.numpy() for s in anchors]).astype(float),mask)
        ix = np.arange(len(q))
        exact = np.log(q[ix,index])-lp[ix,index]
        assert np.all(bounds[:,0]-1e-10<=exact) and np.all(exact<=bounds[:,1]+1e-10)
        direct = (q*(np.log(np.where(q>0,q,1.))-lp)).sum(1)
        assert np.all(np.abs(direct-kls)<1e-5*(1+abs(kls)))
    records = []
    pos = 0
    for r in student:
        key = [r['goal'],r['rep'],r['step']]
        length = len(r['answer'])
        s = slice(pos,pos+length)
        v = float(exact[s].sum()) if block_start else None
        lo,hi = [v,v] if block_start else bounds[s].sum(0).tolist()
        records.append([*key,length,v,lo,hi])
        pos += length
    records.extend([r['goal'],r['rep'],r['step'],0,None,None,None] for r in m['na_records'])
    assert pos==len(q) and len(records)==len({tuple(r[:3]) for r in records})==m['rollout_records']
    return records


def paths(arm):
    return sorted(f.parent for f in (FORMAL/'pilot').glob('*/'+arm+'/optimizer.json'))


def prepare():
    CACHE.mkdir(exist_ok=True)
    for arm in ('R-gap','P-gap'):
        for path in paths(arm):
            k = int(path.parent.name)
            save(CACHE/f'scores_{arm}_{k:02}.json', archive_rows(path))


def summarize(rows):
    # Each form of coverage is explicit. Bounds exclude structurally unscored records.
    exact = [r for r in rows if r[8] is not None]
    valid = [r for r in rows if r[9] is not None]
    point = cluster_mean([r[2] for r in exact], [[r[8]] for r in exact])
    interval = cluster_mean([r[2] for r in valid], [[r[9],r[10]] for r in valid])
    ci = interval['ci95']
    return dict(records=len(rows), scored=len(valid), structural_na=len(rows)-len(valid),
                block_anchor_missing=len(valid)-len(exact), exact_subset=point,
                all_scored_identification_bounds=interval['mean'], all_scored_tasks=interval['tasks'],
                bootstrap_outer_limits=[ci[0][0],ci[1][1]] if ci else None)


def report():
    windows, digests, overall = [], {}, {}
    all_rows = []
    for arm in ('R-gap','P-gap'):
        arm_rows = []
        for path in paths(arm):
            k = int(path.parent.name)
            digests[f'{arm}/{k:02}'] = digest_file(path/'target_manifest.json')
            meta = read(CACHE/f'{arm}_{k:02}.json')
            records = read(CACHE/f'scores_{arm}_{k:02}.json')
            scores = {tuple(r[:3]):r[3:] for r in records}
            assert len(scores)==len(records)==len(meta)
            for g,rep,step,legal,cls in meta:
                arm_rows.append([arm,k,g,rep,step,legal,cls,*scores.pop((g,rep,step))])
            assert not scores
        all_rows.extend(arm_rows)
        def by_class(subset):
            return {status:{c:summarize([r for r in subset if r[5]==legal and (c=='all' or r[6]==c)])
                            for c in CATEGORIES} for status,legal in [('legal',True),('illegal',False)]}
        overall[arm] = by_class(arm_rows)
        for first in range(1,len(paths(arm))+1,5):
            end = min(first+4,len(paths(arm)))
            windows.append(dict(arm=arm,start=first,end=end,by_class=by_class(
                [r for r in arm_rows if first<=r[1]<=end])))
    output = OUT/'sampled_action_scores.csv.gz'
    with output.open('wb') as f, gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as gz, io.TextIOWrapper(gz,newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('arm','update','task','rep','action_index','legal','action_class','scored_tokens',
                         'exact_log_ratio','lower_log_ratio','upper_log_ratio'))
        writer.writerows(all_rows)
    save(OUT/'target_score_windows.json', dict(units='sum of sampled-token log ratios per action, nats',
         seed=SEED,bootstrap_draws=DRAWS,bootstrap_unit='task',weight='equal action records',
         exact_subset='Archived actor is the block student on updates 1,6,11,... only.',
         bounds='All scored records. Non-anchor token bounds invert binary KL after coarse-graining the stored forward KL; allow 1e-5*(1+abs(KL)) FP32 error. Sum token bounds per action. bootstrap_outer_limits are percentiles of bound endpoints, not a point estimate or its CI.',
         row_file_sha256=digest_file(output),input_manifest_sha256=digests,overall=overall,windows=windows))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','report'])
    if parser.parse_args().mode=='prepare':
        prepare()
    else:
        report()
