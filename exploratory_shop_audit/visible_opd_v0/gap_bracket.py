"""Recover the two centered section B contrasts on the archived closing-bracket cohort."""
import hashlib

import numpy as np
from scipy.special import logsumexp
import torch

from .gap_score import PUBLIC, STAGE_B
from .e25_paths import PUBLIC as OLD_PUBLIC
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save

SEED = 20261123


def statistics(x):
    return np.array([x.mean(), np.median(x), (x > 0).mean()])


def main():
    original = OLD_PUBLIC/'migration_2514/tcal_closing_bracket.json'
    ref = read(original)
    values, goal, old = [], [], []
    source = hashlib.sha256()
    for path in sorted((STAGE_B/'features').glob('*.pt')):
        source.update((path.name+' '+digest_file(path)+'\n').encode())
        r = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
        rows, cols = np.where((np.array(r['classes'])=='search')[:,None]
                              & r['mask'].numpy() & (r['ids'].numpy()==ref['token_id']))
        if not len(rows):
            continue
        # B stored full vocabulary student log p, masses, and two centered contrasts.
        # Additive row constants cancel when conditioning each distribution on C.
        s = r['s'][rows].double().numpy()
        a = r['old'][rows].double().numpy()
        d = r['d'][rows].double().numpy()
        mask = r['mask'][rows].numpy()
        qs = r['q'][rows].double().numpy()
        qs /= qs.sum(axis=1, keepdims=True)
        def reconstruct(logits, mass):
            norm = logsumexp(np.where(mask, logits, -np.inf), axis=1, keepdims=True)
            return logits - norm + np.log(r['mass_'+mass][rows].double().numpy())[:,None]
        full = reconstruct(s+a, 'full')
        ph = reconstruct(s+a-d, 'placeholder')
        def center(delta):
            return delta - (qs*delta).sum(1, keepdims=True)
        ix = np.arange(len(rows))
        values.extend(zip(center(full-s)[ix,cols],center(ph-s)[ix,cols]))
        goal.extend([r['goal']]*len(rows))
        old.extend(d[ix,cols])
    x = np.asarray(values)
    tasks, inverse = np.unique(goal, return_inverse=True)
    assert (len(x),len(tasks))==(ref['overall']['n'],ref['overall']['tasks'])==(1701,471)
    assert abs(np.mean(old)-ref['overall']['mean'][0])<1e-9
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(inverse==i) for i in range(len(tasks))]
    samples = np.empty((10000,2,3))
    for b in range(len(samples)):
        ix = np.concatenate([groups[i] for i in rng.integers(len(tasks),size=len(tasks))])
        for j in range(2):
            samples[b,j] = statistics(x[ix,j])
    result = {}
    for j, label in enumerate(('full_minus_none','placeholder_minus_none')):
        result[label] = {name:dict(estimate=float(pt),ci95=ci.tolist()) for name,pt,ci in
            zip(('mean','median','positive_fraction'),statistics(x[:,j]),np.quantile(samples[:,j],[.025,.975],axis=0).T)}
    save(PUBLIC/'closing_bracket_decomposition.json',dict(
        positions=len(x),tasks=len(tasks),bootstrap_draws=len(samples),seed=SEED,
        storage='Reconstructed conditional log probabilities from archived s, centered full−none, centered full−placeholder, and three support masses. The identity is an algebraic storage round trip, not an independent raw-logit audit.',
        max_abs_error=float(np.max(np.abs(x[:,0]-x[:,1]-old))),statistics=result,
        old_cohort_sha256=digest_file(original),feature_files_digest=source.hexdigest()))


if __name__=='__main__':
    main()
