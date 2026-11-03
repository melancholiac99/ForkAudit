"""Paired held-out ALFWorld analysis for Hindsight synchronous OPD.

Reads existing E4-compatible full-game evaluation summaries only; never uses
Teacher logprob/KL as a surrogate for task success. Training seed and game are
both resampled as clusters while preserving paired arm comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exp2_common import read_json, save_json

CONDITIONS = ('M_native','M','L_guarded','LN_guarded')


def load_results(eval_root, seeds, conditions=CONDITIONS):
    """Returns arms -> [training_seed, game, evaluation_repeat] arrays."""
    eval_root=Path(eval_root)
    datasets={}
    game_ids=None;reps=None
    for c in conditions:
        per_seed=[]
        for seed in seeds:
            path=eval_root/f'{c}_s{seed}'/'summary.json'
            src=read_json(path)
            assert src['split']=='valid_unseen', f'{path}: expected unseen held-out split'
            assert 'per_game' in src
            games=sorted(src['per_game'])
            assert games and (game_ids is None or games==game_ids)
            game_ids=games
            actual=[src['per_game'][k] for k in games]
            arr=np.array(actual,dtype=float)
            assert arr.ndim==2 and all(x in (0,1) for x in arr.flat)
            assert reps is None or reps==arr.shape[1]
            reps=arr.shape[1]
            assert len(actual)==len(games)
            assert abs(arr.mean()-src['success'])<1e-10
            per_seed.append(arr)
        datasets[c]=np.stack(per_seed,axis=0)
    assert len(game_ids)>=2 and len(seeds)>=2, 'Need >=2 held-out games and training seeds'
    return datasets,game_ids


def paired_cluster_bootstrap(x,y,nboot=10000,seed=20261103):
    """Paired two-axis bootstrap: resample games and training seeds jointly.

    y and x use the exact same random game and seed selections. Replicates are
    averaged within each game/seed cell, not falsely treated as independent.
    """
    assert x.shape==y.shape and x.ndim==3
    assert x.shape[0]>=2 and x.shape[1]>=2
    diff=(x-y).mean(axis=-1)   # [training_seed, heldout_game]
    rng=np.random.default_rng(seed)
    vals=np.empty(nboot)
    for i in range(nboot):
        train_idx=rng.integers(0,diff.shape[0],size=diff.shape[0])
        game_idx=rng.integers(0,diff.shape[1],size=diff.shape[1])
        vals[i]=100*diff[np.ix_(train_idx,game_idx)].mean()
    lo,hi=np.percentile(vals,[2.5,97.5])
    return dict(diff_pp=float(100*diff.mean()),ci95_pp=[float(lo),float(hi)],
                n_training_seeds=x.shape[0],n_heldout_games=x.shape[1],n_eval_repeats=x.shape[2],
                bootstrap_reps=nboot,cluster_axes=['training_seed','heldout_game'])


def report(args):
    data,game_ids=load_results(args.eval_root,args.seeds)
    out={
        'status':'heldout_game_and_training_seed_paired_analysis',
        'design':'synchronous on-policy; exactly one optimizer step per fresh rollout batch',
        'conditions':{k:dict(mean_success=float(100*v.mean()),per_training_seed_success=
                                 [float(100*x.mean()) for x in v]) for k,v in data.items()},
        'primary_LN_minus_L':paired_cluster_bootstrap(data['LN_guarded'],data['L_guarded'],
                                                     nboot=args.nboot,seed=args.seed),
        'secondary_LN_minus_M':paired_cluster_bootstrap(data['LN_guarded'],data['M'],
                                                     nboot=args.nboot,seed=args.seed+1),
        'secondary_LN_minus_M_native':paired_cluster_bootstrap(data['LN_guarded'],data['M_native'],
                                                     nboot=args.nboot,seed=args.seed+2),
        'n_games':len(game_ids), 'training_seeds':args.seeds,
        'caveats':['Primary L_guarded vs LN_guarded contrast removes other-turn rationales; '
                   'current-turn think remains in the autoregressive prefix in both arms.',
                   'Teacher future logs contain the factual action in both privileged arms.',
                   'No Teacher logprob or KL comparison is interpreted as gradient quality proof.']
    }
    save_json(args.out,out)
    print(json.dumps(out,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root',required=True,help='E4 data/rollout/valid_unseen')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--nboot',type=int,default=10000)
    args=p.parse_args()
    assert len(set(args.seeds))==len(args.seeds)
    report(args)


if __name__=='__main__':main()
