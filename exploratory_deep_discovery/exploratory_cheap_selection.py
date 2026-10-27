"""Outcome-free one-step selectors evaluated on existing replay pools; exploratory."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exploratory_analyze import ci, KEY
OUT=Path(__file__).resolve().parent

def main():
    p=pd.read_parquet(OUT/'exploratory_point_mechanisms.parquet').set_index(KEY).sort_index()
    a=pd.read_parquet(OUT/'exploratory_arms.parquet');res={}
    for feat in ['target_mention','target_action','new_to_history']:
        f=a.pivot(index=KEY,columns='j',values=feat).loc[p.index].to_numpy(float)
        w=f/np.maximum(f.sum(1)[:,None],1);w[f.sum(1)==0,0]=1
        res[feat]={}
        for bank in ['12','3456','gold']:
            q=a.pivot(index=KEY,columns='j',values='q_'+bank).loc[p.index].to_numpy()
            selected=(q*w).sum(1)
            for name,mask in [('Live',p.live),('Masked',p.masked),('Gold_unbiased',p.gold_layer.eq('unbiased'))]:
                ix=np.asarray(mask)&np.isfinite(q).all(1)
                res[feat][bank+'_'+name]={'vs_factual':ci((selected-q[:,0])[ix],p.index.get_level_values(0)[ix]),
                    'vs_uniform_five':ci((selected-q.mean(1))[ix],p.index.get_level_values(0)[ix])}
    # Confirm world-pair matches have identical admissible action sets and terminal flags too.
    pairs=pd.read_parquet(OUT/'exploratory_same_facts_details.parquet');bad=[]
    for z in pairs.itertuples():
        pk=json.load(open(OUT.parent/f'data/peek_alts/{z.traj_id}/t{z.turn:02d}.json'))
        arms=sorted(pk['peeks'],key=lambda q:(q['role']!='factual',q['alt_index'] or 0));h,l=arms[z.hi],arms[z.lo]
        if sorted(h['admissible_after'])!=sorted(l['admissible_after']) or h['done']!=l['done'] or h['won']!=l['won']:
            bad.append((z.traj_id,z.turn,z.hi,z.lo))
    assert not bad
    res['matched_world_validation']={'n_pairs':len(pairs),'admissible_done_won_mismatches':len(bad)}
    (OUT/'exploratory_cheap_selection.json').write_text(json.dumps(res,indent=2))
    for f in ['target_mention','target_action','new_to_history']:
        print(f,json.dumps({k:res[f][k] for k in ['3456_Live','gold_Live','gold_Masked','gold_Gold_unbiased']}))

if __name__=='__main__':main()
