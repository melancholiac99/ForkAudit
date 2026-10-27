"""Post-freeze extension of the FINAL ruling: intervene on recorded arm identity.

No policy, environment, outcomes, frozen labels, or blind-analysis inputs are changed.
Permutation expectations are exact, not Monte Carlo approximations.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from exploratory_analyze import ci, KEY

OUT=Path(__file__).resolve().parent

def main():
    p=pd.read_parquet(OUT/'exploratory_point_mechanisms.parquet').set_index(KEY).sort_index()
    a=pd.read_parquet(OUT/'exploratory_arms.parquet')
    def mat(col):return a.pivot(index=KEY,columns='j',values=col).reindex(p.index).to_numpy(float)
    f=mat('target_mention');counts=f.sum(1);present=counts>0
    w=f/np.maximum(counts[:,None],1);w[~present,0]=1
    # Randomize only alternative identities. Factual observation, positive count,
    # and every permutation-invariant set statistic remain exactly unchanged.
    w_alt=np.column_stack([f[:,0],np.repeat(f[:,1:].sum(1)[:,None]/4,4,axis=1)])
    w_alt/=np.maximum(counts[:,None],1);w_alt[~present,0]=1
    w_all=np.ones_like(f)/5;w_all[~present]=0;w_all[~present,0]=1
    assert np.allclose(w.sum(1),1) and np.allclose(w_alt.sum(1),1)
    assert np.allclose(w_alt[:,0],w[:,0])
    res={'status':'exploratory; exact expectation over uniform identity permutations',
         'definition':'Select uniformly among target-response arms, or retain factual if none. Compare exact same rule with only alternative feedback identities permuted.',
         'groups':{},'counts':{}}
    groups={'all_live':p.live,'gold_unbiased':p.gold_layer.eq('unbiased'),
            'gold_candidate':p.gold_layer.eq('candidate'),'masked':p.masked,
            'at_least_four_sibling_arms':p.live&p.g_supported_arms.ge(4),
            'fewer_than_four_sibling_arms':p.live&p.g_supported_arms.lt(4),
            'has_exact_input_sibling':p.live&p.g_exact_input_hits.gt(0),
            'no_exact_input_sibling':p.live&p.g_exact_input_hits.eq(0)}
    ent=p.loc[p.live,'entropy_sampled'];ranks=ent.rank(method='average',pct=True)
    for label,mask in [('low',ranks.le(1/3)),('mid',ranks.gt(1/3)&ranks.le(2/3)),('high',ranks.gt(2/3))]:
        groups['entropy_'+label]=p.live & p.index.isin(mask[mask].index)
    tr=p.index.get_level_values(0).to_numpy()
    rows=[]
    for bank in ['3456','gold']:
        q=mat('q_'+bank);valid=np.isfinite(q).all(1)
        expected=(q*w).sum(1);permuted_alt=(q*w_alt).sum(1);permuted_all=(q*w_all).sum(1)
        for name,mask in groups.items():
            ix=np.asarray(mask)&valid
            res['groups'][bank+'_'+name]={
                'aligned_minus_factual':ci((expected-q[:,0])[ix],tr[ix]),
                'alt_permuted_minus_factual':ci((permuted_alt-q[:,0])[ix],tr[ix]),
                'alignment_increment':ci((expected-permuted_alt)[ix],tr[ix]),
                'all_permuted_minus_factual':ci((permuted_all-q[:,0])[ix],tr[ix]),
                'full_alignment_increment':ci((expected-permuted_all)[ix],tr[ix])}
        for j,key in enumerate(p.index):
            if valid[j]:rows.append(dict(traj_id=key[0],turn=int(key[1]),bank=bank,
                aligned_gain=float(expected[j]-q[j,0]),alt_permuted_gain=float(permuted_alt[j]-q[j,0]),
                alignment_increment=float(expected[j]-permuted_alt[j])))
    for k in range(6):
        ix=np.asarray(p.live)&(counts==k);x=p.iloc[np.where(ix)[0]]
        res['counts'][str(k)]={'n':len(x),'n_trajectories':x.index.get_level_values(0).nunique(),
            'fork_rate':float(x.het.mean()) if len(x) else None,
            'mean_C_cross':ci(x.C_cross,x.index.get_level_values(0))}
    pd.DataFrame(rows).to_csv(OUT/'exploratory_alignment_extension_points.csv',index=False)
    (OUT/'exploratory_alignment_extension.json').write_text(json.dumps(res,indent=2))
    print(json.dumps({k:res['groups'][k] for k in ['gold_gold_unbiased','gold_masked','3456_all_live',
         'gold_at_least_four_sibling_arms','gold_has_exact_input_sibling']},indent=2))

if __name__=='__main__':main()
