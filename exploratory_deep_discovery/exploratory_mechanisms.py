"""Targeted follow-ups: world-identical contrasts, masked geometry, CWGT replication."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, hashlib, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from exploratory_analyze import ci, ratio_ci, auc, oof, KEY, SEED

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'scripts'))
import importlib
R=importlib.import_module('15_regimes')

def main():
    p=pd.read_parquet(OUT/'exploratory_point_mechanisms.parquet').set_index(KEY)
    a=pd.read_parquet(OUT/'exploratory_arms.parquet')
    result={};pairs=[];case_ids=[]
    for key,x in a.groupby(KEY,sort=True):
        if not p.loc[key,'live']:continue
        tid,t=key;pk=json.load(open(ROOT/f'data/peek_alts/{tid}/t{t:02d}.json'))
        pp=sorted(pk['peeks'],key=lambda q:(q['role']!='factual',q['alt_index'] or 0))
        sig=[(tuple(sorted(q['facts_added'])),tuple(sorted(q['facts_removed']))) for q in pp]
        ar=x.sort_values('j').to_dict('records')
        for i,j in itertools.combinations(range(5),2):
            if ar[i]['target_mention']==ar[j]['target_mention']:continue
            hi,lo=(i,j) if ar[i]['target_mention'] else (j,i)
            z=dict(traj_id=tid,turn=t,hi=hi,lo=lo,action_hi=ar[hi]['action'],action_lo=ar[lo]['action'],
                obs_hi=ar[hi]['obs'],obs_lo=ar[lo]['obs'],same_facts=sig[hi]==sig[lo],
                neither_changes_facts=not ar[hi]['facts_changed'] and not ar[lo]['facts_changed'],
                neither_manipulates=ar[hi]['action_type'] in ['goto','look','examine','inventory','open','close'] and
                                   ar[lo]['action_type'] in ['goto','look','examine','inventory','open','close'],
                same_action_type=ar[hi]['action_type']==ar[lo]['action_type'],
                hi_type=ar[hi]['action_type'],lo_type=ar[lo]['action_type'],
                new_history=ar[hi]['new_to_history'],new_prompt=ar[hi]['new_to_prompt'],
                take_already_available=ar[hi]['take_available_before'],
                prompt_target=bool(p.loc[key,'prompt_target']),het=bool(p.loc[key,'het']),masked=bool(p.loc[key,'masked']))
            for bank in ['12','34','56','3456','gold']:
                z['diff_'+bank]=ar[hi]['q_'+bank]-ar[lo]['q_'+bank]
            pairs.append(z)
    pairs=pd.DataFrame(pairs)
    pairs.to_parquet(OUT/'exploratory_mention_pairs.parquet',index=False)
    result['matched_world']={}
    conditions={'all_mixed':np.ones(len(pairs),bool),'same_facts':pairs.same_facts,
        'neither_changes_facts':pairs.neither_changes_facts,
        'same_facts_known_prompt':pairs.same_facts&~pairs.new_prompt,
        'same_facts_new_prompt':pairs.same_facts&pairs.new_prompt,
        'same_facts_same_action_type':pairs.same_facts&pairs.same_action_type,
        'non_manipulation':pairs.neither_manipulates,
        'new_to_history':pairs.new_history,'known_to_history':~pairs.new_history}
    for name,mask in conditions.items():
        x=pairs[mask];pt=x.groupby(KEY)[['diff_12','diff_3456','diff_gold']].mean().reset_index()
        result['matched_world'][name]={'n_pairs':len(x),'n_points':len(pt),
            **{bank:ci(pt['diff_'+bank],pt.traj_id) for bank in ['12','3456','gold']}}
    # Cases selected by a large contrast in discovery bank, accompanied by independent counts (not population estimates).
    cand=pairs[pairs.same_facts&~pairs.new_prompt].sort_values('diff_3456',ascending=False)
    seen=set()
    for row in cand.to_dict('records'):
        if row['traj_id'] in seen:continue
        case_ids.append(row);seen.add(row['traj_id'])
        if len(case_ids)==8:break
    result['world_pair_cases']=case_ids
    result['samefacts_type_pairs']=pairs[pairs.same_facts].groupby(['hi_type','lo_type']).agg(n=('diff_3456','size'),
          discovery=('diff_3456','mean'),bank12=('diff_12','mean')).reset_index().to_dict('records')
    # Non-rejection vs exact/near cancellation, pair effects, and independent chosen ranks.
    pair_orig=pd.read_parquet(ROOT/'eval/pairs_points.parquet')
    summary=[]
    for key,x in a.groupby(KEY):
        if not p.loc[key,'masked']:continue
        x=x.sort_values('j');q=x.q_3456.to_numpy();g=x.q_gold.to_numpy();u=x.q_12.to_numpy()
        sx=pair_orig[(pair_orig.traj_id==key[0])&(pair_orig.turn==key[1])]
        sigbetter=(sx.sig&(sx.C_j<0)).any();sigworse=(sx.sig&(sx.C_j>0)).any()
        best=int(p.loc[key,'best_j_3456']);worst=int(p.loc[key,'worst_j_3456'])
        lo,hi=R.newcombe(x.iloc[best].s_gold,x.iloc[best].n_gold,x.iloc[0].s_gold,x.iloc[0].n_gold) if np.isfinite(g).all() else (np.nan,np.nan)
        summary.append(dict(traj_id=key[0],turn=key[1],A=q[0]-q[1:].mean(),
            sample_adv_exact_zero=abs(q[0]-q[1:].mean())<1e-9,small=abs(q[0]-q[1:].mean())<=1/12+1e-9,
            sig_better=bool(sigbetter),sig_worse=bool(sigworse),sig_both=bool(sigbetter and sigworse),
            gap_best=q[best]-q[0],gap_worst=q[0]-q[worst],gain12=u[best]-u[0],gainGold=g[best]-g[0],
            gold_sig_better=bool(lo>0) if np.isfinite(lo) else np.nan,
            rank12=1+sum(u[1:]>u[0])+.5*sum(u[1:]==u[0]),
            rankGold=1+sum(g[1:]>g[0])+.5*sum(g[1:]==g[0]) if np.isfinite(g).all() else np.nan))
    m=pd.DataFrame(summary);m.to_csv(OUT/'exploratory_masked_details.csv',index=False)
    result['masked_detailed']={col:ci(m[col],m.traj_id) for col in ['sample_adv_exact_zero','small','sig_better','sig_worse','sig_both',
        'gap_best','gap_worst','gain12','gainGold','gold_sig_better','rank12','rankGold']}
    result['masked_small']={k:ci(x.gainGold,x.traj_id) for k,x in m.groupby('small')}
    # A low-dimensional logging-vs-peek contrast, fixed logistic C=1, no feature search.
    L=p[p.live].reset_index();base=['turn','n_admissible','obs_target','prompt_target']
    models={'logging':base,'logging_plus_peek':base+['any_target_mention','factual_target_mention','alt_target_mention'],
        'logging_plus_take':base+['any_take_available_after'],'factual_response':base+['factual_target_mention'],
        'factual_plus_counterfactual':base+['factual_target_mention','alt_target_mention']}
    predictions={k:oof(L[fs],L.het,L.traj_id) for k,fs in models.items()}
    result['logging_vs_peek']={k:auc(L.het,v) for k,v in predictions.items()}
    # Use exact bootstrap AUC code from frozen evaluator; reduced 2k exploratory repetitions.
    import evalcore as E
    boot=E.Boot(L.traj_id,B=2000,seed=SEED)
    distributions={k:boot.auroc_dist(L.het,v) for k,v in predictions.items()}
    result['logging_vs_peek_ci']={k:E.Boot.ci(v) for k,v in distributions.items()}
    result['counterfactual_increment']={'diff':result['logging_vs_peek']['factual_plus_counterfactual']-result['logging_vs_peek']['factual_response'],
        'ci95':E.Boot.ci(distributions['factual_plus_counterfactual']-distributions['factual_response'])}
    # Preserve alignment vs per-state arm-label shuffling; no classifier needs fitting.
    rng=np.random.default_rng(SEED)
    armrows=a.set_index(KEY).loc[p[p.live].index].reset_index().sort_values(KEY+['j'])
    ff=armrows.target_mention.to_numpy(float).reshape(-1,5);qq=armrows.q_3456.to_numpy().reshape(-1,5)
    label=p[p.live].sort_index().A_3456.to_numpy();nz=label!=0;cap=p[p.live].sort_index().cap.to_numpy()
    score=ff[:,0]-ff[:,1:].mean(1);null=[];nullcap=[]
    for _ in range(1000):
        fp=rng.permuted(ff,axis=1);sc=fp[:,0]-fp[:,1:].mean(1)
        null.append(auc(label[nz]>0,sc[nz]));nullcap.append(auc(label[cap]>0,sc[cap]))
    result['alignment_shuffle']={'all_live_nonzero':auc(label[nz]>0,score[nz]),
        'null95':np.quantile(null,[.025,.975]).tolist(),'B3':auc(label[cap]>0,score[cap]),
        'B3_null95':np.quantile(nullcap,[.025,.975]).tolist()}
    # CWGT: all 172 fixed turns, counts only. Never execute downloaded code.
    path=ROOT/'downloads/openreview_supp/supplementry/submission_source/scientific/inputs/evidence.json'
    ev=json.load(open(path));turns={}
    for bank in ['A','B']:
        for row in ev['arms'][bank]:
            key=(row['trajectory'],row['turn']);j=0 if row['kind']=='factual' else 1+row['k']
            turns.setdefault(key,{}).setdefault(bank,[np.nan]*5)[j]=row['successes']/row['n']
    keys=sorted(k for k,v in turns.items() if set(v)=={'A','B'})
    qa,qb=[np.array([turns[k][bank] for k in keys]) for bank in ['A','B']];tr=np.array([k[0] for k in keys])
    ca=np.mean((qa-qa.mean(1)[:,None])*(qb-qb.mean(1)[:,None]),1)
    ac=4/25*(qa[:,0]-qa[:,1:].mean(1))*(qb[:,0]-qb[:,1:].mean(1))
    best=qa[:,1:].argmax(1)+1;gain=qb[np.arange(len(qb)),best]-qb[:,0]
    small=np.abs(qa[:,0]-qa[:,1:].mean(1))<=1/16+1e-9
    result['cwgt']={'n':len(keys),'n_traj':len(set(tr)),'C':ci(ca,tr),'chosen_C':ci(ac,tr),
        'alt_share':ratio_ci(ca-ac,ca,tr),'gain_Aselect_Beval':ci(gain,tr),'small_A_gain':ci(gain[small],tr[small]),
        'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'note':'All fixed turns; no M0 labels transferred; A selects and B evaluates.'}
    def default(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    (OUT/'exploratory_mechanisms.json').write_text(json.dumps(result,indent=2,default=default))
    print(json.dumps({k:v for k,v in result.items() if k!='world_pair_cases'},indent=2,default=default),flush=True)

if __name__=='__main__':main()
