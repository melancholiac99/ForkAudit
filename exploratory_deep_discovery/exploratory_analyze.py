"""Mechanism-oriented exploratory analyses. No frozen file is written."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

OUT=Path(__file__).resolve().parent
KEY=['traj_id','turn']
SEED=27102026
def ci(x,g,B=4000):
    x=np.asarray(x,float);g=np.asarray(g);ok=np.isfinite(x);x=x[ok];g=g[ok]
    if not len(x):return dict(n=0,mean=None,ci95=None)
    u,iv=np.unique(g,return_inverse=True);s=np.bincount(iv,weights=x);n=np.bincount(iv)
    w=np.random.default_rng(SEED).multinomial(len(u),np.ones(len(u))/len(u),size=B)
    v=(w@s)/(w@n)
    return dict(n=len(x),n_traj=len(u),mean=float(x.mean()),ci95=np.quantile(v,[.025,.975]).tolist())
def ratio_ci(x,y,g):
    u,iv=np.unique(g,return_inverse=True);s=np.bincount(iv,weights=x);n=np.bincount(iv,weights=y)
    w=np.random.default_rng(SEED).multinomial(len(u),np.ones(len(u))/len(u),size=4000)
    v=(w@s)/(w@n)
    return dict(ratio=float(np.sum(x)/np.sum(y)),ci95=np.quantile(v,[.025,.975]).tolist())
def auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float);ok=np.isfinite(s)
    return float(roc_auc_score(y[ok],s[ok])) if len(np.unique(y[ok]))==2 else None
def report_group(df,cols):
    rows=[]
    for key,x in df.groupby(cols,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        d=dict(zip(cols,key));d.update(n=len(x),n_traj=x.traj_id.nunique(),live=int(x.live.sum()),fork=int(x.het.sum()),
            fork_live=x.loc[x.live,'het'].mean(),masked=int(x.masked.sum()))
        rows.append(d)
    return rows
def oof(x,y,g):
    x=np.asarray(x,float);y=np.asarray(y);g=np.asarray(g);out=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(x,y,g):
        m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=1,max_iter=500))
        m.fit(x[tr],y[tr]);out[te]=m.predict_proba(x[te])[:,1]
    return out

def main():
    p=pd.read_parquet(OUT/'exploratory_points.parquet').sort_values(KEY).reset_index(drop=True)
    a=pd.read_parquet(OUT/'exploratory_arms.parquet')
    c=pd.read_parquet(OUT/'exploratory_continuations.parquet')
    ix=pd.MultiIndex.from_frame(p[KEY])
    def mat(col):return a.pivot(index=KEY,columns='j',values=col).reindex(ix).to_numpy(float)
    Q={s:mat('q_'+s) for s in ['12','34','56','3456','gold']}
    q1,q2=Q['34'],Q['56'];mu1,mu2=q1.mean(1),q2.mean(1)
    p['C_cross']=np.mean((q1-mu1[:,None])*(q2-mu2[:,None]),1)
    p['T_cross']=(mu1+mu2)/2-mu1*mu2
    p['N_cross']=np.mean((q1+q2)/2-q1*q2,1)
    adv=lambda q:q[:,0]-q[:,1:].mean(1)
    p['C_chosen_cross']=4/25*adv(q1)*adv(q2)
    p['C_alt_cross']=4/5*np.mean((q1[:,1:]-q1[:,1:].mean(1)[:,None])*(q2[:,1:]-q2[:,1:].mean(1)[:,None]),1)
    assert np.allclose((p.C_chosen_cross+p.C_alt_cross)[p.live],p.C_cross[p.live])
    assert np.allclose((p.C_cross+p.N_cross)[p.live],p.T_cross[p.live])
    p['C_plugin']=np.var(Q['3456'],axis=1)
    q=Q['3456'];ridx=np.arange(len(p))
    best=1+np.nan_to_num(q[:,1:],nan=-1).argmax(1)
    worst=1+np.nan_to_num(q[:,1:],nan=2).argmin(1)
    best34=1+np.nan_to_num(q1[:,1:],nan=-1).argmax(1)
    p['gain_12']=Q['12'][ridx,best]-Q['12'][:,0]
    p['gain_gold']=Q['gold'][ridx,best]-Q['gold'][:,0]
    p['gain_56_from34']=Q['56'][ridx,best34]-Q['56'][:,0]
    p['best_j_3456']=best;p['worst_j_3456']=worst
    p['chosen_over_worst_12']=Q['12'][:,0]-Q['12'][ridx,worst]
    p['chosen_over_worst_gold']=Q['gold'][:,0]-Q['gold'][ridx,worst]
    p['chosen_rank_3456']=1+(q[:,1:]>q[:,0,None]).sum(1)+.5*(q[:,1:]==q[:,0,None]).sum(1)
    p['bracket_3456']=(q[:,1:].max(1)>q[:,0])&(q[:,1:].min(1)<q[:,0])
    p['bracket_confirm_12']=(p.gain_12>0)&(p.chosen_over_worst_12>0)
    p['bracket_confirm_gold']=(p.gain_gold>0)&(p.chosen_over_worst_gold>0)
    p['bracket_confirm_gold']=p['bracket_confirm_gold'].astype(float).where(p.gain_gold.notna())
    for col in ['gain_12','chosen_over_worst_12','chosen_rank_3456','bracket_3456','bracket_confirm_12']:
        p[col]=p[col].astype(float).where(p.live)
    p['small_adv']=(np.abs(p.A_3456)<=1/12+1e-9)&p.live
    p['regime']=np.where(~p.live,'Dead',np.where(p.masked,'Masked',np.where(p.het,'CreditFork','Indifferent')))
    res={'definitions':{'C_cross':'mean_j[(q34_j-mean(q34))*(q56_j-mean(q56))]',
          'gain_12':'best alternative selected on 3-6, evaluated in 1-2 vs factual; Live selection uses 1-2',
          'gain_gold':'same selector evaluated on disjoint gold; gold sampled in two strata',
          'gain_56_from34':'select on 3-4; evaluate 5-6; main result on all Live, no label-based selection'},
         'n_points':len(p),'n_arms':len(a),'n_continuations':len(c),
         'task_regimes':report_group(p,['task_type']),
         'time_regimes':report_group(p,['won','prog3'])}
    L=p[p.live]
    res['decomposition']={k:ci(L[k],L.traj_id) for k in ['C_cross','T_cross','N_cross','C_chosen_cross','C_alt_cross','C_plugin']}
    res['decomposition']['controllable_share']=ratio_ci(L.C_cross,L.T_cross,L.traj_id)
    res['decomposition']['alt_share']=ratio_ci(L.C_alt_cross,L.C_cross,L.traj_id)
    res['decomposition']['corr_uncertainty_control']=float(spearmanr(L.T_cross,L.C_cross).statistic)
    res['regime_summary']={}
    for name,x in p.groupby('regime'):
        res['regime_summary'][name]={k:ci(x[k],x.traj_id) for k in ['gain_12','gain_gold','chosen_over_worst_12',
            'chosen_over_worst_gold','chosen_rank_3456','bracket_3456','bracket_confirm_12','bracket_confirm_gold','small_adv']}
    res['gain_allLive_split']=ci(L.gain_56_from34,L.traj_id)
    res['small_adv_gain']=ci(L.loc[L.small_adv,'gain_12'],L.loc[L.small_adv,'traj_id'])
    res['masked_gold_strata']={}
    for k,x in p[p.masked&p.gold_layer.notna()].groupby('gold_layer'):
        res['masked_gold_strata'][k]={col:ci(x[col],x.traj_id) for col in ['gain_gold','chosen_over_worst_gold','bracket_confirm_gold']}
    # Reference distributions: restricted to four sampled alternatives, not true full-policy V.
    weights=mat('draft_count')[:,1:]+.5;weights/=weights.sum(1)[:,None]
    for bank in ['3456','12','gold']:
        p['A_policy_'+bank]=Q[bank][:,0]-(weights*Q[bank][:,1:]).sum(1)
    p['reference_flip']=(p.A_3456*p.A_policy_3456<0)&p.live
    p['reference_flip_confirm12']=p.reference_flip&(adv(Q['12'])*p.A_policy_12<0)
    res['reference']={'flip':ci(L.index.map(p.reference_flip),L.traj_id),
        'flip_replication12':ci(p.loc[p.reference_flip,'reference_flip_confirm12'],p.loc[p.reference_flip,'traj_id'])}
    # W1 semantics and within-point action alignment
    res['w1_semantics']={}
    for label,mask in [('D',np.ones(len(p),bool)),('Live',p.live),('Fork',p.het),('Masked',p.masked)]:
        x=p[mask&p.any_target_mention]
        res['w1_semantics'][label]={k:ci(x[k],x.traj_id) for k in ['ever_target','prompt_target','obs_target',
           'any_new_to_history','any_new_to_prompt','factual_target_mention','alt_target_mention','any_take_available_after']}
    res['w1_subsets']=report_group(p,['prompt_target','any_target_mention','any_new_to_history'])
    res['w1_aucs']={k:auc(L.het,L[k]) for k in ['target_revealed','obs_target','prompt_target','ever_target',
        'any_new_to_history','any_new_to_prompt','factual_target_mention','alt_target_mention','any_take_available_after','any_take_newly_available']}
    res['w1_alignment']={};res['w1_selector']={}
    for feat in ['target_mention','new_to_history','new_to_prompt','take_available_after','take_newly_available','target_action']:
        f=mat(feat);mixed=(f.sum(1)>0)&(f.sum(1)<5)&p.live
        with np.errstate(invalid='ignore',divide='ignore'):
            delta=(q*f).sum(1)/f.sum(1)-(q*(1-f)).sum(1)/(1-f).sum(1)
            # Choose uniformly among arms with f=1, otherwise retain factual.
            gain=np.where(f.sum(1)>0,(q*f).sum(1)/f.sum(1)-q[:,0],0)
        p['aligned_'+feat]=delta
        res['w1_alignment'][feat]=ci(delta[mixed],p.loc[mixed,'traj_id'])
        res['w1_selector'][feat]=ci(gain[p.live],p.loc[p.live,'traj_id'])
    # Mention-aligned chosen score preserves the same total number of mentions at a state.
    f=mat('target_mention');aligned=f[:,0]-f[:,1:].mean(1);p['mention_adv']=aligned
    nz=p.live&p.A_3456.ne(0)
    res['w1_sign']={'allLive_nonzero':auc(p.loc[nz,'A_3456']>0,aligned[nz]),
         'B3':auc(p.loc[p.cap,'A_3456']>0,aligned[p.cap]),
         'pearson_allLive':float(pearsonr(aligned[p.live],p.loc[p.live,'A_3456']).statistic)}
    # Siblings: raw coverage is different from independent support.
    res['siblings']={}
    for name,x in p.groupby('regime'):
        res['siblings'][name]={k:ci(x[k],x.traj_id) for k in ['g_cover','g_unique_traj','g_unique_actions','g_supported_arms',
            'g_supported_alts','g_chosen_supported','g_outcome_diverse','g_two_arms_two_traj','g_adm_match_frac','g_history_match_frac','g_exact_input_hits','g_singleton']}
    res['siblings_aucs']={}
    for label,x in [('B1',p),('B2',L)]:
        y=x.live if label=='B1' else x.het
        res['siblings_aucs'][label]={s:auc(y,x[s]) for s in ['g_task_success','g_succ_state','g_succ_alt_sd','g_unique_actions','g_supported_arms','g_outcome_diverse']}
        for name,features in [('task_only',['g_task_success','turn','n_admissible']),
                              ('frozenG',['g_cover','g_succ_state','g_succ_alt_sd','g_singleton','turn','n_admissible']),
                              ('support',['g_task_success','g_unique_traj','g_unique_actions','g_supported_arms','g_outcome_diverse','turn','n_admissible'])]:
            score=oof(x[features],y,x.traj_id);res['siblings_aucs'][label]['oof_'+name]=auc(y,score)
    res['siblings_B2_supported']={}
    for name,mask in [('all',L.g_cover>=0),('ge2_replay_arms',L.g_supported_arms>=2),
        ('ge4_replay_arms',L.g_supported_arms>=4),('outcome_diverse',L.g_outcome_diverse),
        ('two_arms_two_traj',L.g_two_arms_two_traj)]:
        x=L[mask];res['siblings_B2_supported'][name]={'n':len(x),'fork_rate':x.het.mean(),'auc_g_alt_sd':auc(x.het,x.g_succ_alt_sd)}
    # Within-observation/trajectory sibling action contrasts versus replay contrasts.
    gr=mat('g_rate');grc=gr-np.nanmean(gr,axis=1)[:,None];qc=q-np.nanmean(q,axis=1)[:,None]
    ok=np.isfinite(grc)&np.isfinite(qc)
    res['siblings_arm_alignment']={'n':int(ok.sum()),'pearson':float(pearsonr(grc[ok],qc[ok]).statistic)}
    # Adjacent point transitions require literal adjacent turns, not gaps from cf exclusions.
    transitions=[]
    for _,x in p.groupby('traj_id'):
        for i in range(len(x)-1):
            z,v=x.iloc[i],x.iloc[i+1]
            if v.turn==z.turn+1:transitions.append({'from':z.regime,'to':v.regime,'traj_id':z.traj_id,'turn':z.turn})
    td=pd.DataFrame(transitions);res['transitions']=td.groupby(['from','to']).size().unstack(fill_value=0).to_dict()
    res['dead_to_live']=ci((td.loc[td['from']=='Dead','to']!='Dead'),td.loc[td['from']=='Dead','traj_id'])
    # Raw continuation marker, a mediator diagnostic, never causal proof.
    cm=c[c.bank.isin([3,4,5,6])].groupby(KEY+['j']).agg(take3=('take_within3','mean'),take_ever=('take_ever','mean'),
                won=('won','mean'),steps=('steps','mean')).reset_index()
    aa=a.merge(cm,on=KEY+['j'],how='inner')
    res['continuation']={}
    for name,x in aa.groupby('target_mention'):
        res['continuation'][str(name)]={'n_arms':len(x),'take3':x.take3.mean(),'take_ever':x.take_ever.mean(),'q':x.q_3456.mean()}
    aa.to_parquet(OUT/'exploratory_arm_mechanisms.parquet',index=False)
    p.to_parquet(OUT/'exploratory_point_mechanisms.parquet',index=False)
    def conv(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    (OUT/'exploratory_results.json').write_text(json.dumps(res,indent=2,default=conv,allow_nan=True))
    print(json.dumps({k:res[k] for k in ['decomposition','gain_allLive_split','masked_gold_strata','w1_aucs','w1_alignment','w1_sign','siblings_aucs']},indent=2,default=conv),flush=True)

if __name__=='__main__':main()
