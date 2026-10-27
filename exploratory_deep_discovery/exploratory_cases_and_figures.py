"""Final targeted diagnostics and shareable figures; no new environment interaction."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exploratory_analyze import ci, ratio_ci, KEY

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def main():
    p=pd.read_parquet(OUT/'exploratory_point_mechanisms.parquet')
    a=pd.read_parquet(OUT/'exploratory_arms.parquet')
    c=pd.read_parquet(OUT/'exploratory_continuations.parquet')
    pairs=pd.read_parquet(OUT/'exploratory_mention_pairs.parquet')
    r=json.load(open(OUT/'exploratory_results.json'));m=json.load(open(OUT/'exploratory_mechanisms.json'))
    rr={};manifest={};roll={}
    def trajectory(tid):
        if tid not in roll:
            f=ROOT/f'data/rollout/{tid}.json';x=json.load(open(f))
            roll[tid]={'task':x['task'],'steps':[{k:s[k] for k in ['turn','obs_before','action','admissible','prompt_path']} for s in x['steps']]}
        return roll[tid]
    sf=pairs[pairs.same_facts].copy();flags=[]
    for z in sf.to_dict('records'):
        tr=trajectory(z['traj_id']);t=z['turn'];fp=ROOT/tr['steps'][t-1]['prompt_path'];pr=fp.read_text()
        manifest[str(fp.relative_to(ROOT))]=hashlib.sha256(fp.read_bytes()).hexdigest()
        flags.append(dict(exact_obs_in_prompt=z['obs_hi'] in pr,
            exact_obs_in_history=any(z['obs_hi']==s['obs_before'] for s in tr['steps'][:t]),
            generic_target_response=z['obs_hi'].startswith("There's nothing special about")))
    sf=pd.concat([sf.reset_index(drop=True),pd.DataFrame(flags)],axis=1)
    sf.to_parquet(OUT/'exploratory_same_facts_details.parquet',index=False)
    rr['same_facts_refined']={}
    masks={'exact_response_already_in_prompt':sf.exact_obs_in_prompt,
        'exact_response_already_in_history':sf.exact_obs_in_history,
        'generic_response':sf.generic_target_response,
        'same_type_and_generic_response':sf.generic_target_response&sf.same_action_type,
        'same_type_and_exact_prompt_repeat':sf.exact_obs_in_prompt&sf.same_action_type}
    for name,mask in masks.items():
        x=sf[mask];pt=x.groupby(KEY)[['diff_12','diff_3456','diff_gold']].mean().reset_index()
        rr['same_facts_refined'][name]={'n_pairs':len(x),'n_points':len(pt),
            **{b:ci(pt['diff_'+b],pt.traj_id) for b in ['12','3456','gold']}}
    L=p[p.live];rr['g_effective_coverage']={name:ci(L[col]>0,L.traj_id) for name,col in
        [('observation','g_cover'),('observation_admissible','g_adm_hits'),('history','g_history_hits'),('exact_input','g_exact_input_hits')]}
    # Chosen contrast compared with every possible anchor, not an arbitrary 0% null.
    q1=a.pivot(index=KEY,columns='j',values='q_34').loc[p.set_index(KEY).index].to_numpy()
    q2=a.pivot(index=KEY,columns='j',values='q_56').loc[p.set_index(KEY).index].to_numpy()
    random_anchor=np.mean((q1-q1.mean(1)[:,None])*(q2-q2.mean(1)[:,None]),1)/4
    assert np.allclose(random_anchor[p.live],p.loc[p.live,'C_cross']/4)
    rr['geometry']={'chosen_share':ratio_ci(L.C_chosen_cross,L.C_cross,L.traj_id),
        'random_uniform_anchor_expected_share':.25,
        'masked_share_of_bank12_net_gain_sum':float(p.loc[p.masked,'gain_12'].sum()/L.gain_12.sum()),
        'masked_share_of_allLive':float(p.masked.sum()/len(L))}
    # All-Live independent validation of concentration: 34 selects top states, 56 evaluates.
    gain_discovery=q1[:,1:].max(1)-q1[:,0]
    order=np.argsort(-gain_discovery[p.live],kind='stable');ll=L.iloc[order]
    rr['concentration']={}
    for f in [.1,.2,.5]:
        n=int(np.ceil(len(ll)*f));x=ll.iloc[:n]
        rr['concentration'][str(f)]={'n':n,'mean_gain56':ci(x.gain_56_from34,x.traj_id),
            'share_of_all_net_gain56':float(x.gain_56_from34.sum()/L.gain_56_from34.sum())}
    # Within-point action feedback vs subsequent usage, excluding the forced action.
    usage=c[c.bank.isin([3,4,5,6])].groupby(KEY+['j']).agg(take_next3=('take_after_first3','mean'),
        take_ever=('take_ever','mean'),repeat_next3=('repeat_first_next3','mean')).reset_index()
    aa=a.merge(usage,on=KEY+['j']);u=[]
    for k,x in aa.groupby(KEY):
        if x.target_mention.nunique()!=2:continue
        hi=x[x.target_mention];lo=x[~x.target_mention]
        u.append(dict(traj_id=k[0],turn=k[1],any_new=bool(x.new_to_history.any()),
            take_available_before=bool(x.take_available_before.any()),
            **{v:hi[v].mean()-lo[v].mean() for v in ['take_next3','take_ever','repeat_next3','q_3456']}))
    uu=pd.DataFrame(u);rr['usage']={}
    for name,x in [('all',uu),('new_history',uu[uu.any_new]),('known_history',uu[~uu.any_new])]:
        rr['usage'][name]={v:ci(x[v],x.traj_id) for v in ['take_next3','take_ever','repeat_next3','q_3456']}
    # Case selection uses discovery data; independent outcomes are displayed, never used to estimate selection benefit.
    mask=p[p.masked&p.gold_layer.notna()].copy()
    mask['abs_adv']=mask.A_3456.abs()
    chosen=[tuple(x) for x in mask.sort_values(['abs_adv','traj_id','turn']).head(3)[KEY].values]
    chosen += [('s017',29),('s097',38),('s025',39)]
    new=a[a.new_to_history&a.q_3456.notna()].sort_values('q_3456')
    if len(new):chosen.append(tuple(new.iloc[0][KEY]))
    cases=[]
    for tid,t in dict.fromkeys(chosen):
        t=int(t);tr=trajectory(tid);pt=p[(p.traj_id==tid)&(p.turn==t)].iloc[0]
        ar=a[(a.traj_id==tid)&(a.turn==t)].sort_values('j')
        cc=c[(c.traj_id==tid)&(c.turn==t)&c.bank.isin([3,4,5,6])]
        examples=[]
        for j,x in cc.groupby('j'):
            examples.append(dict(j=int(j),next_action_counts=x.next_action.value_counts().head(6).to_dict()))
        cases.append(dict(traj_id=tid,turn=t,task=tr['task'],regime=pt.regime,A=pt.A_3456,
            g_task_success=pt.g_task_success,g_observation_visits=pt.g_cover,g_distinct_actions=pt.g_unique_actions,
            g_exact_input_hits=pt.g_exact_input_hits,
            recent_steps=tr['steps'][max(0,t-4):t],
            arms=ar[['j','action','obs','facts_changed','new_to_prompt','new_to_history','q_12','q_3456','q_gold','s_gold','n_gold']].to_dict('records'),
            continuations=examples,raw_path=f'data/replay_b2/{tid}/t{t:02d}.json'))
    def conv(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x))
    (OUT/'exploratory_cases.json').write_text(json.dumps(cases,indent=2,default=conv))
    (OUT/'exploratory_followup_results.json').write_text(json.dumps(rr,indent=2,default=conv))
    (OUT/'exploratory_prompt_hashes.json').write_text(json.dumps(manifest,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),bbox_inches='tight');fig.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    ax[0].barh(['Outcome variance'],[1],color='#ddd',label='Continuation residual')
    share=r['decomposition']['controllable_share']['ratio'];ax[0].barh(['Outcome variance'],[share],color='#24758e',label='Between actions')
    ax[0].set_xlim(0,1);ax[0].set_xlabel('Share of outcome variance (Live)');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.2))
    chosen_share=rr['geometry']['chosen_share']['ratio'];ax[1].barh(['Action contrast variance'],[1],color='#dba24c',label='Among alternatives')
    ax[1].barh(['Action contrast variance'],[chosen_share],color='#24758e',label='Chosen vs mean alternatives')
    ax[1].axvline(.25,color='black',ls='--',label='Random anchor: 25%');ax[1].set_xlim(0,1);ax[1].set_xlabel('Share of between-action variance')
    ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.2));fig.tight_layout();save(fig,'01_variance_geometry')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    vals=[r['gain_allLive_split'],m['masked_detailed']['gainGold'],m['masked_small']['true'],m['cwgt']['small_A_gain']]
    labs=['All Live (34 select / 56 test)','Masked (independent gold)','Masked, small advantage (gold)','CWGT, small advantage (bank B)']
    for i,(v,lab) in enumerate(zip(vals,labs)):
        mean=v['mean'];lo,hi=v['ci95'];ax[0].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#24758e')
    ax[0].set_yticks(range(4),labs);ax[0].invert_yaxis();ax[0].axvline(0,color='gray',ls='--');ax[0].set_xlabel('Success gain of selected alternative')
    example=cases[0];qs=np.array([[z['q_3456'],z['q_gold']] for z in example['arms']]);im=ax[1].imshow(qs,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax[1].set_xticks([0,1],['Discovery','Gold']);ax[1].set_yticks(range(5),['chosen']+[f'alt {j}' for j in range(1,5)])
    ax[1].set_title(f"{example['traj_id']} / t{example['turn']}: A={example['A']:.3f}")
    for i in range(5):
        for j in range(2):ax[1].text(j,i,f'{qs[i,j]:.2f}',ha='center',va='center',color='white' if qs[i,j]>.6 else 'black')
    fig.colorbar(im,ax=ax[1],label='Success probability');fig.tight_layout();save(fig,'02_masked_opportunity')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    labs=['Current observation','Actual model prompt','Factual response','Any alternative response'];keys=['obs_target','prompt_target','factual_target_mention','alt_target_mention']
    ax[0].barh(labs,[r['w1_aucs'][k] for k in keys],color=['#b7c5cc']*3+['#24758e']);ax[0].axvline(.5,color='black',ls='--');ax[0].set_xlim(.45,.75);ax[0].set_xlabel('Fork AUROC: target mention only')
    vv=[m['matched_world']['same_facts']['gold'],m['matched_world']['same_facts_same_action_type']['gold'],rr['same_facts_refined']['same_type_and_exact_prompt_repeat']['gold']]
    labs=['Same facts','Same facts + action type','Same facts/type + repeated text']
    for i,v in enumerate(vv):
        if v['mean'] is None:continue
        mean=v['mean'];lo,hi=v['ci95'];ax[1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#24758e')
    ax[1].set_yticks(range(3),labs);ax[1].invert_yaxis();ax[1].axvline(0,color='gray',ls='--');ax[1].set_xlabel('Independent gold: mention minus no mention')
    fig.tight_layout();save(fig,'03_information_response')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    gg=rr['g_effective_coverage'];ax[0].bar(['Obs','+ admissible','+ history','+ time'],[v['mean'] for v in gg.values()],color='#24758e');ax[0].set_ylabel('Fraction of Live points with >=1 sibling match');ax[0].set_ylim(0,1)
    ax[1].bar(['Task-only sibling\nsuccess (raw)','Frozen G\n(learned)'],[r['siblings_aucs']['B1']['g_task_success'],.839],color=['#24758e','#dba24c']);ax[1].axhline(.5,ls='--',color='gray');ax[1].set_ylim(.45,.95);ax[1].set_ylabel('B1 AUROC');fig.tight_layout();save(fig,'04_siblings_support')
    print(json.dumps(rr,indent=2,default=conv),flush=True)

if __name__=='__main__':main()
