"""Read-only extraction of frozen artifacts. Writes only this exploratory directory."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json, re, hashlib, subprocess, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
inputs = {}
def read(p):
    p = ROOT / p
    b = p.read_bytes()
    inputs[str(p.relative_to(ROOT))] = hashlib.sha256(b).hexdigest()
    return json.loads(b)
def parquet(p):
    p = ROOT / p
    inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_parquet(p)
def atype(a):
    return 'goto' if a.startswith('go to ') else a.split()[0]
def history(tr, i):
    return tuple((s['obs_before'], s['action']) for s in tr['steps'][max(0,i-2):i])
def slim(tr):
    """Discard generated token arrays immediately; the no-GPU instance has 2 GB RAM."""
    keep=['traj_id','seed','task_type','task','won']
    return {**{k:tr[k] for k in keep},'steps':[
        {k:s[k] for k in ['turn','obs_before','action','admissible','prompt_path']}
        for s in tr['steps']]}

def main():
    start=time.time()
    lab=parquet('data/labels_final.parquet')
    w=parquet('data/features_W1.parquet').set_index(['traj_id','turn'])
    b=parquet('data/features_B.parquet').set_index(['traj_id','turn'])
    g=parquet('data/features_G.parquet').set_index(['traj_id','turn'])
    gold=parquet('eval/gold_points_flags.parquet').set_index(['traj_id','turn'])
    roll={p.stem:slim(read(p.relative_to(ROOT))) for p in sorted((ROOT/'data/rollout').glob('*.json'))}
    sib=defaultdict(list)
    for p in sorted((ROOT/'data/siblings').glob('*.json')):
        tr=slim(read(p.relative_to(ROOT)));sib[tr['seed']].append(tr)
    si={}
    for seed,trs in sib.items():
        ix=defaultdict(list)
        for tr in trs:
            for i,s in enumerate(tr['steps']):
                ix[s['obs_before']].append(dict(traj=tr['traj_id'],turn=s['turn'],action=s['action'],
                    won=int(tr['won']),adm=tuple(sorted(s['admissible'])),hist=history(tr,i)))
        si[seed]=ix
    pts,arms,cont=[],[],[]
    print('Loaded slim trajectories',flush=True)
    for point_idx,z in enumerate(lab.to_dict('records')):
        tid,t=z['traj_id'],z['turn'];key=(tid,t);tr=roll[tid];st=tr['steps'][t-1]
        pk=read(f'data/peek_alts/{tid}/t{t:02d}.json')
        peeks=sorted(pk['peeks'],key=lambda q:(q['role']!='factual',q['alt_index'] or 0))
        rep=read(f'data/replay/{tid}/t{t:02d}.json')
        acts=[rep['factual_action']]+rep['alternatives']
        assert [q['action'] for q in peeks]==acts
        target=w.loc[key,'target_word'];rx=re.compile(rf'\b{re.escape(target)} \d+\b')
        tokens=lambda text:set(rx.findall(text.lower()))
        hist_obs='\n'.join(s['obs_before'] for s in tr['steps'][:t])
        hist_all='\n'.join(s['obs_before']+'\n'+'\n'.join(s['admissible']) for s in tr['steps'][:t])
        prompt=(ROOT/st['prompt_path']).read_text()
        full_ids,prompt_ids,obs_ids=tokens(hist_all),tokens(prompt),tokens(st['obs_before'])
        hits=si[z['seed']].get(st['obs_before'],[])
        byaction=defaultdict(list)
        for h in hits:byaction[h['action']].append(h)
        unique_tr={h['traj']:h['won'] for h in hits}
        supported=[a for a in acts if a in byaction]
        adm_hits=[h for h in hits if h['adm']==tuple(sorted(st['admissible']))]
        hist_hits=[h for h in adm_hits if h['hist']==history(tr,t-1)]
        peract=[len(set(h['traj'] for h in byaction.get(a,[]))) for a in acts]
        z.update(task_type=tr['task_type'],task=tr['task'],chosen=st['action'],chosen_type=atype(st['action']),
            target=target,masked=bool(z['het'] and not z['credit_pivotal']),
            obs_target=bool(obs_ids),prompt_target=bool(prompt_ids),ever_target=bool(full_ids),
            obs_target_ids=sorted(obs_ids),prompt_target_ids=sorted(prompt_ids),ever_target_ids=sorted(full_ids),
            g_unique_traj=len(unique_tr),g_unique_actions=len(byaction),g_supported_arms=len(supported),
            g_supported_alts=sum(a in byaction for a in acts[1:]),g_chosen_supported=acts[0] in byaction,
            g_min_arm_traj=min(peract),g_two_arms_two_traj=sum(n>=2 for n in peract)>=2,
            g_outcome_diverse=len(set(unique_tr.values()))==2,
            g_task_success=np.mean([x['won'] for x in sib[z['seed']]]),
            g_unique_success=np.mean(list(unique_tr.values())) if unique_tr else np.nan,
            g_adm_hits=len(adm_hits),g_history_hits=len(hist_hits),
            g_exact_input_hits=sum(h['turn']==t for h in hist_hits),
            g_adm_match_frac=len(adm_hits)/len(hits) if hits else np.nan,
            g_history_match_frac=len(hist_hits)/len(hits) if hits else np.nan)
        for name in ['g_cover','g_succ_state','g_succ_alt_sd','g_singleton']:z[name]=g.loc[key,name]
        for name in ['entropy_q','entropy_sampled','chosen_admissible','chosen_prob','chosen_freq','n_admissible']:z[name]=b.loc[key,name]
        z['target_revealed']=w.loc[key,'target_revealed']
        z['gold_layer']=gold.loc[key,'layer'] if key in gold.index else None
        ar=[]
        drafts=Counter(d['action'] for d in rep['drafts'][:32] if d['valid'] and d['admissible'])
        for j,(act,q) in enumerate(zip(acts,peeks)):
            ids=tokens(q['obs']);take_before=any(a.startswith('take '+target+' ') for a in st['admissible'])
            take_after=any(a.startswith('take '+target+' ') for a in q['admissible_after'])
            changes=q['facts_added']+q['facts_removed']
            row=dict(traj_id=tid,turn=t,j=j,action=act,action_type=atype(act),obs=q['obs'],
                target_mention=bool(ids),new_to_prompt=bool(ids-prompt_ids),new_to_history=bool(ids-full_ids),
                new_to_obs_history=bool(ids-tokens(hist_obs)),target_action=bool(tokens(act)),
                take_available_before=take_before,take_available_after=take_after,
                take_newly_available=take_after and not take_before,
                immediate_win=q['won'],facts_changed=bool(changes),n_fact_changes=len(changes),
                object_change=any(not f.startswith(('atlocation(','not_atlocation(')) for f in changes),
                draft_count=drafts[act],g_n_traj=peract[j],
                g_rate=np.mean([h['won'] for h in byaction[act]]) if byaction[act] else np.nan,
                g_rate_unique=np.mean(list({h['traj']:h['won'] for h in byaction[act]}.values())) if byaction[act] else np.nan)
            ar.append(row)
        for bank in range(1,8):
            folder=('replay' if bank==1 else f'replay_b{bank-1}') if bank<7 else 'replay_gold'
            p=ROOT/f'data/{folder}/{tid}/t{t:02d}.json'
            if not p.exists():continue
            r=rep if bank==1 else read(p.relative_to(ROOT))
            assert r['alternatives']==acts[1:] and r['factual_action']==acts[0]
            outcomes=[r['factual_outcomes']]+r['alt_outcomes']
            for j,oo in enumerate(outcomes):ar[j][f's{bank}']=sum(oo);ar[j][f'n{bank}']=len(oo)
            for c in r['continuations']:
                j=acts.index(c['first_action']);aa=c['actions']
                takes=[k for k,a in enumerate(aa) if a.startswith('take '+target+' ')]
                cont.append(dict(traj_id=tid,turn=t,j=j,bank=bank,won=c['won'],steps=len(aa),
                    first_take=min(takes) if takes else -1,take_within3=any(k<=3 for k in takes),
                    take_after_first3=any(1<=k<=3 for k in takes),take_ever=bool(takes),
                    next_action=aa[1] if len(aa)>1 else '',repeat_first_next3=acts[j] in aa[1:4]))
        for name in ['target_mention','new_to_prompt','new_to_history','take_available_after','take_newly_available','object_change']:
            z['any_'+name]=any(a[name] for a in ar);z['n_'+name]=sum(a[name] for a in ar)
            z['factual_'+name]=ar[0][name];z['alt_'+name]=any(a[name] for a in ar[1:])
        assert z['any_target_mention']==bool(z['target_revealed'])
        pts.append(z);arms.extend(ar)
        if point_idx%500==0:print('Extracted',point_idx+1,'points',flush=True)
    p,a,c=pd.DataFrame(pts),pd.DataFrame(arms),pd.DataFrame(cont)
    for name,banks in [('12',[1,2]),('34',[3,4]),('56',[5,6]),('3456',[3,4,5,6]),('gold',[7])]:
        a['s_'+name]=a[[f's{k}' for k in banks]].sum(axis=1,min_count=len(banks))
        a['n_'+name]=a[[f'n{k}' for k in banks]].sum(axis=1,min_count=len(banks))
        a['q_'+name]=a['s_'+name]/a['n_'+name]
    q=a.pivot(index=['traj_id','turn'],columns='j',values='q_3456')
    delta=q[0]-q[[1,2,3,4]].mean(axis=1)
    frozen=p.set_index(['traj_id','turn']).A_3456
    assert np.allclose(delta.loc[frozen.dropna().index],frozen.dropna())
    p.to_parquet(OUT/'exploratory_points.parquet',index=False)
    a.to_parquet(OUT/'exploratory_arms.parquet',index=False)
    c.to_parquet(OUT/'exploratory_continuations.parquet',index=False)
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs,
        n_points=len(p),n_arms=len(a),n_continuations=len(c),seconds=time.time()-start)
    (OUT/'exploratory_input_manifest.json').write_text(json.dumps(manifest,indent=2))
    print({k:v for k,v in manifest.items() if k!='inputs'},flush=True)

if __name__=='__main__':main()
