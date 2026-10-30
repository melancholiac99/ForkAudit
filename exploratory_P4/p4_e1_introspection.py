"""P4 E1 frozen OOF audit; no fits or new environment continuations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,subprocess,time
import numpy as np
import pandas as pd
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from evalcore import Boot,auroc,shuffle_null
KEY=['traj_id','turn'];SEED=30102026

def main():
    start=time.time();inputs={}
    def read(path):
        b=(ROOT/path).read_bytes();inputs[path]=hashlib.sha256(b).hexdigest();return b
    def pq(path):
        read(path);return pd.read_parquet(ROOT/path)
    protocol=json.loads(read('exploratory_P4/protocol.json'))
    audit=json.loads(read('exploratory_P4/strata_audit.json'));assert audit['passed']
    strata=pq('exploratory_P4/strata.parquet')
    labels=pq('data/labels_final.parquet');labels=labels[labels.live]
    d=labels.merge(strata[KEY+['stratum']],on=KEY,validate='one_to_one').merge(pq('eval/scores_B2.parquet'),on=KEY,validate='one_to_one')
    assert len(d)==1478 and np.array_equal(d.het,d.y)
    boot=Boot(d.traj_id,B=10000,seed=SEED)
    state={};distributions={}
    signals=protocol['introspective']+protocol['controls']
    for layer in ['I','U','N']:
        mask=d.stratum.eq(layer).to_numpy();g=d[mask];state[layer]={}
        for name in signals:
            scores=d[name].to_numpy(float);masked=np.where(mask,scores,np.nan)
            dist=boot.auroc_dist(d.het,masked);distributions[layer,name]=dist
            sh,ch=shuffle_null(g.het,g[name],g.traj_id,g.prog3,n=1000,seed=SEED+1)
            a=auroc(g.het,g[name]);hi=float(np.nanquantile(sh,.975))
            state[layer][name]=dict(auroc=a,ci95=Boot.ci(dist),shuffle_ci95=Boot.ci(sh),above_shuffle=bool(a>hi),
                shuffle_changed_fraction=ch,n=int(np.isfinite(g[name]).sum()),n_traj=int(g.traj_id.nunique()),invalid_bootstrap=int((~np.isfinite(dist)).sum()))
        print('state completed',layer,flush=True)
    differences={name:dict(diff=state['U'][name]['auroc']-state['I'][name]['auroc'],
        ci95=Boot.ci(distributions['U',name]-distributions['I',name])) for name in signals}
    scored=pq('data/scores_fp32.parquet').set_index(KEY+['action'])
    n=len(d);Q={k:np.full((n,5),np.nan) for k in ['12','34','56','gold']}
    pol=np.zeros((n,5));world=np.zeros((n,5))
    for i,z in enumerate(d.to_dict('records')):
        tid,t=z['traj_id'],int(z['turn']);banks={}
        acts=None
        for b in range(1,8):
            folder=('replay' if b==1 else f'replay_b{b-1}') if b<7 else 'replay_gold'
            path=f'data/{folder}/{tid}/t{t:02d}.json'
            if b==7 and not (ROOT/path).exists():continue
            r=json.loads(read(path));aa=[r['factual_action']]+r['alternatives']
            if acts is None:acts=aa
            assert aa==acts and len(set(aa))==5
            oo=[r['factual_outcomes']]+r['alt_outcomes'];assert all(len(v)==(20 if b==7 else 3) for v in oo)
            banks[b]=np.asarray(oo,float)
        for name,bs in [('12',[1,2]),('34',[3,4]),('56',[5,6]),('gold',[7])]:
            if all(b in banks for b in bs):Q[name][i]=np.concatenate([banks[b] for b in bs],axis=1).mean(1)
        lp=[float(scored.loc[(tid,t,a),'logp_sum']) for a in acts];assert np.isfinite(lp).all()
        pol[i]=softmax(lp)
        pk=json.loads(read(f'data/peek_alts/{tid}/t{t:02d}.json'))
        peeks=sorted(pk['peeks'],key=lambda p:(p['role']!='factual',p['alt_index'] or 0));assert [p['action'] for p in peeks]==acts
        # Match the registered instance-ID marker, not a broader target-word rule.
        import re
        # Target word comes from frozen W1; lazy-load once below before this loop in a future version is unnecessary.
        if i==0:
            w=pq('data/features_W1.parquet').set_index(KEY)
        rx=re.compile(rf"\b{re.escape(w.loc[(tid,t),'target_word'])} \d+\b")
        hits=np.array([bool(rx.search(p['obs'].lower())) for p in peeks],float)
        world[i]=hits/hits.sum() if hits.sum() else np.array([1.,0,0,0,0])
        if i%300==0:print('outcome audit',i,flush=True)
    q34,q56=Q['34'],Q['56'];adv=lambda q:q[:,0]-q[:,1:].mean(1)
    assert np.allclose((adv(q34)+adv(q56))/2,d.A_3456)
    total=((q34-q34.mean(1)[:,None])*(q56-q56.mean(1)[:,None])).mean(1)
    chosen=.16*adv(q34)*adv(q56)
    def stat(x,mask,den=None):
        x=np.asarray(x,float);mask=np.asarray(mask)&np.isfinite(x)
        if den is not None:mask &= np.isfinite(den)
        if not mask.any():return dict(n=0,estimate=None,ci95=None),np.full(10000,np.nan)
        v=np.bincount(boot.inv[mask],weights=x[mask],minlength=len(boot.t))
        dv=np.bincount(boot.inv[mask],weights=np.asarray(den)[mask] if den is not None else None,minlength=len(boot.t))
        with np.errstate(divide='ignore',invalid='ignore'):dist=(boot.W@v)/(boot.W@dv)
        estimate=float(x[mask].sum()/(np.asarray(den)[mask].sum() if den is not None else mask.sum()))
        return dict(n=int(mask.sum()),n_traj=int(d.loc[mask,'traj_id'].nunique()),estimate=estimate,ci95=Boot.ci(dist),
            invalid_bootstrap=int((~np.isfinite(dist)).sum())),dist
    action={};label={};ad={};ld={}
    for layer in ['I','U','N']:
        mask=d.stratum.eq(layer).to_numpy();fork=mask & d.het.to_numpy();action[layer]={};label[layer]={}
        for source in ['12','gold']:
            action[layer][source]={};q=Q[source]
            for name,p in [('policy',pol),('world',world)]:
                gu=(p*q).sum(1)-q.mean(1);gc=(p*q).sum(1)-q[:,0]
                action[layer][source][name]={}
                for metric,x in [('G_u',gu),('G_c',gc)]:
                    res,dist=stat(x,fork);action[layer][source][name][metric]=res;ad[layer,source,name,metric]=dist
        for name,x,subset,den in [('positive_sign',d.A_3456.gt(0).to_numpy(float),mask & d.credit_pivotal.to_numpy(),None),
            ('chosen_share_live',chosen,mask,total),('chosen_share_fork',chosen,fork,total)]:
            res,dist=stat(x,subset,den);label[layer][name]=res;ld[layer,name]=dist
    label_diffs={name:dict(diff=label['U'][name]['estimate']-label['I'][name]['estimate'],ci95=Boot.ci(ld['U',name]-ld['I',name])) for name in label['I']}
    pred={'P4-1':all(not state['I'][s]['above_shuffle'] for s in protocol['introspective']),
          'P4-2':any(differences[s]['ci95'][0]>0 for s in ['C_sel','B_multi']),
          'P4-3':action['I']['12']['policy']['G_u']['ci95'][0]<=0<=action['I']['12']['policy']['G_u']['ci95'][1] and action['U']['12']['policy']['G_u']['ci95'][0]>0,
          'P4-4_direction':all(label_diffs[s]['diff']>0 for s in ['positive_sign','chosen_share_live']),
          'positive_control':state['I']['W1pure_multi']['above_shuffle']}
    result=dict(script='exploratory_P4/p4_e1_introspection.py',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preregistration_commit='20f0976606ca01929bce67d3b850df1045fe8c4e',seed=SEED,bootstrap=10000,shuffle=1000,
        state=state,U_minus_I=differences,action=action,labels=label,label_U_minus_I=label_diffs,predictions=pred,
        verdict='PENDING_E2' if pred['positive_control'] else 'R-MIXED (E1 positive control failed)',
        seconds=time.time()-start,inputs=inputs)
    for path in ['exploratory_P4/p4_e1_introspection.py','scripts/evalcore.py','scripts/common.py']:read(path)
    (OUT/'e1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(pred,indent=2),flush=True)
if __name__=='__main__':main()
