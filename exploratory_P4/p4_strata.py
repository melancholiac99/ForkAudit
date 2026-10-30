"""P4 E0: rebuild text-only strata; labels used solely for registered count audit."""
from pathlib import Path
import hashlib,json,re,subprocess
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent

def main():
    hashes={}
    def read(p):
        b=(ROOT/p).read_bytes();hashes[p]=hashlib.sha256(b).hexdigest();return b
    labels=pd.read_parquet(ROOT/'data/labels_final.parquet')
    read('data/labels_final.parquet')
    w=pd.read_parquet(ROOT/'data/features_W1.parquet').set_index(['traj_id','turn'])
    read('data/features_W1.parquet')
    rows=[]
    for tid, group in labels.groupby('traj_id',sort=True):
        tr=json.loads(read(f'data/rollout/{tid}.json'))
        for z in group.to_dict('records'):
            t=int(z['turn']);st=tr['steps'][t-1];assert st['turn']==t
            target=w.loc[(tid,t),'target_word']
            rx=re.compile(rf'\b{re.escape(target)} \d+\b')
            tokens=lambda s:set(rx.findall(s.lower()))
            prompt=read(st['prompt_path']).decode()
            hist='\n'.join(s['obs_before']+'\n'+'\n'.join(s['admissible']) for s in tr['steps'][:t])
            full=tokens(hist);pids=tokens(prompt)
            pk=json.loads(read(f'data/peek_alts/{tid}/t{t:02d}.json'))
            peeks=sorted(pk['peeks'],key=lambda q:(q['role']!='factual',q['alt_index'] or 0))
            assert len(peeks)==5 and peeks[0]['action']==st['action']
            ids=[tokens(p['obs']) for p in peeks]
            new=set.union(*ids)-full
            pt=bool(pids);mention=any(ids);nh=bool(new)
            layer='I' if not pt and nh else 'U' if pt and mention and not nh else 'N' if not pt and not mention else 'other'
            future=tokens('\n'.join(s['obs_after'] for s in tr['steps'][t-1:]))
            rows.append(dict(traj_id=tid,turn=t,stratum=layer,prompt_target=pt,any_target_mention=mention,
                any_new_to_history=nh,future_reveals=bool(new & future),new_ids=sorted(new),live=bool(z['live']),het=bool(z['het'])))
    df=pd.DataFrame(rows)
    live=df[df.live]
    counts={k:{'live':int(len(g)),'fork':int(g.het.sum())} for k,g in live.groupby('stratum')}
    expected={'I':{'live':299,'fork':184},'U':{'live':574,'fork':202},'N':{'live':502,'fork':65},'other':{'live':103,'fork':24}}
    old=pd.read_parquet(ROOT/'exploratory_deep_discovery/exploratory_points.parquet')
    read('exploratory_deep_discovery/exploratory_points.parquet')
    merged=df.merge(old,on=['traj_id','turn'],suffixes=('_new','_old'),validate='one_to_one')
    mismatch={k:int((merged[k+'_new']!=merged[k+'_old']).sum()) for k in ['prompt_target','any_target_mention','any_new_to_history']}
    passed=counts==expected and len(live)==1478 and not any(mismatch.values())
    audit=dict(passed=passed,counts=counts,expected=expected,pointwise_flag_mismatches=mismatch,
        script='exploratory_P4/p4_strata.py',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inputs=hashes)
    (OUT/'strata_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not passed:raise RuntimeError('E0 consistency failed; stop and record DECISIONS')
    df.to_parquet(OUT/'strata.parquet',index=False)
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
