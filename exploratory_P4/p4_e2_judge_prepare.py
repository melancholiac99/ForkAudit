"""Prepare all E2 contexts without reading counterfactual outcome rates."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    inputs={}
    def read(p):
        b=(ROOT/p).read_bytes();inputs[p]=sha(b);return b
    read('exploratory_P4/strata.parquet');s=pd.read_parquet(OUT/'strata.parquet')
    s=s[s.live & s.stratum.isin(['I','U'])];assert len(s)==873
    templates={k:read(f'exploratory_P4/prompts/{k}.txt').decode() for k in ['base','log','feedback']}
    rows=0;audit=[];tr=None;last=None
    with (OUT/'e2_requests.jsonl').open('w') as f:
        for z in s.sort_values(['traj_id','turn']).to_dict('records'):
            tid,t=z['traj_id'],int(z['turn']);pid=f'{tid}_t{t:02d}'
            if tid!=last:tr=json.loads(read(f'data/rollout/{tid}.json'));last=tid
            st=tr['steps'][t-1];prompt=read(st['prompt_path']).decode()
            pk=json.loads(read(f'data/peek_alts/{tid}/t{t:02d}.json'))
            peeks=sorted(pk['peeks'],key=lambda p:(p['role']!='factual',p['alt_index'] or 0))
            assert len(peeks)==5 and peeks[0]['action']==st['action'] and peeks[0]['obs']==st['obs_after']
            assert pk['checks']['factual_obs_match'] and pk['checks']['refeed_div_total']==0 and pk['checks']['repeat_all_match']
            actions=[p['action'] for p in peeks]
            seed=int(sha(f'P4-v1:30102026:{pid}'.encode())[:16],16)
            order=np.random.default_rng(seed).permutation(5).tolist();shift=1+seed%3
            mapping=[0]+[1+(i-1+shift)%4 for i in range(1,5)]
            def feedback(m):return [dict(action=actions[i],response=peeks[m[i]]['obs']) for i in order]
            aligned=feedback(list(range(5)));shuffled=feedback(mapping)
            assert sorted(p['response'] for p in aligned)==sorted(p['response'] for p in shuffled)
            assert mapping[0]==0 and all(mapping[j]!=j for j in range(1,5))
            episode=dict(task=tr['task'],initial_observation=tr['initial_obs'],steps=[dict(turn=a['turn'],response=a['response'],executed_action=a['action'],observation_after=a['obs_after']) for a in tr['steps']],final_reward=int(tr['won']))
            base=templates['base']+f'\nMarked decision: turn {t}.\n<decision_time_prompt>\n'+prompt+'\n</decision_time_prompt>\nCandidates:\n'+json.dumps([actions[j] for j in order])
            texts={'S':base,'L':base+'\n'+templates['log']+json.dumps(episode,ensure_ascii=False),
                'F':base+'\n'+templates['feedback']+json.dumps(aligned,ensure_ascii=False),
                'Fx':base+'\n'+templates['feedback']+json.dumps(shuffled,ensure_ascii=False)}
            for condition,text in texts.items():
                row=dict(request_id=pid+'__'+condition,point_id=pid,traj_id=tid,turn=t,condition=condition,text=text,actions=actions)
                row['request_sha256']=sha(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())
                f.write(json.dumps(row,ensure_ascii=False)+'\n');rows+=1
            audit.append(dict(point_id=pid,display_order=order,alternative_mapping=mapping))
    read('exploratory_P4/p4_e2_judge_prepare.py')
    out=dict(points=len(s),contexts=rows,candidate_sequences_per_model=rows*5,models=2,
        requests_sha256=sha((OUT/'e2_requests.jsonl').read_bytes()),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=inputs)
    (OUT/'e2_preparation.json').write_text(json.dumps(out,indent=2)+'\n')
    (OUT/'e2_point_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print({k:v for k,v in out.items() if k!='inputs'})
if __name__=='__main__':main()
