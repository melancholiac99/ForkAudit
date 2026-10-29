from pathlib import Path
import sys,json,csv
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
sys.path.insert(0,str(BASE))
from analyze import cluster_ci


def main():
    scores=[json.loads(x) for x in (OUT/'scores.jsonl').read_text().splitlines()]
    assert len(scores)==24 and len({s['point_id'] for s in scores})==24
    old={x['point_id']:x for x in json.loads((BASE/'cases.json').read_text())}
    exposures={x['point_id']:x for x in json.loads((BASE/'hindsight_exposure_diagnostic.json').read_text())}
    rows=[]
    for s in scores:
        o=old[s['point_id']]
        p=np.array(s['candidate_prob'])
        assert o['cohort']=='development' and np.isclose(p.sum(),1)
        for condition,q in o['q'].items():
            q=np.array(q); vals={c:float(np.array(pp)@q) for c,pp in o['probabilities'].items()}
            value=float(p@q)
            rows.append(dict(point_id=o['point_id'],task_id=o['task_id'],regime=o['regime'],
                pool=condition,C=value,**vals,C_minus_H=value-vals['H'],C_minus_S=value-vals['S'],
                C_minus_A=value-vals['A'],chosen_mass=float(p[0]),
                hindsight_mentions_clue=exposures[o['point_id']]['instance_in_factual_observations']))
    with (OUT/'values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(2026102914);summary={}
    for pool in ['native','completed']:
        for group in ['all','Fork','Indifferent','Dead']:
            rr=[r for r in rows if r['pool']==pool and (group=='all' or r['regime']==group)]
            summary[pool+'/'+group]={k:cluster_ci(rr,k,rng) for k in ['C','H','S','A','C_minus_H','C_minus_S','C_minus_A','chosen_mass']}
    (OUT/'summary.json').write_text(json.dumps(dict(status='post-main exploratory explicit-clue diagnostic',groups=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
