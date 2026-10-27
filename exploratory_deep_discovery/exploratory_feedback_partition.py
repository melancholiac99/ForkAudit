"""How much independent action-value variation is aligned with a frozen feedback bit?

Uses the law of total variance as a descriptive partition, not causal mediation.
No fitting of the partition to outcomes. All findings remain exploratory.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
from exploratory_analyze import ci,ratio_ci,KEY

OUT=Path(__file__).resolve().parent

def components(q1,q2,f):
    n=f.sum(1);m=5-n;mix=(n>0)&(m>0)
    d1=(q1*f).sum(1)/np.maximum(n,1)-(q1*(1-f)).sum(1)/np.maximum(m,1)
    d2=(q2*f).sum(1)/np.maximum(n,1)-(q2*(1-f)).sum(1)/np.maximum(m,1)
    b=np.where(mix,n*m/25*d1*d2,0)
    total=np.mean((q1-q1.mean(1)[:,None])*(q2-q2.mean(1)[:,None]),1)
    random=np.where(mix,total/4,0)
    return total,b,total-b,random,mix

def main():
    p=pd.read_parquet(OUT/'exploratory_point_mechanisms.parquet').set_index(KEY).sort_index()
    p=p[p.live];a=pd.read_parquet(OUT/'exploratory_arms.parquet')
    def mat(col):return a.pivot(index=KEY,columns='j',values=col).loc[p.index].to_numpy(float)
    q1,q2=mat('q_34'),mat('q_56');tr=p.index.get_level_values(0);res={};rows=[]
    for feature in ['target_mention','target_action','new_to_history']:
        f=mat(feature);total,b,w,random,mix=components(q1,q2,f)
        assert np.allclose(total,b+w)
        # Verify random partition's dimension-matched expectation by exact enumeration.
        for j in np.flatnonzero(mix)[:30]:
            vals=[]
            for indices in itertools.combinations(range(5),int(f[j].sum())):
                z=np.zeros((1,5));z[0,list(indices)]=1
                vals.append(components(q1[j:j+1],q2[j:j+1],z)[1][0])
            assert np.allclose(np.mean(vals),random[j])
        res[feature]={'n':len(p),'n_mixed':int(mix.sum()),'total_C':ci(total,tr),
            'between_feedback_groups':ci(b,tr),'within_feedback_groups':ci(w,tr),
            'between_share':ratio_ci(b,total,tr),'random_partition_share':ratio_ci(random,total,tr),
            'excess_over_random':ci(b-random,tr),'excess_share':ratio_ci(b-random,total,tr)}
        for j,k in enumerate(p.index):rows.append(dict(traj_id=k[0],turn=int(k[1]),feature=feature,
            C=float(total[j]),between=float(b[j]),within=float(w[j]),random_between=float(random[j])))
    (OUT/'exploratory_feedback_partition.json').write_text(json.dumps(res,indent=2))
    pd.DataFrame(rows).to_csv(OUT/'exploratory_feedback_partition_points.csv',index=False)
    print(json.dumps(res,indent=2))

if __name__=='__main__':main()
