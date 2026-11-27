"""Late half-dose and switch endpoints, using archived baseline generations."""
import os
import sys
import numpy as np
from endpoint200 import HERE, ROOT, collect, probe
from launch import evaluate, read, save, sha

ARMS={'half':'R-lambda05','switch':'R-switch'}


def contest(key):
    arm=ARMS[key]
    model=arm+'@200'
    refs=['R-lambda05@100'] if key=='half' else ['R-extension@200','R@100','R-noclip@100']
    rec=evaluate(ROOT,os.environ.copy(),model,'aime24',1001)
    assert rec['adapter_sha256']==read(ROOT/'run'/arm/'extended.json')['adapters']['200']
    rows={}; digests={}; ids=None
    for name in [model,*refs]:
        folder=ROOT/'run/evaluation'/name/'aime24'
        digest=sha(folder/'1001.json')
        assert digest==read(folder/'1001.receipt.json')['sha256']
        digests[name]=digest
        qs=read(folder/'1001.json')['results']
        problems=[q['problem'] for q in qs]
        assert len(set(problems))==len(qs)==30 and (ids is None or problems==ids)
        assert all(len(q['generations'])==12 for q in qs)
        ids=problems
        rows[name]=np.array([sum(s['correct'] for s in q['generations']) for q in qs])
    w=np.random.default_rng(20261124).multinomial(30,np.full(30,1/30),10000)
    diff={}
    for ref in refs:
        d=rows[model]-rows[ref]
        diff[ref]=dict(pp=float(d.sum()*100/360),ci95=np.quantile(w@d*100/360,[.025,.975]).tolist())
    if key=='half':
        lo,hi=diff[refs[0]]['ci95']
        verdict=dict(optimum_across_interventions=lo>0,training_duration=hi<0)
    else:
        lo,hi=diff['R@100']['ci95']
        verdict=dict(preserved=diff['R-extension@200']['ci95'][0]>0 and lo<=0<=hi,
                     below_r100=hi<0,requires_continued_deviation=None,
                     note='接近 R-noclip 的界限未冻结时，此判据保留。')
    save(HERE/f'late_{key}_aime.json',dict(model=model,receipt=rec,answer_hashes=digests,
        answers_per_model=360,correct={n:int(a.sum()) for n,a in rows.items()},paired=diff,verdict=verdict))


if __name__=='__main__':
    action,key=sys.argv[1:3]
    if action=='aime':contest(key)
    elif action=='probe':probe(int(sys.argv[3]),(ARMS[key],))
    elif action=='collect':collect((ARMS[key],),200,f'late_{key}_fixed.json')
