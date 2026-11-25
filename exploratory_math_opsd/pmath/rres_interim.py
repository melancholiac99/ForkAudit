"""CPU-only boundary closeout; preserve the partial matrix and named denominators."""
import json,hashlib
from pathlib import Path
import numpy as np
from common import ROOT,json_save
from rres_chinese import OUT
from launch import validate_eval,sha

def close():
    base=Path('/mnt/data2/ForkAudit/math-opsd')
    boundary=json.loads((base/'rres_stop_boundary.json').read_text())
    benches=('aime24','aime25','hmmt25')
    complete=[b for b in benches if (ROOT/'run/evaluation/Rres@100'/b/'1001.receipt.json').is_file()]
    assert boundary['batch'] in complete and len(complete)<3
    rates={};markers={};marks=('**','---','$$')
    draws=np.random.default_rng(20261124).integers(30,size=(10000,3,30))
    for model in ('B','R@100','P@100','Rres@100'):
        common_counts=dict.fromkeys(marks,0)
        section={}
        for b in benches:
            p=ROOT/'run/evaluation'/model/b/'1001.json'
            if not p.with_suffix('.receipt.json').exists():continue
            rec=json.loads(p.with_suffix('.receipt.json').read_text())
            assert sha(p)==rec['sha256']
            row=json.loads(p.read_text())
            adapter=None if model=='B' else str(ROOT/'run'/model.split('@')[0]/'checkpoint-100')
            validate_eval(row,b,1001,adapter)
            texts=[g['full_generation'] for r in row['results'] for g in r['generations']]
            counts={m:sum(t.count(m) for t in texts) for m in marks}
            section[b]=dict(responses=len(texts),counts=counts,per_response={m:v/len(texts) for m,v in counts.items()})
            if b in complete:
                for m in marks:common_counts[m]+=counts[m]
            if model in ('R@100','Rres@100') and b in complete:
                scores=sorted((r['problem'],r['num_correct']/12) for r in row['results'])
                rates.setdefault(b,{})[model]=scores
        size=360*len(complete)
        section['common_completed']=dict(benchmarks=complete,responses=size,counts=common_counts,
                                      per_response={m:v/size for m,v in common_counts.items()})
        markers[model]=section
    result={}
    for i,b in enumerate(benches):
        if b not in complete:result[b]=dict(status='not run');continue
        r,rr=(rates[b][m] for m in ('R@100','Rres@100'))
        assert [q for q,_ in r]==[q for q,_ in rr]
        a=np.array([v for _,v in r])*100;c=np.array([v for _,v in rr])*100
        result[b]=dict(R=a.mean(),Rres=c.mean(),difference_pp=(c-a).mean(),
                    ci95_pp=np.percentile((c-a)[draws[:,i]].mean(1),[2.5,97.5]).tolist())
    json_save(OUT/'marker_counts.json',dict(method='literal non-overlapping per full generated response',models=markers))
    json_save(OUT/'evaluation_interim_stop.json',dict(reason='研究者在看到 AIME24 中期结果（Rres − R = −5.0 个百分点，95% 区间 [−10.0, 0.0]）后决定提前停止已登记的评测',boundary=boundary,benchmarks=result,three_benchmark_comparison=None))
if __name__=='__main__':close()
