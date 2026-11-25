"""Position-weighted outside probability summaries and literal response marker counts."""
import hashlib,json,sys
from pathlib import Path
import torch
from common import ROOT,WORK,json_save
from rres_chinese import OUT
from token_kl import TEMPERATURES

def probabilities():
    blocks=[torch.load(WORK/'rres-outside-probability'/f'w{i}-{j}.pt',weights_only=True)
            for i in range(4) for j in range(0,375,5)]
    x=torch.cat([b['probabilities'] for b in blocks])
    prior=json.loads((OUT/'residual_token_kl.json').read_text())
    assert len(x)==prior['positions'] and sum(b['questions'] for b in blocks)==1500
    assert x.min()>=0 and x.max()<=1.000001
    # Reaggregate archived position-block totals on CPU; no new GPU scoring.
    outs=torch.zeros(2,dtype=torch.float64)
    totals=torch.zeros_like(outs)
    for path in sorted((WORK/'rres-gpu').glob('w*-*.pt')):
        block=torch.load(path,weights_only=True,map_location='cpu')
        outs+=block['outside'][:,0]
        totals+=block['mass'][:,0].sum(-1)
    assert (totals>0).all()
    out={}
    for t,temp in enumerate(TEMPERATURES):
        groups={}
        for a,name in enumerate(('R','Rres')):
            col=x[:,t,a].contiguous()
            q=torch.quantile(col,torch.tensor([.5,.99],dtype=col.dtype))
            groups[name]=dict(mean=col.mean().item(),median=q[0].item(),p99=q[1].item())
        groups['R_outside_positive_KL_share']=(outs[t]/totals[t]).item()
        assert abs(groups['R_outside_positive_KL_share']-prior['temperatures'][str(temp)]['R']['outside_C_share'])<1e-9
        out[str(temp)]=groups
    json_save(OUT/'outside_probability.json',dict(positions=len(x),questions=1500,
        folds='OOF',probability_source='CPU reconstructed teacher, archived candidate IDs',positive_KL_source='CPU aggregation of archived positive KL statistics',max_centered_label_error=max(b['max_label_error'] for b in blocks),temperatures=out))
def markers(models):
    marks=('**','---','$$')
    report={}
    for model in models:
        section={}
        sums=dict.fromkeys(marks,0)
        for bench in ('aime24','aime25','hmmt25'):
            p=ROOT/'run/evaluation'/model/bench/'1001.json'
            assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(p.with_suffix('.receipt.json').read_text())['sha256']
            texts=[g['full_generation'] for r in json.loads(p.read_text())['results'] for g in r['generations']]
            assert len(texts)==360
            counts={m:sum(s.count(m) for s in texts) for m in marks}
            section[bench]=dict(responses=len(texts),counts=counts,per_response={m:v/len(texts) for m,v in counts.items()})
            for m in marks:sums[m]+=counts[m]
        section['all']=dict(responses=1080,counts=sums,per_response={m:v/1080 for m,v in sums.items()})
        report[model]=section
    name='marker_counts.json' if 'Rres@100' in models else 'marker_counts_before_rres.json'
    json_save(OUT/name,dict(method='Literal non-overlapping occurrences in full generated text including reasoning; each response weighted equally.',models=report))
if __name__=='__main__':
    if sys.argv[1]=='probabilities':probabilities()
    else:markers(sys.argv[2:])
