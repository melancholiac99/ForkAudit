"""Compare the new arms to the archived original R@100 on exactly the same content positions."""
from report import ROOT, HERE, ARMS, read, schedule, estimate, render
import json
import numpy as np
prior=read(HERE.parent/'stop_decomposition/results.json')
files=sorted((ROOT/'cond-kl'/ARMS[0]).glob('fixed-*.npz'))
values=[]
for file in files:
    idx=int(file.stem.split('-')[-1])
    with np.load(ROOT/'stop-decomposition-v1'/f'{idx}.npz') as z:
        mask=(z['p']<.01)&(z['u']>=prior['content_u80'])
        old=z['second'][mask,2,:].sum(0)
    with np.load(file) as z:
        assert mask.sum()==z['counts'][-1,1]
        diff=z['sums'][-1,1,1,:5]-old
    values.append([mask.sum(),*diff])
a=np.array(values)
assert len(a)==1500
s=a.sum(0)
d=np.concatenate([w@a for w in schedule(len(a))])
out=read(HERE/'results.json')
out['fixed_content_noclip100_minus_R100']={k:estimate(s[j]/s[0],d[:,j]/d[:,0]) for j,k in enumerate(('p_stop','log_continue_stop','top1','KL','entropy'),1)}
(HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
render(out)
