import numpy as np
from finish import ROOT,SEED,HERE,verify_secondary,save
def main():
    sums=[];weights=[]
    for f in sorted((ROOT/'run/cond-kl/clip-prefix-v2').glob('*.npz')):
        with np.load(f) as z:
            sums.append(z['sums']);weights.append([1,1/float(z['length']),1/float(z['micro_length'])])
    n=len(sums);assert n==1500
    x=np.array(sums);assert int(x[:,0,0,:,0].sum())==1185192
    assert np.allclose(x[:,0,:,:,0],x[:,1,:,:,0])
    answer={};lines=['# 前缀边界截断预检','','序列内平均为主表，百分比为各子集内份额。stop05 的位置行直接给出其边界比例。原始计数、未加权及探针四序列微批参照见 JSON。','',
        '|学生|子集|指标|边界 % [95% CI]|句内 % [95% CI]|','|---|---|---|---:|---:|']
    for w,mode in enumerate(('raw','sequence','probe_microbatch')):
        a=(x*np.array(weights)[:,w,None,None,None,None]);total=a.sum(0);den=total.sum(axis=2,keepdims=True)
        rng=np.random.default_rng(SEED);draws=[]
        with np.errstate(divide='ignore',invalid='ignore'):
            point=total/den
            for _ in range(100):
                b=(rng.multinomial(n,np.full(n,1/n),100)@a.reshape(n,-1)).reshape(-1,*total.shape)
                draws.append(b/b.sum(axis=3,keepdims=True))
        lo,hi=np.nanquantile(np.concatenate(draws),[.025,.975],axis=0)
        answer[mode]={}
        for i,student in enumerate(('B','R@50')):
            for j,group in enumerate(('all','stop05','content')):
                stats={};answer[mode][student+'/'+group]=stats
                for k,key in enumerate(('positions','clipped_elements','removed_KL','preclip_KL_on_A','gradient_L2')):
                    cells=[]
                    for h,kind in enumerate(('boundary','interior')):
                        idx=(i,j,h,k);v=point[idx]
                        stats.setdefault(key,{})[kind]=dict(total=float(total[idx]),share=None if not np.isfinite(v) else float(v),
                            ci95=[float(lo[idx]),float(hi[idx])] if np.isfinite(v) else None)
                        cells.append('NA' if not np.isfinite(v) else '%.3f [%.3f, %.3f]'%tuple(100*a[idx] for a in (point,lo,hi)))
                    if mode=='sequence':lines.append('|'+student+'|'+group+'|'+key+'|'+'|'.join(cells)+'|')
    save(HERE/'results.json',answer)
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
