"""Compute the three contrasts on the same resampled questions for all four arms."""
import json,re
import numpy as np
from train import ROOT,HERE,ARMS,sha,read,save
NAMES=('R-noclip',*ARMS,'R')
COEF=np.array([[-.5,.5,-.5,.5],[-.5,-.5,.5,.5],[1.,-1.,-1.,1.]])
CONTRASTS=('boundary_main','interior_main','interaction')
MATH=('accuracy_percent','mean_generated_tokens','truncated_percent','Wait_per_answer')
FIXED=('content_entropy','content_top1_agreement','content_KL','stop05_stop_probability')
def summarize(sums,den,metrics):
    # questions x arms x metrics; same multinomial weights in every cell
    n=len(sums);rng=np.random.default_rng(20261124)
    arrays=[]
    for i in range(101):
        w=np.ones((1,n)) if i==0 else rng.multinomial(n,np.full(n,1/n),100)
        m=((w@sums.reshape(n,-1))/(w@den.reshape(n,-1))).reshape(len(w),4,-1)
        arrays.append(np.concatenate([m,COEF@m],axis=1))
    v=np.concatenate(arrays);ci=np.quantile(v[1:],[.025,.975],axis=0)
    return {key:{name:dict(estimate=v[0,i,j],ci95=ci[:,i,j].tolist()) for j,name in enumerate(metrics)}
            for i,key in enumerate((*NAMES,*CONTRASTS))}
def math_data(budget,arms=NAMES,reducer=summarize):
    sums=[];ids=None
    for arm in arms:
        key=arm+'@100'
        d=ROOT/'run'/('math500-trajectory-k20' if budget==4096 else 'math500-long-seed1001-n2')/(key if budget==4096 else 'nonthinking-'+key)
        h=read(d/'receipt.json'); assert sha(d/'answers.json')==h['output_sha256' if budget==4096 else 'answers_sha256']
        data=read(d/'answers.json')
        assert data['max_new_tokens']==budget and not data['enable_thinking'] and data['seed']==1001
        rows=data['results'];ks=[q['problem'] for q in rows]
        assert len(ks)==len(set(ks))==500 and (ids is None or ks==ids)
        ids=ks;armdata=[]
        for q in rows:
            g=q['generations'];assert len(g)==(4 if budget==4096 else 2)
            armdata.append(np.mean([[100*v['correct'],v['generated_tokens'],100*(v['finish_reason']=='length'),
                len(re.findall(r'\bwait\b',v['full_generation'],re.I))] for v in g],axis=0))
        sums.append(armdata)
    v=np.transpose(sums,(1,0,2));return reducer(v,np.ones_like(v),MATH)
def fixed_data(arms=NAMES,reducer=summarize):
    sums=[];dens=[]
    work=ROOT/'run/cond-kl'
    cut=read(HERE.parents[1]/'stop_decomposition/results.json')['content_u80']
    for f in sorted((ROOT/'run/stop-decomposition-v1').glob('*.npz'),key=lambda p:int(p.stem)):
        with np.load(f) as z:
            content=(z['p']<.01)&(z['u']>=cut);stop=z['p']>=.05
            v=z['second'][:,2,:]
            ref=[v[content,4].sum(),v[content,2].sum(),v[content,3].sum(),v[stop,0].sum()]
        row=[]
        assert arms[-1]=='R'
        for arm in arms[:-1]:
            with np.load(work/arm/('fixed-'+f.name)) as z:
                k=3 if arm in ('R-noclip','R-cond') else 0
                assert z['counts'][k,1]==content.sum() and z['counts'][k,2]==stop.sum()
                a=z['sums'][k,:,1,:];row.append([a[1,4],a[1,2],a[1,3],a[2,0]])
        sums.append([*row,ref]);dens.append([[content.sum()]*3+[stop.sum()]]*len(arms))
    assert len(sums)==1500
    return reducer(np.array(sums),np.array(dens),FIXED)
def main():
    out={'fixed':fixed_data(),'math_4k':math_data(4096),'math_16k':math_data(16384)}
    save(HERE/'factorial.json',out);lines=['# 两类位置截断的因子效应','','准确率和截断率以百分数计，差以百分点计。固定探针一致率、停止概率为 0–1。区间为按题配对 bootstrap 95%。']
    for name,table in out.items():
        cols=list(next(iter(table.values())))
        lines+=['','## '+name,'','|组或效应|'+'|'.join(cols)+'|','|---|'+'---:|'*len(cols)]
        for row,x in table.items():
            lines.append('|'+row+'|'+'|'.join('%.5g [%.5g, %.5g]'%tuple([x[k]['estimate'],*x[k]['ci95']]) for k in cols)+'|')
    (HERE/'FACTORIAL.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
