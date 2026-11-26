import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forcing'))
from common_io import ROOT,SEED,read,save
import secondary_report
def verify_secondary():
    before=read(secondary_report.HERE/'secondary.json')
    # Run the registered aggregator against all 1,500 archived per-question files.
    secondary_report.run()
    after=read(secondary_report.HERE/'secondary.json')
    def check(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:check(a[k],b[k])
        else:np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)
    check(before,after)
    save(HERE/'secondary_verification.json',dict(questions=1500,positions=sum(after['groups'][g]['positions'] for g in ('clipped','unclipped')),
        atol=1e-14,rtol=1e-12,all_registered_metrics_reproduced=True))
    return after
def main():
    paths=sorted((ROOT/'run/cond-kl/clip-positions').glob('*.npz'),key=lambda x:int(x.stem))
    assert len(paths)==1500
    sums=np.array([np.load(p)['sums'] for p in paths])
    assert sums[:,0,0,0].sum()==1185192 and (sums[:,:,0]>=sums[:,:,1]-1e-9).all()
    n=len(paths);rng=np.random.default_rng(SEED)
    total=sums.sum(0); draws=[]
    for _ in range(100):
        w=rng.multinomial(n,np.full(n,1/n),100)
        x=(w@sums.reshape(n,-1)).reshape(-1,3,2,4)
        with np.errstate(divide='ignore',invalid='ignore'):draws.append(x[:,:,1]/x[:,:,0])
    ci=np.nanquantile(np.concatenate(draws),[.025,.975],axis=0);out={}
    lines=['# 句首／段首的截断预检','','表中百分比以各子集自己的总量为分母，区间按题整簇重抽。','','|子集|指标|边界位置的量|子集总量|占比 % [95% CI]|','|---|---|---:|---:|---:|']
    for j,g in enumerate(('all','stop05','content')):
        out[g]={}
        for k,name in enumerate(('positions','clipped_elements','removed_KL','preclip_KL_of_clipped_elements')):
            den,num=total[j,:,k]
            share=num/den if den else None
            out[g][name]=dict(boundary=num,total=den,share=share,ci95=ci[:,j,k].tolist() if den else None)
            lines.append(f'|{g}|{name}|{num:,.3f}|{den:,.3f}|'+(f'{100*share:.3f} [{100*ci[0,j,k]:.3f}, {100*ci[1,j,k]:.3f}]|' if den else 'NA|'))
    save(HERE/'results.json',out)
    secondary=verify_secondary()['groups']
    lines+=['','## 次选放大复核','','按原登记重新汇总，复算与存档数值一致。首选被截与全部元素被截是不同的计数单位。','',
            '|首选状态|位置数|模型|首选为老师第二名 %|老师第 2–5 名的模型/老师概率几何比|反思词质量 %|','|---|---:|---|---:|---:|---:|']
    for group in ['unclipped','clipped']:
        d=secondary[group]
        for arm in secondary_report.MODELS:
            def f(k,scale):
                val=d['metrics'][arm+'/'+k]
                return '%.3f [%.3f, %.3f]' % tuple(scale*x for x in (val['mean'],*val['ci95']))
            lines.append('|'+group+'|'+str(d['positions'])+'|'+arm+'|'+'|'.join(f(k,scale) for k,scale in [('top1_matches_teacher2',100),('rank2_5_geom_ratio',1),('reflection_total',100)])+'|')
    lines+=['','各反思词、模型间配对差、q(U) 及 1/q(U) 五分位见 [原登记全量表](../forcing/secondary.json)。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
