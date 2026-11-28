import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize import HERE,ARMS,BUDGET,load
S=json.loads((HERE/'summary.json').read_text());labels=['B','λ=0','λ=0.5','R (λ=1)','λ=1.5']
colors=['#667085','#007f83','#409dde','#7e46aa','#d0521d']
def finish(f,name):
    f.tight_layout();f.savefig(HERE/(name+'.png'),dpi=115,bbox_inches='tight');plt.close(f)
def legend(ax):ax.legend(fontsize=7,frameon=False)
for ds,o in S.items():
    f,axes=plt.subplots(2,3,figsize=(12.6,6.8));a,b,c,d,e,g=axes.flat
    bins=np.linspace(0,BUDGET[ds][-1],31); pos=np.arange(5)
    for j,(m,v) in enumerate(o.items()):
        ns=np.sort([r['tokens'] for r in load(ds,m)]);col=colors[j];lab=labels[j]
        a.hist(ns,bins=bins,weights=np.ones(len(ns))/len(ns),histtype='step',color=col,label=lab)
        b.step(ns,np.arange(1,len(ns)+1)/len(ns),where='post',color=col,label=lab)
        e.plot(BUDGET[ds],100*np.asarray(v['budget']),color=col,label=lab,marker='.')
        e.plot(BUDGET[ds],100*np.array(v['explicit']['by_budget']),color=col,ls='--',lw=1)
        p=v['explicit'];s=np.array([p['pre'],p['post'],p['no_hit']]);bt=0
        for k,vv in enumerate(s/s.sum()):
            g.bar(j,vv,bottom=bt,color=['#2a7791','#e6ad48','#aaa9af'][k],label=['before first correct','after first correct','never explicit correct'][k] if not j else None);bt+=vv
    for k in ('p10','p25','p50','p75','p90','p99'):
        c.plot(pos,[v['length'][k] for v in o.values()],label=k,marker='.')
    bt=np.zeros(5)
    for h,col in zip(['stop_right','stop_wrong','stop_missing','cap_right','cap_wrong','cap_missing'],['#277a83','#8b8c60','#b6c8cc','#6150a3','#de844d','#b83249']):
        z=np.array([v['strata'][h].get('tokens',0)/v['tokens'] for v in o.values()])
        d.bar(pos,z,bottom=bt,color=col,label=h);bt+=z
    for ax in (c,d,g):ax.set_xticks(pos,labels,rotation=15)
    for ax,title in zip(axes.flat,['Length histogram (share per bin)','Empirical CDF','Length quantiles (tokens)','Share of generated tokens','Official accuracy (solid), ever correct (dashed), %','Token budget by explicit answer timing']):ax.set_title(title,fontsize=10)
    for ax in (a,b,e):ax.set_xlabel('Generated tokens')
    for ax in (a,c,d,g):legend(ax)
    finish(f,ds+'_distribution')
    f,axes=plt.subplots(2,3,figsize=(12.6,6.8))
    for j,(m,v) in enumerate(o.items()):
        seg=v['segments'][1:];col=colors[j]
        for ax,k,den,factor in zip(axes.flat[:3],['cov8','rep_expression','role'],['n','expression','n'],[100,100,1000]):
            ax.plot(range(4),[z[k]*factor/max(1,z[den]) for z in seg],marker='.',color=col,label=labels[j])
        for ax,k in zip(axes.flat[3:],['cap','accuracy','tokens_per_correct']):
            ax.plot(range(3),[v['bands'][z][k]*(1 if k=='tokens_per_correct' else 100) if v['bands'][z][k] is not None else np.nan for z in ['hard','mid','easy']],marker='.',color=col)
    for ax,title in zip(axes.flat,['Repeated 8-gram coverage (%)','Repeated math expressions (%)','Role cues / 1k tokens','Cap rate (%)','Final accuracy (%)','Tokens per correct']):
        ax.set_title(title,fontsize=10)
    for ax in axes.flat[:3]:ax.set_xticks(range(4),['Early','Middle','Late','Last 1024'])
    for ax in axes.flat[3:]:ax.set_xticks(range(3),['Hard','Mid','Easy'])
    legend(axes.flat[0]);finish(f,ds+'_content')
    f,axes=plt.subplots(1,2,figsize=(10,3.6))
    for j,(m,v) in enumerate(o.items()):
        for ax,rule in zip(axes,('explicit','boxed')):
            t=v[rule]['transitions'];bt=0
            for i,k in enumerate(['right_right','right_wrong','right_missing','wrong_right','wrong_wrong','wrong_missing','missing_right','missing_wrong','missing_missing']):
                z=t.get(k,0)/v['n'];ax.bar(j,z,bottom=bt,color=plt.cm.tab20(i/10),label=k if not j else None);bt+=z
            ax.set_xticks(pos,labels,rotation=15);ax.set_title(rule+' first → final')
    axes[-1].legend(loc='center left',bbox_to_anchor=(1,.5),fontsize=7);finish(f,ds+'_transitions')
