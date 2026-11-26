import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'clip_positions'))
import effects
from train import ROOT,read,save,sha
ARMS=('R-noclip','R-condA','R')
def reduce_paired(sums,den,metrics,names=ARMS,exponent=None):
    n=len(sums);rng=np.random.default_rng(20261124);draws=[]
    for b in range(101):
        w=np.ones((1,n)) if b==0 else rng.multinomial(n,np.full(n,1/n),100)
        v=((w@sums.reshape(n,-1))/(w@den.reshape(n,-1))).reshape(len(w),len(names),len(metrics))
        if exponent is not None:v[:,:,exponent]=np.exp(v[:,:,exponent])
        labels=list(names);cols=[v]
        for ref in ('R','R-noclip'):
            a=names.index('R-condA');j=names.index(ref)
            cols.append(v[:,a:a+1]-v[:,j:j+1]);labels.append('R-condA minus '+ref)
        draws.append(np.concatenate(cols,axis=1))
    vals=np.concatenate(draws);ci=np.quantile(vals[1:],[.025,.975],axis=0)
    return {label:{k:dict(estimate=vals[0,j,i],ci95=ci[:,j,i].tolist()) for i,k in enumerate(metrics)} for j,label in enumerate(labels)}
def second():
    rows=[];counts=[];teacher=[]
    for p in sorted((ROOT/'run/cond-kl/R-condA-secondary').glob('*.npz')):
        with np.load(p) as z:rows.append(z['sums']);counts.append(z['counts']);teacher.append(z['teacher'])
    assert len(rows)==1500
    x=np.stack(rows);ns=np.array(counts);out={}
    assert ns.sum()==1185192
    for j,k in enumerate(('unclipped','clipped')):
        den=np.broadcast_to(ns[:,j,None,None],x[:,j].shape)
        stats=reduce_paired(x[:,j],den,('top1_teacher_second','geometric_ratio_ranks2_5','reflection_probability','word_Wait','word_But','word_Alternatively','word_Hmm'),('B','R','R-noclip','R-cond','R-condA'),1)
        archived=read(HERE.parent/'forcing/secondary.json')['groups'][k]['metrics']
        stats['teacher']={m:archived['teacher/'+m] for m in ('reflection_total','word_Wait','word_But','word_Alternatively','word_Hmm')}
        np.testing.assert_allclose(np.sum(teacher,axis=0)[j]/ns[:,j].sum(),stats['teacher']['reflection_total']['mean'])
        out[k]=stats
    return out
def first_scores(budget):
    sys.path.insert(0,str(HERE.parents[1]/'reasoning_organization'))
    import analyze as org
    org.init();schools=[];order=None
    for arm in ARMS:
        folder=ROOT/'run'/('math500-trajectory-k20' if budget==4096 else 'math500-long-seed1001-n2')/(('' if budget==4096 else 'nonthinking-')+arm+'@100')
        receipt=read(folder/'receipt.json');assert sha(folder/'answers.json')==receipt['output_sha256' if budget==4096 else 'answers_sha256']
        rows=read(folder/'answers.json')['results'];ids=[q['problem'] for q in rows]
        assert order is None or ids==order
        order=ids;groups=[]
        for q in rows:
            data=[]
            for v in q['generations']:
                txt=v['full_generation'];item=org.first(txt,org.box(txt))
                a=int(bool(item) and org.grade(item[1],q['ground_truth']));b=int(v['correct'])
                data.append([100*a,100*(1-a)*b,100*a*(1-b),100*b,100*int(bool(item))])
            groups.append(np.mean(data,axis=0))
        schools.append(groups)
    x=np.transpose(schools,(1,0,2))
    np.testing.assert_allclose(x[:,:,0]+x[:,:,1]-x[:,:,2],x[:,:,3])
    return reduce_paired(x,np.ones_like(x),('first_accuracy_pp','wrong_to_correct_pp','correct_to_wrong_pp','final_accuracy_pp','first_coverage_pp'))
def main():
    results=dict(fixed=effects.fixed_data(ARMS,reduce_paired),secondary=second())
    for budget in (4096,16384):
        results[f'math_{budget}']=effects.math_data(budget,ARMS,reduce_paired)
        results[f'explicit_{budget}']=first_scores(budget)
    save(HERE/'results.json',results)
    lower,upper=results['math_16384']['R-condA minus R']['accuracy_percent']['ci95']
    verdict=('16k 与 R 的配对区间含零，满足登记第一项读数；等效和唯一机制仍未确立。' if lower<=0<=upper else
        '16k 相对 R 的区间上界低于零。与 R-noclip 的距离及区间见表，登记未定义接近的数值界。' if upper<0 else '16k 相对 R 的区间下界高于零，登记的两种预测均未覆盖这种结果。')
    lines=['# 动态分组条件 KL','',verdict]
    tables={k:v for k,v in results.items() if k!='secondary'}
    for group,stats in results['secondary'].items():
        tables['secondary_'+group]={k:v for k,v in stats.items() if k!='teacher'}
        lines.append(f"老师反思词总概率（{group}）：{stats['teacher']['reflection_total']['mean']:.6f}。")
    for title,table in tables.items():
        keys=list(next(iter(table.values())))
        lines+=['','## '+title,'','|模型或配对差|'+'|'.join(keys)+'|','|---|'+'---:|'*len(keys)]
        for arm,values in table.items():
            lines.append('|'+arm+'|'+'|'.join('%.5g [%.5g, %.5g]'%(values[k]['estimate'],*values[k]['ci95']) for k in keys)+'|')
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
