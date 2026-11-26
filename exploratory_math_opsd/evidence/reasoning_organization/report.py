"""Paired percentile intervals for ratios of question-cluster totals."""
import json
import numpy as np
from analyze import ARMS,SETS,SEED,HERE,OUT

def rows(ds,arm):
    full={}
    for lane in range(4):
        s=json.loads((OUT/f'{ds}-{arm}-{lane}.json').read_text())
        for r in s['rows']:
            k=(r['question'],r['sample']);assert k not in full
            full[k]=r
    return full,s['sha']

def features(r,o,thinking):
    pair={}
    def add(k,x,n=1):pair[k]=(x,n)
    add('wait_count',r['wait']);add('native_tokens',r['native'])
    for k in ('near','null'):add('wait_'+k,r[k],r['pairs'])
    add('wait_excess',r['near']-r['null'],r['pairs'])
    den=sum(r[g] for g in ('more','other','end'))
    for g in ('more','other','end'):add('next_'+g,r[g],den)
    if thinking:
        for rule in ('explicit','boxed'):
            pre,post,has,wrong,wc,cw=r[rule];common=has*o[rule][2]
            for k,x in [('pre',pre),('post',post),('coverage',has),('wrong_right',wc),('right_wrong',cw)]:
                add(rule+'_'+k,x)
            add(rule+'_rescue',wc,wrong)
            add(rule+'_common_pre',pre*common,common)
            add(rule+'_common_post',post*common,common)
    return pair

def evaluate(ds):
    arms=[];hashes={}
    for a in ARMS:
        data,h=rows(ds,a);arms.append(data);hashes[a]=h
    assert set(arms[0])==set(arms[1])
    keys=sorted(arms[0]);qs=sorted({q for q,_ in keys});qindex={q:i for i,q in enumerate(qs)}
    assert len(qs)==(500 if ds in SETS[:2] else 30)
    assert len(keys)==len(qs)*(2 if ds in SETS[:2] else 12)
    tabs=[];totals={}
    for a in range(2):
        x=[features(arms[a][k],arms[1-a][k],ds!='nonthinking') for k in keys]
        cols=list(x[0]);z=np.zeros((len(qs),len(cols),2))
        np.add.at(z,[qindex[q] for q,_ in keys],[[r[v] for v in cols] for r in x])
        tabs.append(z)
        totals[ARMS[a]]={k:z[:,i,:].sum(axis=0).tolist() for i,k in enumerate(cols)}
    raw=np.stack(tabs,axis=1)
    rng=np.random.default_rng([SEED,SETS.index(ds)])
    weights=np.vstack([np.ones(len(qs)),rng.multinomial(len(qs),np.full(len(qs),1/len(qs)),10000)])
    v=(weights@raw.reshape(len(qs),-1)).reshape(-1,2,len(cols),2)
    with np.errstate(invalid='ignore',divide='ignore'):est=v[:,:,:,0]/v[:,:,:,1]
    return dict(questions=len(qs),responses_per_arm=len(keys),input_sha256=hashes,totals=totals),cols,est

def summarize(cols,est):
    out={}
    for a,name in enumerate((*ARMS,'R_minus_B')):
        x=est[:,a] if a<2 else est[:,1]-est[:,0]
        ci=np.nanquantile(x[1:],[.025,.975],axis=0)
        out[name]={k:[x[0,i],ci[0,i],ci[1,i]] for i,k in enumerate(cols)}
    return out
def main():
    result={};contest=[]
    for ds in SETS:
        meta,cols,est=evaluate(ds);result[ds]=dict(**meta,statistics=summarize(cols,est))
        if ds in SETS[2:]:contest.append(est)
    result['contest_equal_weight']=dict(statistics=summarize(cols,np.mean(contest,axis=0)))
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def fmt(ds,who,k,factor=1):
        v=np.array(result[ds]['statistics'][who][k])*factor
        return f'{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]'
    lines=['# 长推理的组织：归档回答的事后分析','','区间按题配对重抽 10,000 次。成串率以相邻 Wait 对为分母，验证转移以带显式验证标记的段为分母。','',
           '|数据集|B 实际减打乱（百分点）|R 实际减打乱（百分点）|R−B 成串超额（百分点）|R−B 验证后继续反思（百分点）|','|---|---:|---:|---:|---:|']
    for ds in result:
        lines.append('|'+ds+'|'+'|'.join(fmt(ds,arm,k,100) for arm,k in [('B','wait_excess'),('R@100','wait_excess'),('R_minus_B','wait_excess'),('R_minus_B','next_more')])+'|')
    for rule in ('explicit','boxed'):
        lines+=['',f'## 首次可判分答案（{rule}）','','|数据集|R−B 总长（原生 token）|R−B 首答案前（含答案）|R−B 首答案后|R−B 先错后对（百分点，全体回答）|','|---|---:|---:|---:|---:|']
        for ds in list(result)[1:]:
            lines.append('|'+ds+'|'+'|'.join(fmt(ds,'R_minus_B',k,f) for k,f in [('native_tokens',1),(rule+'_pre',1),(rule+'_post',1),(rule+'_wrong_right',100)])+'|')
    lines+=['','## Wait 与显式验证段的后续','','|数据集|模型|Wait 成串 %|打乱 %|继续反思 %|其他段 %|结束 %|','|---|---|---:|---:|---:|---:|---:|']
    for ds in result:
        for a in ARMS:
            lines.append('|'+ds+'|'+a+'|'+'|'.join(fmt(ds,a,k,100) for k in ('wait_near','wait_null','next_more','next_other','next_end'))+'|')
    lines+=['','## 首答案覆盖与转移（明确陈述或 boxed）','','|数据集|模型|首答案覆盖 %|先错后对 / 全体 %|改对 / 首答案错 %|先对后错 / 全体 %|','|---|---|---:|---:|---:|---:|']
    for ds in list(result)[1:]:
        for a in ARMS:
            lines.append('|'+ds+'|'+a+'|'+'|'.join(fmt(ds,a,'explicit_'+k,100) for k in ('coverage','wrong_right','rescue','right_wrong'))+'|')
    lines+=['','原始计数、覆盖率、两模型都检出首答案时的长度、先错后对条件率、段落继续／转入其他段／结束及分基准区间见 results.json。未检出的回答全部归入首答案前；首答案前后以重新分词长度计算。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
