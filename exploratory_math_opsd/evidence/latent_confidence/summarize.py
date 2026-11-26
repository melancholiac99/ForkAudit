"""Exact tied AUROC and Spearman with paired question-cluster bootstrap."""
import numpy as np
from numba import njit
from experiment import HERE, RAW, ROOT, BASE, REF, ARMS, SEED, read, save, digest
from transformers import AutoTokenizer

@njit
def auc_boot(sorted_scores,y,group,weights):
    out=np.empty(len(weights))
    for b,w in enumerate(weights):
        neg=0.;pos=0.;wins=0.;i=0
        while i<len(y):
            a=0.;z=0.;v=sorted_scores[i]
            while i<len(y) and sorted_scores[i]==v:
                if y[i]:a+=w[group[i]]
                else:z+=w[group[i]]
                i+=1
            wins+=a*(neg+0.5*z);neg+=z;pos+=a
        out[b]=wins/(neg*pos) if pos*neg else np.nan
    return out

@njit
def ranks(x,order,w):
    r=np.empty(len(x));i=0;cum=0.
    while i<len(x):
        j=i;n=0.
        while j<len(x) and x[order[i]]==x[order[j]]:
            n+=w[order[j]];j+=1
        r[order[i:j]]=cum+n/2;cum+=n;i=j
    return r

@njit
def corr_boot(x,y,g,weights):
    a=np.argsort(x);b=np.argsort(y);out=np.empty(len(weights))
    for i,row in enumerate(weights):
        w=row[g];m=w.sum()/2
        u=ranks(x,a,w)-m;v=ranks(y,b,w)-m
        out[i]=(w*u*v).sum()/np.sqrt((w*u*u).sum()*(w*v*v).sum())
    return out

def ci(values):
    a=np.nanpercentile(values[1:],[2.5,97.5],axis=0)
    return dict(estimate=values[0].tolist(),lo=a[0].tolist(),hi=a[1].tolist())

def aggregate(g,x,w):
    return w@np.bincount(g,weights=x,minlength=w.shape[1])

def metrics(out,arm,ids,w,kind):
    data=[];prompts=[];events=[];ctrl=[];counts=[];drops=[];lengths=[]
    for j,i in enumerate(ids):
        with np.load(out/(kind+'-'+arm)/f'{i}.npz') as a:
            for k in ([''] if kind=='fixed' else ['0','1']):
                p=a['p'+k];v=a['proj'+k];n=len(p);length=int(a['length'+k]);y=int(a['label'+k])
                data.extend(zip([j]*n,v,[y]*n,np.minimum(2,(3*(p-1))//max(length,1))))
                if kind=='fixed':
                    ask=a['ask'];ix=np.searchsorted(p,ask);assert np.array_equal(p[ix],ask)
                    prompts.extend(zip([j]*len(ix),a['ec'],a['mass'],v[ix],[y]*len(ix)))
                else:
                    counts.append(a['count'+k]);events.append(a['events'+k]);ctrl.append(a['controls'+k]);drops.append(a['drops'+k])
                    lengths.append(length-int(a['archived_length'+k]))
    g,x,y,b=np.array(data,dtype=float).T;g=g.astype(int)
    stats={}
    def mean(g,x):return aggregate(g,x,w)/aggregate(g,np.ones(len(x)),w)
    def auc(g,x,y):
        ix=np.where(y>=0)[0];ix=ix[np.argsort(x[ix])]
        return auc_boot(x[ix],y[ix],g[ix],w)
    stats['projection']=mean(g,x)
    for k,key in [(-1,'all'),(0,'early'),(1,'middle'),(2,'late')]:
        ix=np.ones(len(g),bool) if k==-1 else b==k
        stats['auc_'+key]=auc(g[ix],x[ix],y[ix])
    coverage=dict(questions=len(ids),boundaries=len(g))
    if kind=='fixed':
        j,ec,mass,proj,lab=np.array(prompts).T;j=j.astype(int)
        stats.update(ec=mean(j,ec),digit_mass=mean(j,mass),ec_auc=auc(j,ec,lab))
        if arm=='B':stats['validation_spearman']=corr_boot(proj,ec,j,w)
        coverage['prompted_boundaries']=len(j)
    else:
        count=np.reshape(counts,(-1,2)).sum(1)
        denom=w@count
        for label,series in [('event',events),('random',ctrl),('decline_frequency',drops)]:
            stats[label]=(w@np.reshape(series,(len(ids),2,-1)).sum(1))/denom[:,None]
        stats['event_minus_random']=stats['event']-stats['random']
        stats['decline_minus_random']=np.diff(-stats['decline_frequency'],axis=1)[:,0]
        coverage.update(events=int(count.sum()),answers=len(lengths),mean_retokenized_minus_archived_length=float(np.mean(lengths)),exact_archived_length_fraction=float(np.mean(np.array(lengths)==0)))
    return stats,coverage

def main():
    out=RAW/'full';m=read(RAW/'manifest.json')
    res=dict(fit=read(out/'fit.json'),seed=SEED,bootstrap=10000)
    pairs=[('R@100','B'),('T0@100','B'),('R-noclip@100','B'),('R-cond@100','B'),('R@100','T0@100'),('P@100','B')]
    tok=AutoTokenizer.from_pretrained(BASE)
    res['provenance']=dict(manifest_sha256=digest(RAW/'manifest.json'),adapter_sha256=read(REF/'audit.json')['adapters'],
        sources={arm:read(ROOT/'run/math500-long-seed1001-n2'/('nonthinking-'+arm)/'receipt.json')['answers_sha256'] for arm in ARMS},
        prompt_digit_encodings={s:tok.encode(s,add_special_tokens=False) for n in range(10) for s in (str(n),' '+str(n))},
        boundary_rule='Original reflection rule, including ASCII !', residual_stream='Transformer block output before final RMSNorm',
        trajectory_lags=list(range(-64,0)),direction_sha256=digest(out/'direction.npy'))
    for kind,ids in [('fixed',m['assess']),('math',list(range(500)))]:
        w=np.vstack([np.ones(len(ids),int),np.random.default_rng(SEED).multinomial(len(ids),np.full(len(ids),1/len(ids)),10000)]).astype(np.float64)
        by={};coverage={}
        for arm in ARMS:
            by[arm],coverage[arm]=metrics(out,arm,ids,w,kind)
        res[kind]=dict(coverage=coverage,models={arm:{k:ci(v) for k,v in d.items()} for arm,d in by.items()},
            paired={a+' minus '+b:{k:ci(by[a][k]-by[b][k]) for k in by[a].keys()&by[b].keys()} for a,b in pairs})
    save(HERE/'results.json',res)
    def show(a):return f"{a['estimate']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}]"
    v=res['fixed']['models']['B']
    lines=['# 隐状态置信度','', '事后探索，已见前序结果。区间按题整簇重抽 10,000 次。','',
           'B 的提示期望 AUROC 为 '+show(v['ec_auc'])+'，数字概率质量为 '+show(v['digit_mass'])+'。独立评估题上的自然投影与提示期望 Spearman 为 '+show(v['validation_spearman'])+'。',
           ('低于登记的 0.2 门槛，隐式指标仅描述，以显式提示为主。' if v['validation_spearman']['estimate']<.2 else '达到登记的 0.2 门槛。'),
           'MATH-500 的生成档案保存文本，按同一 tokenizer 重新分词。固定探针使用归档 token ID。']
    for kind in ('fixed','math'):
        lines+=['',f'## {kind}','','|模型或配对差|平均投影|AUROC（合计）|'+('提示期望|提示 AUROC|' if kind=='fixed' else '事件下降频率−随机|'),'|---|---:|---:|'+('---:|---:|' if kind=='fixed' else '---:|')]
        for table in ('models','paired'):
            for arm,row in res[kind][table].items():
                keys=['projection','auc_all']+(['ec','ec_auc'] if kind=='fixed' else ['decline_minus_random'])
                lines.append('|'+arm+'|'+'|'.join(show(row[key]) for key in keys)+'|')
    lines+=['','逐层验证相关、分段 AUROC、反思词及随机位置前 64 个 token 的逐点轨迹、下降频率和配对区间见 results.json。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
