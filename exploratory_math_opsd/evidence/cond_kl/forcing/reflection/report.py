import numpy as np
from settings import *
from fork import selected

def interval(point,draws):
    return dict(mean=float(point),ci95=np.quantile(draws,[.025,.975]).tolist())

def bootstrap(n):
    rng=np.random.default_rng(SEED)
    for _ in range(100):
        ix=rng.integers(n,size=(100,n))
        yield np.stack([np.bincount(i,minlength=n) for i in ix]).astype(float)

def targeting():
    labels=read(RAW/'labels.json');indices=sorted(map(int,labels))
    meta=read(HERE/'tokens.json')
    tids=meta['included_ids'];groups=[[tids.index(meta['encodings'][s][0]) for s in (word,' '+word)] for word in ('Wait','But','Hmm','Alternatively')]
    shape=(len(indices),2,3,2)
    weights=np.zeros(shape);sums=np.zeros(shape+(6,5));qweights=np.zeros(shape);qsums=np.zeros_like(sums)
    for i,idx in enumerate(indices):
        row=labels[str(idx)]
        if row['correct'] is None:continue
        good=int(row['correct']);types=np.array([b for _,b in row['boundaries']],dtype=str)
        with np.load(RAW/f'score-{idx}.npz') as z:p=z['p']
        metrics=np.stack([p.sum(-1),*[p[:,:,g].sum(-1) for g in groups]],axis=-1)
        for sensitivity in (0,1):
            if sensitivity and not(row['ended'] and row['has_box']):continue
            for j,mask in enumerate([np.ones(len(p),bool),types=='sentence',types=='paragraph']):
                if not mask.any():continue
                k=i,sensitivity,j,good
                weights[k]=mask.sum();sums[k]=metrics[mask].sum(0)
                qweights[k]=1;qsums[k]=metrics[mask].mean(0)
    out={}
    for key,weight,num in [('position',weights,sums),('question',qweights,qsums)]:
        def calc(w):
            den=(w@weight.reshape(len(indices),-1)).reshape(len(w),*weight.shape[1:])
            s=(w@num.reshape(len(indices),-1)).reshape(len(w),*num.shape[1:])
            av=s/den[...,None,None]
            wrong,correct=av[:,:,:,0],av[:,:,:,1]
            lr=np.log(wrong/correct)
            return np.stack((wrong,correct,lr,lr-lr[:,:,:,0:1],np.log(wrong/wrong[:,:,:,0:1]),np.log(correct/correct[:,:,:,0:1])),axis=-1)
        point=calc(np.ones((1,len(indices))))[0]
        # Keep draw precision while avoiding retaining both weight schemes at once.
        draws=np.concatenate([calc(w) for w in bootstrap(len(indices))])
        ci=np.quantile(draws,[.025,.975],axis=0)
        for h,subset in enumerate(('all','completed_boxed')):
            for j,bound in enumerate(('both','sentence','paragraph')):
                rows={}
                for m,model in enumerate(MODELS):
                    rows[model]={}
                    for f,word in enumerate(('combined','Wait','But','Hmm','Alternatively')):
                        rows[model][word]={name:dict(mean=point[h,j,m,f,k],ci95=ci[:,h,j,m,f,k].tolist()) for k,name in enumerate(('wrong','correct','log_wrong_over_correct','log_ratio_minus_B','log_wrong_over_B','log_correct_over_B'))}
                out[f'{key}/{subset}/{bound}']=dict(questions=qweights.sum(0)[h,j].tolist(),positions=weights.sum(0)[h,j].tolist(),models=rows)
    save(HERE/'targeting.json',out)

def forks():
    labels=read(RAW/'labels.json')
    sites=selected(labels);questions=sorted({i for i,_,_ in sites}); grouped={i:[] for i in questions}
    meta=read(HERE/'tokens.json')
    wait=[meta['included_ids'].index(meta['encodings'][s][0]) for s in (' Wait','Wait')]
    for i,pos,_ in sites:
        rec=read(RAW/f'fork-{i}-{pos}.json')
        d=np.array([[v['remaining'],v['correct'],v['cap']] for v in rec]).reshape(2,4,3).mean(1)
        k=[p for p,_ in labels[str(i)]['boundaries']].index(pos)
        with np.load(RAW/f'score-{i}.npz') as z:p=z['p'][k][:,wait].sum(1)
        grouped[i].append((p[2],p[0],d))
    vals=[];pair=[]
    for rows in grouped.values():
        q=np.zeros((3,3,3))
        for r,b,d in rows:
            for j,w in enumerate((1,r,b)):
                q[j,0]+=w;q[j,1]+=w*d[0];q[j,2]+=w*d[1]
        vals.append(q)
        gap=np.zeros(4)
        if len(rows)==2:
            lo,hi=sorted(rows,key=lambda x:x[0])
            if hi[0]>lo[0]:gap=np.r_[1,(hi[2][0]-hi[2][1])-(lo[2][0]-lo[2][1])]
        pair.append(gap)
    vals=np.array(vals);pair=np.array(pair);n=len(vals)
    def calc(w):
        q=(w@vals.reshape(n,-1)).reshape(len(w),3,3,3)
        a=q[:,:,1]/q[:,:,0];b=q[:,:,2]/q[:,:,0];effect=a-b
        h=w@pair
        return np.concatenate([a,b,effect,(effect[:,1]-effect[:,0])[:,None],(effect[:,1]-effect[:,2])[:,None],(h[:,1:]/h[:,:1])[:,None]],axis=1)
    point=calc(np.ones((1,n)))[0];draws=np.concatenate([calc(w) for w in bootstrap(n)])
    keys=[f'{arm}/{w}' for arm in ('Wait','nonWait','Wait_minus_nonWait') for w in ('equal','R_weighted','B_weighted')]+['R_weighted_effect_minus_equal','R_weighted_effect_minus_B_weighted','within_question_high_minus_low_R_propensity_effect']
    out=dict(sites=len(sites),questions=n,paired_sites_questions=int(pair[:,0].sum()),continuations=8*len(sites),metrics={key:{m:interval(point[k,j],draws[:,k,j]) for j,m in enumerate(('remaining_length','accuracy','at_cap'))} for k,key in enumerate(keys)})
    save(HERE/'forks.json',out)

def render():
    c=read(HERE/'coverage.json')
    lines=['# 何时开始反思，以及强制 Wait 的局部效应','','事后探索。B 的最终答案判对 '+str(c['correct'])+' 条、判错 '+str(c['wrong'])+' 条、缺参考 '+str(c['missing_reference'])+' 条。边界共 '+str(c['boundaries'])+' 个；已冻结 '+str(c['fork_sites'])+' 个句末分叉。']
    if (HERE/'cached_B.json').exists():
        b=read(HERE/'cached_B.json')
        for subset in ('all','completed_boxed'):
            r=b[subset]
            lines+=['',f"归档 B 隐藏状态的 CPU 核对（{subset}）：错组概率 {r['wrong']['mean']:.3%}，对组概率 {r['correct']['mean']:.3%}，对数比 {r['log_ratio']['mean']:+.3f}。"]
    if (HERE/'targeting.json').exists():
        v=read(HERE/'targeting.json')
        for subset in ('all','completed_boxed'):
            lines+=['',f'## {subset}，句末与段末合计','','| 模型 | B 答错时概率 | B 答对时概率 | 错/对 log 比 | 相对 B 的 log 比差 [95% CI] |','|---|---:|---:|---:|---:|']
            for model,r in v[f'position/{subset}/both']['models'].items():
                x=r['combined'];p=x['log_ratio_minus_B']
                lines.append(f"| {model} | {x['wrong']['mean']:.5%} | {x['correct']['mean']:.5%} | {x['log_wrong_over_correct']['mean']:+.3f} | {p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] |")
    if (HERE/'pilot.json').exists():
        v=read(HERE/'pilot.json');lines+=['',f"32 个位置、256 条分叉续写，{v.get('gpus',4)} 卡用时 {v['seconds']/60:.2f} 分钟（含引擎启动）。"]
    if (HERE/'forks.json').exists():
        x=read(HERE/'forks.json')
        lines+=['','## R 的局部干预','','| 统计量 | 剩余 token [95% CI] | 正确率（百分点）[95% CI] |','|---|---:|---:|']
        for key,g in x['metrics'].items():
            cells=[]
            for metric,scale in [('remaining_length',1),('accuracy',100)]:
                p=g[metric];cells.append(f"{p['mean']*scale:+.2f} [{p['ci95'][0]*scale:+.2f}, {p['ci95'][1]*scale:+.2f}]")
            lines.append('| '+key+' | '+' | '.join(cells)+' |')
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    if sys.argv[1]=='targeting':targeting()
    if sys.argv[1]=='forks':forks()
    render()
