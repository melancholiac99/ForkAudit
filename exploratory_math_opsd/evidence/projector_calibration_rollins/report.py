"""Question-clustered comparisons for projection calibration and sampled-prefix analysis."""
import json
import sys
import numpy as np
from measure import OUT,HERE,ENT,STEPS,CUTS,SCALES,d,sha,WORK

BINS=[f'D{i}' for i in range(1,11)]+['all']
METRICS=('KL','top1','entropy')


def resamples():
    rng=np.random.default_rng(20261124)
    for _ in range(100):
        draws=rng.integers(1500,size=(100,1500))
        yield np.stack([np.bincount(row,minlength=1500) for row in draws])


def sumq(weights,arrays):
    return (weights@arrays.reshape(1500,-1)).reshape(len(weights),*arrays.shape[1:])


def ci(point,draws):
    assert np.isfinite(point) and np.isfinite(draws).all()
    return dict(mean=float(point),ci95=np.percentile(draws,[2.5,97.5]).tolist())


def fmt(v,scale=1):
    return f"{v['mean']*scale:.2f} [{v['ci95'][0]*scale:.2f}, {v['ci95'][1]*scale:.2f}]"


def cal_report():
    stats=[];mass=[];counts=[]
    for i in sorted(d.split()):
        with np.load(OUT/f'cal-{i}.npz') as a:
            stats.append(a['stats']);mass.append(a['all_tv']);counts.append(a['counts'])
    def full(a,axis):
        a=np.stack(a)
        return np.concatenate([a,a.sum(axis=axis,keepdims=True)],axis=axis)
    stats=full(stats,2);mass=full(mass,2)
    n=np.sum(counts,0);n=list(map(int,n))+[int(n.sum())]
    assert n[-1]==1185192
    def summarize(s,t):
        w,y,yy,err=np.moveaxis(s,-1,0)
        fit=1-err/(yy-y*y/w)
        coverage=w[...,1]/t
        return np.concatenate([fit,coverage[...,None]],axis=-1)
    point=summarize(stats.sum(0),mass.sum(0))
    draws=np.concatenate([summarize(sumq(w,stats),sumq(w,mass)) for w in resamples()])
    result={}
    for j,b in enumerate(BINS):
        rec=dict(positions=n[j])
        for k,key in enumerate(('R2','TV_weighted_R2','union_TV_coverage')):
            values=point[:,j,k];sample=draws[:,:,j,k]
            rec[key]={t:ci(values[a],sample[:,a]) for a,t in enumerate(('R','T0'))}
            rec[key]['R minus T0']=ci(values[0]-values[1],sample[:,0]-sample[:,1])
        result[b]=rec
    old=json.loads(ENT.read_text())['bins']
    for b in BINS:assert abs(result[b]['R2']['R']['mean']-old[b]['OOF_R2']['value'])<1e-5
    return result


def trajectory():
    mats=[];n=[];limit=[]
    for i in sorted(d.split()):
        with np.load(OUT/f'score-{i}.npz') as a:
            mats.append(a['sums']);n.append(a['counts']);limit.append(a['limit'])
    mats=np.stack(mats);n=np.stack(n);limit=np.stack(limit)
    assert (n[:, :, 0].sum(0)==[1185192,236122]).all()
    def entries(s,counts):
        means=s/counts[...,None,:,None]
        fixed_base=means[...,0,0,:,:];fixed=means[...,0,1,:,:]
        own_base=means[...,1,0,:,:];own=means[...,1,1,:,:]
        return np.stack([fixed_base,fixed,own_base,own,own-fixed,
                         fixed-fixed_base,own-own_base,(own-own_base)-(fixed-fixed_base)],axis=-3)
    # Equal-question sensitivity: for each checkpoint and subset retain the questions
    # with positions under both prefix sources, so every paired contrast has common support.
    eligible=n.min(axis=2)>0
    valid=np.broadcast_to(eligible[:,:,None,:],n.shape).astype(float)
    per_question=np.divide(mats,n[:,:,:,None,:,None],out=np.zeros_like(mats),
                           where=n[:,:,:,None,:,None]>0)*valid[:,:,:,None,:,None]
    equal=entries(per_question.sum(0),valid.sum(0))
    equal_draws=np.concatenate([entries(sumq(w,per_question),sumq(w,valid)) for w in resamples()])
    point=entries(mats.sum(0),n.sum(0))
    draws=np.concatenate([entries(sumq(w,mats),sumq(w,n)) for w in resamples()])
    assert point.shape==(4,8,2,3)
    result={}
    prev=json.loads((HERE.parent/'decision_projector/results.json').read_text())
    for j,k in enumerate(STEPS):
        raw={}
        for h,subset in enumerate(('all','content')):
            vals={}
            for a,kind in enumerate(('B_fixed','R_fixed','B_own','R_own','own_minus_fixed','R_minus_B_fixed','R_minus_B_own','paired_shift')):
                vals[kind]={name:ci(point[j,a,h,f],draws[:,j,a,h,f]) for f,name in enumerate(METRICS)}
            equal_values={}
            for a,kind in enumerate(('B_fixed','R_fixed','B_own','R_own','own_minus_fixed','R_minus_B_fixed','R_minus_B_own','paired_shift')):
                equal_values[kind]={name:ci(equal[j,a,h,f],equal_draws[:,j,a,h,f]) for f,name in enumerate(METRICS)}
            raw[subset]=dict(metrics=vals,question_equal_weight=equal_values,paired_eligible_questions=int(eligible[:,j,h].sum()))
            for side,tag in enumerate(('fixed','own')):
                raw[subset][tag+'_positions']=int(n[:,j,side,h].sum())
                raw[subset][tag+'_questions']=int((n[:,j,side,h]>0).sum())
            if subset=='content':
                for f,key in enumerate(('KL','top1_agreement','entropy')):
                    assert abs(point[j,1,h,f]-prev['trajectory'][f'R@{k}'][key]['mean'])<.0002
        raw['mean_fixed_length']=float(n[:,j,0,0].mean());raw['fixed_fraction_at_limit']=float(limit[:,j,0].mean())
        raw['mean_own_length']=float(n[:,j,1,0].mean());raw['fraction_at_limit']=float(limit[:,j,1].mean())
        # Per-question resampling pairs checkpoint rows even when generated prefixes differ.
        if j:
            raw['adjacent_own_difference']={scope:{name:ci(point[j,3,h,f]-point[j-1,3,h,f],
                draws[:,j,3,h,f]-draws[:,j-1,3,h,f]) for f,name in enumerate(METRICS)}
                for h,scope in enumerate(('all','content'))}
        result[f'R@{k}']=raw
    return result


def main(only_cal=False):
    cal=cal_report()
    out=dict(questions=1500,positions=1185192,bootstrap_seed=20261124,draws=10000,
             entropy_cuts_nats=CUTS,scales=SCALES,calibration=cal)
    lines=['# 全域投影校准与自生成前缀（事后探索）','','## 全域校准','',
        '下表为解释比例（%）及按题聚类的 95% 区间。Δ 为 R−T0（百分点）。熵为固定无解答初始学生在温度 1.1 下的全词表熵。行为加权采用候选 token 在对应老师和学生全词表分布间的绝对概率变化，各老师使用各自的权重及拟合支持集。','',
        '| 熵组 | 位置数 | R | T0 | Δ | 加权 R | 加权 T0 | 加权 Δ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b,row in cal.items():
        lines.append('| '+b+' | '+str(row['positions'])+' | '+' | '.join(fmt(row[key][a],100) for key in ('R2','TV_weighted_R2') for a in ('R','T0','R minus T0'))+' |')
    if only_cal:
        (OUT/'calibration.json').write_text(json.dumps(out))
        print(lines[-1]);return
    roll=trajectory();out.update(rollins=roll,content_cutoff=d.CUT,
                 seed_rule='42 + archived dataset row ID',adapter_sha256={f'R@{k}':d.PRIOR['inputs']['adapters'][f'R@{k}'] for k in STEPS},
                 archives_sha256={f'roll-{lane}-{k}.pt':sha(OUT/f'roll-{lane}-{k}.pt') for lane in range(4) for k in STEPS},
                 fold_sha256={t:[sha(path/f'fit-{f}.pt') for f in range(4)] for t,path in [('R',WORK),('T0',d.OUT)]})
    lines+=['','全探针两套候选并集分别覆盖 R '+fmt(cal['all']['union_TV_coverage']['R'],100)+'% 和 T0 '+fmt(cal['all']['union_TV_coverage']['T0'],100)+'% 的全词表绝对概率变化。R² 小于零表示加权平方预测误差大于该组围绕加权均值的标签方差。T0 去掉了参考解答，仍保留思考提示。原标尺估计的是当前输入、目标、拟合器、收缩规则与支持集组合的题外可预测比例。两种老师标签的方差和行为权重均可能不同，二者 R² 的差值不能换算成特权信息份额。']
    for subset in ('all','content'):
        lines+=['',f'## 前缀比较：{subset}','','所有状态在相同的 1,500 题上各采一条回答。全位置或沿用原阈值的内容分歧子集。区间整题配对重抽，KL 与熵的单位为 nats/位置。一致率为 %，配对差为百分点。','',
                '| 检查点 | 前缀与模型 | 位置数 | 熵 | KL | 老师 top-1 一致率 |','|---|---|---:|---:|---:|---:|']
        for step,row in roll.items():
            v=row[subset]
            for name in v['metrics']:
                m=v['metrics'][name]
                count=v['fixed_positions'] if name in ('R_fixed','B_fixed','R_minus_B_fixed') else (v['own_positions'] if name in ('R_own','B_own','R_minus_B_own') else '—')
                lines.append(f"| {step} | {name} | {count} | {fmt(m['entropy'])} | {fmt(m['KL'])} | {fmt(m['top1'],100)} |")
    lines+=['','## 自生成轨迹的相邻检查点配对差','','按同一组题重抽；每个检查点使用自己的回答前缀。','',
            '|范围|后一步减前一步|熵差|KL 差|top-1 一致率差（百分点）|','|---|---|---:|---:|---:|']
    for j,k in enumerate(STEPS[1:]):
        for subset,v in roll[f'R@{k}']['adjacent_own_difference'].items():
            lines.append(f"|{subset}|@{k} − @{STEPS[j]}|{fmt(v['entropy'])}|{fmt(v['KL'])}|{fmt(v['top1'],100)}|")
    lines+=['','## 采样覆盖','','|状态|每题平均生成 token|达 1,024 token 比例|固定内容位置|自生成内容位置|自生成内容题数|','|---|---:|---:|---:|---:|---:|']
    for k,row in roll.items():
        c=row['content'];lines.append(f"|{k}|{row['mean_own_length']:.2f}|{row['fraction_at_limit']:.2%}|{c['fixed_positions']}|{c['own_positions']}|{c['own_questions']}|")
    lines+=['',f"固定 B 轨迹平均长度 {roll['R@25']['mean_fixed_length']:.2f} token，触顶比例 {roll['R@25']['fixed_fraction_at_limit']:.2%}。JSON 另含同一批有效题等权的敏感性结果：每题在指定子集内先取平均，然后题间等权。"]
    lines+=['','own_minus_fixed 是同一检查点在两套前缀上的指标差；paired_shift 为该变化减去基座 B 在同样两套前缀上的变化。固定和自生成前缀按题配对，token 位置不强行对齐。单条、最长 1,024 token 的采样只描述此设置下的前缀分布。']
    lines+=['','## 前缀分布的影响','',
            '下面的照表归纳限于原定义的内容分歧位置。各检查点的新回答都使用固定老师和固定的初始学生筛选阈值。']
    for key,label in [('R_fixed','固定'),('R_own','自生成')]:
        values=[roll[f'R@{k}']['content']['metrics'][key] for k in STEPS]
        best=STEPS[int(np.argmin([v['KL']['mean'] for v in values]))]
        first,last=values[0],values[-1]
        lines.append(f"{label}前缀上的老师 KL 在 @{best} 最低，top-1 一致率由 @25 的 {100*first['top1']['mean']:.2f}% 变为 @100 的 {100*last['top1']['mean']:.2f}%。")
    final=roll['R@100']['content']['metrics']
    lines.append('在 @100 的自生成内容位置上，R 相对该前缀上的 B：熵差 '+fmt(final['R_minus_B_own']['entropy'])+'，KL 差 '+fmt(final['R_minus_B_own']['KL'])+'，top-1 一致率差 '+fmt(final['R_minus_B_own']['top1'],100)+' 个百分点。')
    (HERE/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main('--calibration' in sys.argv)
