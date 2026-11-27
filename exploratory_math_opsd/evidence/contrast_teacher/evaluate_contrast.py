"""Run the registered endpoints and compare archived rows on identical tasks."""
import os
import sys
import numpy as np
from contexts import HERE,ROOT,read,save,sha
from launch import evaluate
from math500_trajectory import prepare
import math500_long as long

ARM='R-contrast'
REFS=('B','R-noclip@100')
FAMILY=HERE.parent/'cond_kl/lambda_family'
sys.path.insert(0,str(FAMILY))


def table(values,den,names,metrics):
    n=len(values)
    means=[]; rng=np.random.default_rng(20261124)
    for i in range(101):
        w=np.ones((1,n)) if i==0 else rng.multinomial(n,np.full(n,1/n),100)
        v=((w@values.reshape(n,-1))/(w@den.reshape(n,-1))).reshape(len(w),len(names),-1)
        means.append(np.concatenate((v,v[:,:1]-v[:,1:]),axis=1))
    draws=np.concatenate(means)
    ci=np.quantile(draws[1:],[.025,.975],axis=0)
    return {key:{m:dict(mean=float(draws[0,i,j]),ci95=ci[:,i,j].tolist()) for j,m in enumerate(metrics)}
            for i,key in enumerate([*names,*[names[0]+' minus '+ref for ref in names[1:]]])}


def behavior(step,bench):
    from launch import call
    target=f'{ARM}@{step}'
    matched='R@100' if step==100 else 'R-extension@200'
    names=[target,matched,*REFS]
    if bench=='math500':
        folder=long.OUT
        rows,hashes=prepare()
        hashes[target]=read(HERE/'training.json')['adapters'][str(step)]
        long.evaluate_cell(False,target,16384,rows,hashes,call)
        if step==200 and not (folder/f'nonthinking-{matched}/receipt.json').exists():
            names[1]='R@100'  # R@200 nonthinking was cancelled separately.
    values=[]; ids=None; archives={}
    opening_counts=[]
    for name in names:
        if bench=='math500':
            d=long.OUT/f'nonthinking-{name}'
            archive=d/'answers.json'
            receipt=read(d/'receipt.json')
            assert sha(archive)==receipt['answers_sha256']
        else:
            receipt=evaluate(ROOT,os.environ.copy(),name,bench,1001)
            archive=ROOT/'run/evaluation'/name/bench/'1001.json'
            assert sha(archive)==receipt['sha256']
        qs=read(archive)['results']
        keys=[q['problem'] for q in qs]
        assert len(set(keys))==len(keys) and (ids is None or keys==ids)
        ids=keys
        if bench=='math500':
            sys.path.insert(0,str(FAMILY.parent/'answer_openings'))
            from openings import opening
            from dose15_cpu import classify
            def features(g):
                text=g['full_generation']
                return [100*opening(text,False)[1],100*any(classify(text)),g['generated_tokens'],100*(g['finish_reason']=='length')]
            opening_counts.append([np.mean([features(g) for g in q['generations']],axis=0) for q in qs])
        values.append([[100*np.mean([g['correct'] for g in q['generations']])] for q in qs])
        archives[name]=sha(archive)
    x=np.transpose(values,(1,0,2))
    save(HERE/f'{bench}_{step}.json',dict(accuracy=table(x,np.ones_like(x),names,['accuracy_pct']),sources=archives))
    if bench=='math500':
        sys.path.insert(0,str(FAMILY.parent/'answer_openings'))
        from openings import measure
        from dose15_cpu import tally
        report=measure(target,16384,False)
        rows=read(long.OUT/f'nonthinking-{target}/answers.json')['results']
        report['first_word_rule']=tally(g['full_generation'] for q in rows for g in q['generations'])
        x=np.transpose(opening_counts,(1,0,2))
        report['paired']=table(x,np.ones_like(x),names,['thinking_opening_pct','gibberish_pct','mean_tokens','truncated_pct'])
        save(HERE/f'openings_{step}.json',report)


def fixed(lane):
    sys.path.insert(0,str(FAMILY.parent))
    import measure
    measure.STEPS=(100,200)
    measure.score('fixed',ARM,lane)


def fixed_report(step):
    base=ROOT/'run/cond-kl'
    ref='R@100' if step==100 else 'R-extension@200'
    names=[f'{ARM}@{step}',ref,*REFS]; sums=[];counts=[]
    for f in sorted((ROOT/'run/stop-decomposition-v1').glob('*.npz')):
        with np.load(f) as z:
            # Use the exact archived cutpoint from the initial probe.
            cut=fixed_report.cut
            c=(z['p']<.01)&(z['u']>=cut);s=z['p']>=.05
            original=z['second'][:,2,:]
            r100=[original[c,4].sum(),original[c,2].sum(),original[c,3].sum(),original[s,0].sum()]
        def take(folder,idx,model=1):
            with np.load(folder/('fixed-'+f.name)) as z:
                assert list(z['counts'][idx,1:3])==[c.sum(),s.sum()]
                v=z['sums'][idx,:,model]
                return [v[1,4],v[1,2],v[1,3],v[2,0]]
        k=0 if step==100 else 1
        sums.append([take(base/ARM,k),r100 if step==100 else take(base/'endpoint200/R-extension',0),
                     take(base/ARM,k,0),take(base/'R-noclip',3)])
        counts.append([[c.sum()]*3+[s.sum()]]*4)
    assert len(sums)==1500
    save(HERE/f'fixed_{step}.json',table(np.array(sums),np.array(counts),names,['entropy','top1','KL','stop05']))


def gate():
    a,b=[read(HERE/f'aime24_{k}.json')['accuracy'] for k in (100,200)]
    key=ARM+'@100'
    ci=a[key+' minus R-noclip@100']['accuracy_pct']['ci95']
    passes=ci[0]>0 and b[ARM+'@200']['accuracy_pct']['mean']>=a[key]['accuracy_pct']['mean']
    save(HERE/'decision.json',dict(confirm=passes,sampled_advantage_followup=ci[0]<=0<=ci[1],interpretation_limit='正上下文来自参考解，负上下文来自 B 自生成回答。风格不同，对比信号不只反映正确性。'))
    neg=read(HERE/'negative_contexts.json')
    stats=read(HERE/'supervision_v2.json')
    supervised=sum(s['supervised'] for s in stats.values())
    eligible=sum(s['eligible'] for s in stats.values())
    mass=sum((s['clipped_target_mass_fraction'] or 0)*s['supervised'] for s in stats.values())
    assert supervised>0 and len(stats)==200
    lines=['# R-contrast 修订版', '',
           f"旧负样本 {neg['old_selected']} 条，其中 {neg['old_without_any_complete_box']} 条缺少完整非空 boxed。修订版 {neg['tasks']} 题中 {neg['no_negative']} 题无合格负例。",
           f"全程实际监督 {supervised}/{eligible} 个回答位置（{100*supervised/eligible:.2f}%）。每步平均 {supervised/len(stats):.1f} 个，每卡每微批平均 {supervised/(8*len(stats)):.1f} 个。修正触及截断区外的元素占监督位置目标质量的 {100*mass/supervised:.3f}%。",
           '', '逐步数据见 [supervision_v2.json](supervision_v2.json)。', '',
           '|检查点|AIME24 思考准确率|MATH-500 非思考 16k 准确率|','|---|---:|---:|']
    for step in (100,200):
        model=f'{ARM}@{step}'
        rates=[read(HERE/f'{dataset}_{step}.json')['accuracy'][model]['accuracy_pct']['mean'] for dataset in ('aime24','math500')]
        lines.append(f'|@{step}|{rates[0]:.2f}%|{rates[1]:.2f}%|')
    lines.extend(['',f'@100 减 R-noclip 的 AIME24 配对差区间为 [{ci[0]:.2f}, {ci[1]:.2f}] 个百分点。',
                  '达到三基准确认条件。' if passes else '未达到三基准确认条件。',
                  '正上下文来自参考解，负上下文来自 B 自己的回答。两者风格不同，对比信号同时可能反映正确性与风格差异。'])
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if passes:
        for dataset in ('aime25','hmmt25'):
            for step in (100,200):behavior(step,dataset)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='probe':fixed(int(sys.argv[2]))
    elif stage=='collect':
        fixed_report.cut=read(HERE.parent/'stop_decomposition/results.json')['content_u80']
        for step in (100,200):fixed_report(step)
    elif stage=='gate':gate()
    else:behavior(int(sys.argv[2]),stage)
