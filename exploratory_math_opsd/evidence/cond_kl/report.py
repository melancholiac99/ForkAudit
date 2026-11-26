"""Paired question-cluster inference, with a fresh 10,000-draw schedule for each dataset."""
import json
import sys
from pathlib import Path
import re
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data/ForkAudit/math-opsd/run')
ARMS=('R-noclip','R-cond')
STEPS=(25,50,75,100)
SETS=('all','content','stop05','actual_eos')
METRICS=('p_stop','log_continue_stop','top1','KL','entropy','top1_outside_S')
def read(f):return json.loads(f.read_text())
def estimate(point, draws):
    return {'mean':float(point),'ci95':np.quantile(draws,[.025,.975]).tolist()}

def schedule(n):
    rng=np.random.default_rng(20261124)
    for _ in range(100):
        ix=rng.integers(n,size=(100,n))
        yield np.stack([np.bincount(row,minlength=n) for row in ix]).astype(float)

def stage_report(stage):
    sums=[];counts=[]
    for arm in ARMS:
        files=sorted((ROOT/'cond-kl'/arm).glob(f'{stage}-*.npz'),key=lambda f:int(f.stem.split('-')[-1]))
        assert len(files)==1500
        subset=[];n=[]
        for f in files:
            with np.load(f) as z:
                subset.append(z['sums']);n.append(z['counts'])
        sums.append(subset);counts.append(n)
    a=np.swapaxes(np.array(sums),0,1)
    c=np.swapaxes(np.array(counts),0,1)
    def means(w):
        num=(w@a.reshape(len(a),-1)).reshape(len(w),*a.shape[1:])
        den=(w@c.reshape(len(c),-1)).reshape(len(w),*c.shape[1:])
        return num/den[...,None,None]
    point=means(np.ones((1,len(a))))[0]
    draws=np.concatenate([means(w) for w in schedule(len(a))])
    out={}
    for i,arm in enumerate(ARMS):
        for j,step in enumerate(STEPS):
            for g,group in enumerate(SETS):
                rows={}
                vals=point[i,j,g];dist=draws[:,i,j,g]
                for m,metric in enumerate(METRICS):
                    rows[metric]={'actor':estimate(vals[1,m],dist[:,1,m]),
                        'B':estimate(vals[0,m],dist[:,0,m]),'actor_minus_B':estimate(vals[1,m]-vals[0,m],dist[:,1,m]-dist[:,0,m])}
                    if i==1:
                        rows[metric]['cond_minus_noclip']=estimate(vals[1,m]-point[0,j,g,1,m],dist[:,1,m]-draws[:,0,j,g,1,m])
                    if step==100:
                        rows[metric]['actor_100_minus_25']=estimate(vals[1,m]-point[i,0,g,1,m],dist[:,1,m]-draws[:,i,0,g,1,m])
                out[f'{arm}@{step}/{group}']=dict(positions=int(c[:,i,j,g].sum()),metrics=rows)
    return out

def training():
    out={}
    for arm in ARMS:
        folder=ROOT/arm
        rec=read(folder/'complete.json')
        rows={}
        for rank in range(4):
            for line in (folder/f'kl-rank-{rank}.jsonl').read_text().splitlines():
                v=json.loads(line);r=rows.setdefault(v['step'],np.zeros(6))
                r+=np.array([v['count'],*v['sums']])
        assert sorted(rows)==list(range(1,101))
        rec['steps']={k:dict(zip(('loss','original_KL','student_S','teacher_S','guard_fraction'),r[1:]/r[0])) for k,r in rows.items()}
        out[arm]=rec
    return out

def behavior(dataset):
    names=['B','R@100',*(a+'@100' for a in ARMS)]
    arr=[]
    for name in names:
        if dataset=='math500':
            f=ROOT/'math500-long-seed1001-n2'/('nonthinking-'+name)/'answers.json'
        else:
            f=ROOT/'evaluation'/name/dataset/'1001.json'
        v=read(f);questions=[r['problem'] for r in v['results']]
        if arr:assert questions==first
        else:first=questions
        items=[]
        for q in v['results']:
            gs=q['generations']
            items.append(np.mean([[g['correct'],g['generated_tokens'],g['finish_reason']=='length',len(re.findall(r'\bWait\b',g['full_generation'],re.I)),'<think>' in g['full_generation']] for g in gs],axis=0))
        arr.append(items)
    return np.array(arr).transpose(1,0,2)

def summarize_behavior(x, stratified=False):
    out={}
    names=['B','R@100',*(a+'@100' for a in ARMS)]
    points=x.mean(0)
    if stratified:
        assert len(x)==90
        rng=np.random.default_rng(20261124)
        draws=[]
        for _ in range(100):
            blocks=[x[b:b+30][rng.integers(30,size=(100,30))].mean(1) for b in (0,30,60)]
            draws.append(np.mean(blocks,axis=0))
        dist=np.concatenate(draws)
    else:
        dist=np.concatenate([(w@x.reshape(len(x),-1)).reshape(-1,*x.shape[1:])/len(x) for w in schedule(len(x))])
    for i,name in enumerate(names):
        row={}
        for j,m in enumerate(('accuracy','tokens','truncated','Wait','think')):
            row[m]={'estimate':estimate(points[i,j],dist[:,i,j])}
            for k in range(i):
                row[m]['minus_'+names[k]]=estimate(points[i,j]-points[k,j],dist[:,i,j]-dist[:,k,j])
        out[name]=row
    return out

def render(data):
    lines=['# Conditional and unclipped KL','', '事后探索，已见前序结果。单个训练种子 42。下列区间按题重抽 10,000 次，种子 20261124。完整分层估计和差值见 results.json。','',
           '| 预检子集 | 位置数 | 控制 token 组间 KL 占比 | 幅度组间 KL 占比 |','|---|---:|---:|---:|']
    for k,r in data['preflight'].items():
        def cell(x):return f"{100*x['share']:.2f}% [{100*x['ci95'][0]:.2f}, {100*x['ci95'][1]:.2f}]"
        lines.append(f"| {k} | {r['positions']} | {cell(r['control'])} | {cell(r['magnitude'])} |")
    for arm,r in data.get('training',{}).items():
        lines+=['',f"{arm} 完成 100 步，实测 {r['wall_seconds']/60:.2f} 分钟。"]
    for stage in ('fixed','own'):
        if stage not in data:continue
        lines+=['',f'## {stage} · 内容分歧位置','','| 模型 | 熵 | 原老师 KL | top-1 一致率 | S 外 top-1 一致率 |','|---|---:|---:|---:|']
        for arm in ARMS:
            for k in STEPS:
                r=data[stage][f'{arm}@{k}/content']['metrics']
                lines.append('| '+f'{arm}@{k} | '+' | '.join(f"{r[m]['actor']['mean']:.5f}" for m in ('entropy','KL','top1','top1_outside_S'))+' |')
    for name in ('math500','aime24','aime25','hmmt25','three_benchmarks'):
        if name not in data:continue
        lines+=['',f'## {name}','','| 模型 | 准确率 | 平均 token | 截断率 | Wait | 含 <think> 比例 |','|---|---:|---:|---:|---:|---:|']
        for model,r in data[name].items():
            lines.append('| '+model+' | '+' | '.join(f"{r[m]['estimate']['mean']:.5f}" for m in ('accuracy','tokens','truncated','Wait','think'))+' |')
    def interval(v,scale=1):
        avg=v['mean']*scale
        lo,hi=np.array(v['ci95'])*scale
        return f"{avg:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    if 'fixed_content_noclip100_minus_R100' in data:
        comp=data['fixed_content_noclip100_minus_R100']
        lines+=['','## P1 · 固定内容分歧位置','',f"@100 R-noclip 减 R，熵差 {interval(comp['entropy'])} nats，top-1 一致率差 {interval(comp['top1'],100)} 个百分点。"]
    if 'fixed' in data:
        lines+=['','## P2 · 停止概率','']
        for group in ('stop05','actual_eos'):
            rv=data['fixed'][f'R-cond@100/{group}']['metrics']['p_stop']
            other=data['fixed'][f'R-noclip@100/{group}']['metrics']['p_stop']
            lines.append(f"{group}，R-cond 减 R-noclip 为 {interval(rv['cond_minus_noclip'],100)} 个百分点，R-cond 减 B 为 {interval(rv['actor_minus_B'],100)}，R-noclip 减 B 为 {interval(other['actor_minus_B'],100)}。")
            lines.append('')
    if 'own' in data:
        lines+=['','## P3 · 自生成前缀','']
        for group in ('all','content'):
            v=data['own'][f'R-cond@100/{group}']['metrics']['top1_outside_S']['actor_100_minus_25']
            lines.append(f"{group}，R-cond 的 S 外 top-1 一致率 @100 减 @25 为 {interval(v,100)} 个百分点。")
            lines.append('')
    if 'math500' in data:
        r=data['math500']['R-cond@100']
        lines+=['',f"非思考 16k，R-cond 长度减 R-noclip 为 {interval(r['tokens']['minus_R-noclip@100'])} tokens，减 R 为 {interval(r['tokens']['minus_R@100'])}。含 <think> 的比例减 B 为 {interval(r['think']['minus_B'],100)} 个百分点。"]
    if 'three_benchmarks' in data:
        v=data['three_benchmarks']['R-cond@100']['accuracy']['minus_R@100']
        conclusion=('点估计达到无明显代价标准。' if v['mean']>=-.01 else '点估计低于无明显代价标准。')
        if v['mean']<-.02:conclusion+=' 准确率代价超过 2 个百分点。'
        lines+=['',f"三个基准等权，R-cond 减 R 的准确率差为 {interval(v,100)} 个百分点。{conclusion}"]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    stage=sys.argv[1]
    data=read(HERE/'results.json')
    if stage=='training':data[stage]=training()
    elif stage in ('own','fixed'):data[stage]=stage_report(stage)
    elif stage=='three_benchmarks':data[stage]=summarize_behavior(np.concatenate([behavior(d) for d in ('aime24','aime25','hmmt25')]),stratified=True)
    elif stage!='preflight':data[stage]=summarize_behavior(behavior(stage))
    (HERE/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    render(data)
