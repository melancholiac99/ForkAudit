"""Fixed-masks, whole-question paired bootstrap with position-weighted estimands."""
import json
import numpy as np
from diagnostic import HERE, OUT, PRE, PRIOR, CUT, GROUPS, split
from probe import SCALE

STATES=('B','R@25','R@50','R@75','R@100')
FIELDS=('KL','top1_agreement','entropy')
NAMES=('sign_agreement','R2','opposite_sign','truth_zero','prediction_zero','truth_positive','prediction_positive','mean_truth','mean_prediction')


def metrics(t):
    n,y,yy,se,ok,opp,zy,zp,yp,pp,sp=np.moveaxis(t,-1,0)
    return np.stack([ok/n,1-se/(yy-y*y/n),opp/n,zy/n,zp/n,yp/n,pp/n,y/n,sp/n],axis=-1)


def entry(point,samples):
    return {'mean':float(point),'ci95':np.percentile(samples,[2.5,97.5]).tolist()}


def interval(val,multiplier=1):
    a,b=val['ci95']
    return f"{val['mean']*multiplier:.3f} [{a*multiplier:.3f}, {b*multiplier:.3f}]"


def read():
    ids=sorted(split());mom=[];traj=[];cnt=[];stop=[];alln=0
    for idx in ids:
        mom.append(np.load(OUT/f'stats-{idx}.npy'))
        traj.append(np.load(OUT/f'traj-{idx}.npy'))
        with np.load(PRE/f'{idx}.npz') as z:
            cnt.append(((z['p']<.01)&(z['u']>=CUT)).sum());stop.append((z['p']>=.05).sum());alln+=len(z['p'])
    assert (len(ids),alln,sum(cnt),sum(stop))==(1500,1185192,236122,946)
    m=np.stack(mom);tr=np.stack(traj);cnt=np.array(cnt)
    rng=np.random.default_rng(20261124)
    proj=[];path=[]
    for _ in range(100):
        draws=rng.integers(1500,size=(100,1500))
        weights=np.stack([np.bincount(row,minlength=1500) for row in draws])
        proj.append(metrics((weights@m.reshape(1500,-1)).reshape(100,2,3,11)))
        path.append((weights@tr.reshape(1500,-1)).reshape(100,5,3)/(weights@cnt)[:,None,None])
    mp=metrics(m.sum(0));tp=tr.sum(0)/cnt.sum()
    mb=np.concatenate(proj);tb=np.concatenate(path)
    diag={}
    for j,name in enumerate(('R','T0')):
        diag[name]={}
        for k,g in enumerate(GROUPS):
            rec={n:entry(mp[j,k,h],mb[:,j,k,h]) for h,n in enumerate(NAMES)}
            rec['eligible_positions']=round(m[:,j,k,0].sum())
            diag[name][g]=rec
    trajectory={}
    for j,s in enumerate(STATES):
        trajectory[s]={f:entry(tp[j,k],tb[:,j,k]) for k,f in enumerate(FIELDS)}
        for i in sorted({0,j-1}) if j else []:
            trajectory[s+' minus '+STATES[i]]={f:entry(tp[j,k]-tp[i,k],tb[:,j,k]-tb[:,i,k]) for k,f in enumerate(FIELDS)}
    xx,xy=np.sum([json.loads((OUT/f'cal-{k}.json').read_text()) for k in range(4)],0)
    result=dict(questions=1500,positions=alln,decision_sites=sum(map(int,stop)),decision_questions=int(np.count_nonzero(stop)),
        content_sites=int(cnt.sum()),content_questions=int(np.count_nonzero(cnt)),u80=CUT,
        R_scale=SCALE,T0_raw_slope=xy/xx,T0_scale=float(np.clip(xy/xx,0,1)),
        bootstrap=dict(unit='question',draws=10000,seed=20261124),
        projector=diag,trajectory=trajectory,checkpoint_sha256=PRIOR['inputs']['adapters'])
    for arm in ('B','R'):
        for key,src in [('KL','G'),('entropy','entropy'),('top1_agreement','top1_agreement')]:
            target=PRIOR['behavior']['content']['groups'][arm][src]['mean']
            assert abs(trajectory[arm if arm=='B' else 'R@100'][key]['mean']-target)<1e-9
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 停止决策位置的题外投影与内容位置的 R 轨迹（事后探索）','',
        f"1,500 题，946 个停止决策位置（{result['decision_questions']} 题），236,122 个内容分歧位置（{result['content_questions']} 题）。所有区间按题配对重抽 10,000 次，种子 20261124。",
        '','## 题外投影器','',
        f"R 和 T0 标签均为老师减无解答非思考学生的中心化候选 logit 差。每个老师分别与学生构造 top-32 并集，各自按原 1,500 题四折、rank-8、Huber 和全位置拟合。R 使用归档因子，T0 使用只看题目的思考提示重拟合。全局 OOF 校准系数 R={SCALE:.8f}，T0={result['T0_scale']:.8f}（原斜率 {xy/xx:.8f}）。946 个评估位置未用作单独的训练或校准集。",
        '', '候选层面对比 λf 与真实中心化训练标签。各位置等权，每个位置的有效候选或 EOS 候选等权。决策层面对比老师和投影相对基座的全词表继续/停止 log odds 改变量 Δ；实际投影只修正学生 top-64 并保留训练精度。符号按负、零、正三类精确比较；R²=1−SSE/SST，SST 围绕评估子集加权均值计算。','',
        '| 老师 | 位置上的计算对象 | 有效位置 | 符号一致率 % | R² | 反号率 % |','|---|---|---:|---:|---:|---:|']
    for t,d in diag.items():
        for g,r in d.items():
            lines.append(f"| {t} | {g} | {r['eligible_positions']} | {interval(r['sign_agreement'],100)} | {interval(r['R2'])} | {interval(r['opposite_sign'],100)} |")
    lines+=['','## 固定内容分歧位置','',
        '沿用上次掩码：初始学生温度 1.1 的停止概率 <1%，且实际 R/P 目标的全词表 KL 位于全探针最高 20%。KL 方向为固定 R 老师相对模型；熵和 KL 均按温度 1.1 计算，单位 nats/位置。top-1 一致率的差以百分点计。B 和 R@100 与既有报告核对；中间检查点逐张量及 SHA-256 核验后重新前向。','',
        '| 状态或配对差 | 模型熵 | 老师 KL | top-1 一致率 % 或差（百分点） |','|---|---:|---:|---:|']
    for name,row in trajectory.items():
        lines.append(f"| {name} | {interval(row['entropy'])} | {interval(row['KL'])} | {interval(row['top1_agreement'],100)} |")
    lines+=['','## 判读','']
    for name in ('R','T0'):
        v=diag[name]['continue_log_odds']
        lines.append(f"{name} 决策层：老师提高继续倾向的位置占 {100*v['truth_positive']['mean']:.2f}%，投影器预测提高继续倾向的位置占 {100*v['prediction_positive']['mean']:.2f}%；符号相反的位置占 {100*v['opposite_sign']['mean']:.2f}%。")
    lines+=['','T0 保留思考提示、去掉参考解答。同样的失准可以定位到这批前缀上的投影流程，仍无法区分停止样本不足、Huber 目标、rank-8 容量或 top-64 应用口径的各自贡献。','',
            '熵反映分布平坦程度，top-1 反映是否选中了固定老师的首选 token。三项轨迹可以确定这些变化出现在哪个已保存的检查点；单凭轨迹无法分配 KL 改善中由变平造成的比例。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':read()
