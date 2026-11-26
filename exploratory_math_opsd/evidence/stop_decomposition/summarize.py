"""Position-weighted summaries with matched, whole-question bootstrap resampling."""
import argparse
import json
from pathlib import Path

import numpy as np

from probe import HERE, MODELS, OUT, PART1, PART2, ROOT, SCALE, TARGETS, WORK, indices, sha

TSETS=('stop05','stop50','actual_eos','all')
BSETS=('stop05','actual_eos','content','all')
PAIRS1=(('Rres','P'),('Rres','R'),('T0','R'))
PAIRS2=tuple((m,'B') for m in MODELS[1:])+(('Rres','P'),)


def summarize_group(matrix, mask):
    x=matrix[mask]
    return np.nansum(x,axis=0),np.isfinite(x).sum(axis=0)


def safe_percentile(values):
    good=values[np.isfinite(values)]
    return np.percentile(good,[2.5,97.5]).tolist() if len(good) else [None,None]


def bootstrap(records,groupnames,names,fields,pairs,weights):
    totals,counts=(np.stack(x) for x in zip(*records))
    q,s,m,f=totals.shape
    count=counts.sum(0)
    with np.errstate(invalid='ignore',divide='ignore'):
        avg=totals.sum(0)/count
        draws=(weights@totals.reshape(q,-1)/(weights@counts.reshape(q,-1))).reshape(-1,s,m,f)
    def metric(point,samples):
        return dict(mean=float(point) if np.isfinite(point) else None,ci95=safe_percentile(samples))
    out={}
    for j,subset in enumerate(groupnames):
        entries={}
        for k,name in enumerate(names):
            entries[name]={field:metric(avg[j,k,h],draws[:,j,k,h]) for h,field in enumerate(fields)}
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b)
            assert np.array_equal(counts[:,j,ia,:],counts[:,j,ib,:])
            entries[a+' minus '+b]={field:metric(avg[j,ia,h]-avg[j,ib,h],draws[:,j,ia,h]-draws[:,j,ib,h]) for h,field in enumerate(fields)}
        out[subset]=dict(positions=int(count[j,0,0]),eligible_questions=int((counts[:,j,0,0]>0).sum()),groups=entries)
        if 'delta_U' in fields:
            h=fields.index('delta_U')
            out[subset]['union_valid']=int(count[j,0,h])
            out[subset]['union_skipped']=int(count[j,0,0]-count[j,0,h])
            # Both contrasts are additive so that a small denominator cannot hide uncertainty.
            ir,ip=names.index('Rres'),names.index('P')
            jc=fields.index('J_C')
            outside=fields.index('outside')
            denom=fields.index('delta_on_U_positions')
            out[subset]['P_JC_minus_half_Rres_JC']=metric(avg[j,ip,jc]-.5*avg[j,ir,jc],draws[:,j,ip,jc]-.5*draws[:,j,ir,jc])
            out[subset]['Rres_outside_minus_half_matched_delta']=metric(avg[j,ir,outside]-.5*avg[j,ir,denom],draws[:,j,ir,outside]-.5*draws[:,j,ir,denom])
            out[subset]['Rres_outside_share']=None if not avg[j,ir,denom]>0 else float(avg[j,ir,outside]/avg[j,ir,denom])
    return out


def run(n, full_probe_u80=None):
    ids=indices(n)
    if full_probe_u80 is None:
        us=[]
        for i in indices(1500):
            with np.load(OUT/f'{i}.npz') as x: us.append(x['u'])
        threshold=float(np.quantile(np.concatenate(us),.8))
    else:
        threshold=float(full_probe_u80)
    first=[];second=[]; errors=[];positions=0
    for idx in ids:
        with np.load(OUT/f'{idx}.npz') as x:
            p=x['p']; end=x['eos'];u=x['u'];a=x['first'];b=x['second']
            positions+=len(p); errors.append(x['error'])
            assert np.isfinite(a[:,:,:5]).all() and np.isfinite(b).all()
            assert np.max(np.abs(a[:,:,0]-a[:,:,1]-a[:,:,4]))<1e-8
            for collection,array,masks in [
                (first,a,(p>=.05,p>=.5,end,np.ones(len(p),bool))),
                (second,b,(p>=.05,end,(p<.01)&(u>=threshold),np.ones(len(p),bool)))]:
                pairs=[summarize_group(array,mask) for mask in masks]
                collection.append(tuple(np.stack(x) for x in zip(*pairs)))
    assert n!=1500 or positions==1185192
    rng=np.random.default_rng(20261124)
    weights=np.stack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(10000)]).astype(float)
    t=bootstrap(first,TSETS,TARGETS,PART1,PAIRS1,weights)
    b=bootstrap(second,BSETS,MODELS,PART2,PAIRS2,weights)
    cfg=json.loads((ROOT/'models/Qwen3-1.7B/generation_config.json').read_text())
    manifest=json.loads((HERE.parent/'internalization/inputs.json').read_text())
    maxerr=np.max(errors,axis=0)
    assert max(maxerr)<1e-8
    result=dict(questions=n,positions=positions,seed=20261124,bootstrap_draws=10000,
            content_u80=threshold,temperature_masks=1.1,temperature_behavior_stop=1.,scale=SCALE,
            eos_ids=cfg['eos_token_id'],targets=t,behavior=b,max_abs_delta_error=dict(zip(TARGETS,maxerr)),
            inputs=manifest,fold_factors_sha256={str(k):sha(WORK/f'fit-{k}.pt') for k in range(4)},
            source_sha256={p.name:sha(p) for p in [ROOT/'source/opsd_trainer.py',ROOT/'source/data_collator.py',
                  HERE.parents[1]/'pmath/common.py',HERE.parents[1]/'pmath/rres_teacher.py']},
            timing=json.loads((OUT/'timing.json').read_text()))
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def cell(m):
    if m['mean'] is None:return 'NA'
    return f"{m['mean']:.5f} [{m['ci95'][0]:.5f}, {m['ci95'][1]:.5f}]"


def tables(result):
    lines=['# 停止概率分解与老师 top-1 一致率（事后探索）','',
           f"{result['questions']} 题，{result['positions']:,} 个归档位置。区间为整题配对 bootstrap 的 95% 百分位区间，10,000 次，种子 20261124。每个位置等权。",
           '', 'Δ 为相对初始学生的继续/停止 log odds 改变，正号表示更倾向继续。表内概率与一致率以 0–1 表示；熵、KL 和 log odds 均为自然对数单位。']
    t=result['targets'];b=result['behavior']
    stop=t['stop05']
    disagree=b['content']['groups']['Rres minus B']
    lines+=['','## 两个问题','',
        '下列口径在测速前登记。配对差的区间来自同一批题的相同重抽。','',
        '| 子集 | Rres Δ | Rres M | Rres J_C | Rres J_S | P Δ | P J_C |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for subset in ('stop05','actual_eos'):
        z=t[subset]['groups']
        lines.append('| '+subset+' | '+' | '.join(cell(z[g][f]) for g,f in
            [('Rres','delta'),('Rres','M'),('Rres','J_C'),('Rres','J_S'),('P','delta'),('P','J_C')])+' |')
    verdict='达到' if stop['P_JC_minus_half_Rres_JC']['mean']>=0 else '低于'
    outside=stop['Rres_outside_share']
    lines+=['',f"在停止概率 ≥5% 的位置，P 的 J_C {verdict} Rres 的一半。配对量 P J_C − ½ Rres J_C 为 {cell(stop['P_JC_minus_half_Rres_JC'])}。",
        '',f"在并集可算的同一批位置，Rres 的并集外占比为 {outside:.2%}，配对量 outside − ½ Δ 为 {cell(stop['Rres_outside_minus_half_matched_delta'])}。" if outside is not None else '并集内可算位置的 Δ 分母不为正，占比无定义。',
        '', '已训练 Rres 的停止概率差如下（单位为概率，0.01 对应 1 个百分点）。','',
        '| 子集 | Rres − B | Rres − P |','|---|---:|---:|']
    for subset in ('stop05','actual_eos'):
        z=b[subset]['groups']
        lines.append('| '+subset+' | '+cell(z['Rres minus B']['p_stop'])+' | '+cell(z['Rres minus P']['p_stop'])+' |')
    agreement=disagree['top1_agreement']
    lines+=['',f"在内容分歧子集，Rres−B 的 KL 差为 {cell(disagree['G'])} nats，模型熵差为 {cell(disagree['entropy'])} nats。老师 top-1 一致率的配对增量为 {100*agreement['mean']:.2f} 个百分点，95% 区间 [{100*agreement['ci95'][0]:.2f}, {100*agreement['ci95'][1]:.2f}]。",
        '', '判读限定于归档固定前缀。停止概率指标描述已生成前缀上的下一步倾向，不能单独证明自由生成长度变化的因果。老师 top-1 一致率衡量首选 token 的区分能力。']
    if stop['P_JC_minus_half_Rres_JC']['mean']>=0:
        lines+=['', '按登记的一半门槛，继续侧 Jensen 项不足以解释 P 与 Rres 的长度差。']
    if outside is not None:
        lines.append('按相同位置计算，并集外项达到一半，触发构造副作用判据。' if outside>=.5
                     else '按相同位置计算，并集外项低于一半，未触发构造副作用判据。')
    if agreement['ci95'][0]>.01:
        lines.append('内容分歧位置的一致率提升区间整体超过 1 个百分点，“只学到形状”的预测不成立。')
    elif disagree['G']['ci95'][1]<0 and agreement['ci95'][1]<=.01:
        lines.append('内容分歧位置的 KL 降低，且 top-1 增量区间上界也未超过 1 个百分点。这与分布形状改善的预测相符；其他排序或细粒度区分能力仍未测量。')
    else:
        lines.append('结合上述配对区间，分别解读 KL 和 top-1 对齐；检验所及限于这一固定子集。')
    for section,subsets,fields in [('目标分解',t,PART1),('已训练模型',b,PART2)]:
        lines+=['','## '+section]
        for subset,item in subsets.items():
            lines+=['','### '+subset,'',f"位置数 {item['positions']:,}，覆盖 {item['eligible_questions']} 题。"]
            if 'union_valid' in item:
                lines.append(f"并集内可算 {item['union_valid']:,} 个，跳过 {item['union_skipped']:,} 个。最后三列使用同一批可算位置。")
            lines+=['','| 模型或配对差 | '+' | '.join(fields)+' |','|---|'+'---:|'*len(fields)]
            for name,metrics in item['groups'].items():
                lines.append('| '+name+' | '+' | '.join(cell(metrics[f]) for f in fields)+' |')
    lines+=['','## 实施核验','',
            f"按训练实现调用 projected_logits（学生 top-64）及 residual_logits（双方 top-32 并集），并调用 R/T0 训练分支使用的原模型 teacher forward。OOF λ={SCALE}。T0 使用 question_only_teacher collator。模型适配器通过 Trainer 恢复检查点所用的 PEFT load_adapter 加载；文件 SHA-256 和每个加载后的张量均逐项核对。实际 EOS 只计算归档中确实生成的末 token。两部分都以温度 1.1 的无适配器学生定义停止子集。全探针 u 的 80% 分位为 {result['content_u80']:.6f}。",
            '',f"四目标独立 log odds 恒等核验最大误差为 {max(result['max_abs_delta_error'].values()):.3g}。测速 {result['timing']['seconds']} 秒/100 题，四卡全量线性预算 {result['timing']['estimated_full_seconds']/60:.1f} 分钟。"]
    return lines


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-probe-u80',type=float,help='Previously measured full-probe threshold from the same training targets')
    result=run(json.loads((OUT/'timing.json').read_text())['selected_questions'],parser.parse_args().full_probe_u80)
    (HERE/'REPORT.md').write_text('\n'.join(tables(result))+'\n')
