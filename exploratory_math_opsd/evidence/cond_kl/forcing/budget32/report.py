"""Summarize paired question-level outcomes from the second forcing round."""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from common_io import RAW, SEED, archived, read, save

def interval(x):
    """Bootstrap question means, keeping both draws of a question together."""
    if not len(x):
        return dict(mean=None, ci95=None)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(100):
        indices = rng.integers(len(x), size=(100, len(x)))
        estimates.extend(x[indices].mean(axis=1))
    return dict(mean=float(x.mean()), ci95=np.quantile(estimates,[.025,.975]).tolist())

def length_correlation(x, y):
    # Six sufficient statistics per question preserve the paired sample structure.
    x=x.astype(float); y=y.astype(float)
    sums=np.column_stack((np.full(len(x),x.shape[1]),x.sum(1),y.sum(1),
                          (x*x).sum(1),(y*y).sum(1),(x*y).sum(1)))
    def pearson(totals):
        n,sx,sy,sxx,syy,sxy=totals.T
        return (n*sxy-sx*sy)/np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
    rng=np.random.default_rng(SEED)
    draws=[]
    for _ in range(100):
        ix=rng.integers(len(x),size=(100,len(x)))
        draws.extend(pearson(sums[ix].sum(axis=1)))
    estimate=float(pearson(sums.sum(axis=0)))
    np.testing.assert_allclose(estimate,np.corrcoef(x.ravel(),y.ravel())[0,1])
    return dict(mean=estimate,ci95=np.quantile(draws,[.025,.975]).tolist())

def differences(forced, baseline, matched):
    delta = 100 * (forced.astype(float) - baseline.astype(float))
    both = matched.all(axis=1)
    any_match = matched.any(axis=1)
    partial = (delta * matched).sum(axis=1)[any_match] / matched.sum(axis=1)[any_match]
    return dict(all_questions=interval(delta.mean(axis=1)),
                both_answers_matched=interval(delta[both].mean(axis=1)),
                matched_answers_equal_question_weight=interval(partial))

def main():
    baselines = {a: archived(a) for a in ('B','T0@100','R@100')}
    rows = baselines['R@100']
    assert len(rows)==500 and all([q['problem'] for q in data]==[q['problem'] for q in rows] for data in baselines.values())
    def values(data, key):
        a = np.array([[s[key] for s in q] for q in data])
        assert a.shape==(500,2)
        return a
    target = values([q['generations'] for q in rows], 'generated_tokens')
    assert np.all(target>0)
    result = dict(seed=SEED, bootstrap_draws=10000, models={}, contrasts_pp={})
    lines=['# MATH-500 非思考预算强制（32 次追加）','',
           '|模型|正确率|平均长度|平均长度减 L|长度与 L 的相关|达到 0.9L|两条都达标的题数|',
           '|---|---:|---:|---:|---:|---:|---:|']
    paired_scores={}
    matched_per_arm=[]
    for arm in ('B','T0@100'):
        folder=RAW/'budget32'/arm
        data=[read(folder/f'{i}.json') for i in range(500)]
        assert np.array_equal(values(data,'target'),target)
        length=values(data,'tokens')
        count=values(data,'insertions')
        forced=values(data,'correct')
        assert np.all((count>=0)&(count<=32)) and np.all((length>0)&(length<=16384))
        match=10*length>=9*target
        matched_per_arm.append(match.all(axis=1))
        hist=np.bincount(count.ravel(),minlength=33)
        corr=length_correlation(length,target)
        result['models'][arm]=dict(accuracy=interval(forced.mean(axis=1)), mean_length=length.mean(),
            length_minus_target=interval((length-target).mean(axis=1)), length_correlation=corr,
            insertion_counts=hist.tolist(),insertion_fractions=(hist/1000).tolist(),
            matched_fraction=interval(match.mean(axis=1)),matched_answers=int(match.sum()),
            both_matched_questions=int(match.all(axis=1).sum()),any_matched_questions=int(match.any(axis=1).sum()))
        lines.append(f"|{arm}|{forced.mean():.1%}|{length.mean():.1f}|{(length-target).mean():+.1f}|{corr['mean']:.3f}|{match.mean():.1%}|{match.all(axis=1).sum()}/500|")
        for other in (arm,'R@100'):
            baseline=values([q['generations'] for q in baselines[other]],'correct')
            name=arm+'-强制32 − '+other
            paired_scores[name]=100*(forced.astype(float)-baseline.astype(float)).mean(axis=1)
            result['contrasts_pp'][name]=differences(forced,baseline,match)
    common=matched_per_arm[0]&matched_per_arm[1]
    result['common_matched_questions']=int(common.sum())
    for name,delta in paired_scores.items():
        result['contrasts_pp'][name]['both_models_both_answers_matched']=interval(delta[common])
    lines+=['','|正确率差（百分点，题级 95% 区间）|全部题|两条均达标|仅达标采样，题等权|两模型共同达标题|','|---|---:|---:|---:|---:|']
    for name,groups in result['contrasts_pp'].items():
        cells=['无样本' if v['mean'] is None else f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for v in groups.values()]
        lines+=['|'+name+'|'+'|'.join(cells)+'|']
    lines+=['',f'达标子集由强制后实际生成的长度决定，有选择偏差。各模型可选中不同题目；共同子集要求两模型各两次采样均达标，共 {common.sum()} 题。','',
            '|追加次数|B 回答数（占比）|T0 回答数（占比）|','|---:|---:|---:|']
    for j in range(33):
        counts=[v['insertion_counts'][j] for v in result['models'].values()]
        lines.append(f'|{j}|{counts[0]} ({counts[0]/1000:.1%})|{counts[1]} ({counts[1]/1000:.1%})|')
    from paired import supplement
    supplement(result,lines)
    save(OUT/'results.json',result)
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    main()
