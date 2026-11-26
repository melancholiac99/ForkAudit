from common_io import *
def show(point,ci):
    return f'{point:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]'
def main():
    results={}
    for key in ('difficulty','pilot','budget','secondary'):
        p=HERE/f'{key}.json'
        if p.exists():results[key]=read(p)
    save(HERE/'results.json',results)
    lines=['# MATH-500 增写与次选放大','', '事后探索，沿用 1 个训练种子。区间按题重抽 10,000 次。','',
           '## 难度分层','','表中为层内增写最多的一半减最少的一半，两次采样交叉计分的准确率增益差（百分点）。','',
           '| 分层变量 | 层 | 题数 | 差值与 95% 区间 |','|---|---|---:|---:|']
    for k,r in results['difficulty'].items():
        for g,v in r['strata'].items():
            lines.append(f"| {k} | {g} | {v['n']} | {show(v['diff_pp'],v['ci95_pp'])} |")
        lines.append(f"| {k} | 题数加权 | 500 | {show(r['weighted_pp'],r['ci95_pp'])} |")
    lines+=['','官方 level 与 B 实际正确率两种分层的点估计方向不同。B 正确率由本次计分的两条 B 回答确定。']
    if 'pilot' in results:
        lines+=['','## 预算强制','',f"均匀取 20 题、每组 40 条，两组四卡串行实测 {results['pilot']['wall_seconds']/60:.2f} 分钟。"]
    if 'budget' in results:
        lines+=['','| 强制组 | 准确率 | 平均 token | 平均长度减 L | 与 L 的相关系数 |','|---|---:|---:|---:|---:|']
        for arm in ('B','R-cond@100'):
            v=results['budget'][arm]
            lines.append('| '+arm+' | '+' | '.join(f"{v[k]['mean']:.4f}" for k in ('accuracy','length','minus_target','correlation'))+' |')
            assert sum(v['insertions'])==1000
        for arm in ('B','R-cond@100'):
            lines+=['',f"{arm} 实际追加 0 至 8 次的回答数依次为 {results['budget'][arm]['insertions']}。"]
        for k,v in results['budget']['contrasts_pp'].items():lines+=['',f"{k}，{show(v['mean'],v['ci95'])} 个百分点。"]
    if 'secondary' in results:
        v=results['secondary']['groups']
        lines+=['','## 次选放大','','a 为模型首选老师第二名的比例，b 为第 2–5 名模型与老师概率比的几何均值，c 为指定单 token 反思词的总概率。','',
               '| 老师首选 | 位置数 | 模型 | a | b | c |','|---|---:|---|---:|---:|---:|']
        for k in ('unclipped','clipped'):
            g=v[k]
            for model in ('B','R@100','R-noclip@100','R-cond@100'):
                lines.append(f"| {k} | {g['positions']} | {model} | "+' | '.join(f"{g['metrics'][model+'/'+m]['mean']:.6f}" for m in ('top1_matches_teacher2','rank2_5_geom_ratio','reflection_total'))+' |')
            lines+=['',f"{k} 老师反思词总质量 {g['metrics']['teacher/reflection_total']['mean']:.6f}。",'']
        lines+=['','| 首选被截，1/q(U) 五分位 | 位置数 | b 的 R−B | b 的 R/B |','|---|---:|---:|---:|']
        for j in range(1,6):
            g=v[f'clipped_Q{j}']
            cells=[show(g['metrics'][m]['mean'],g['metrics'][m]['ci95']) for m in ('R_minus_B/rank2_5_geom_ratio','R_over_B_rank2_5_ratio')]
            lines.append(f"| {j} | {g['positions']} | "+' | '.join(cells)+' |')
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
