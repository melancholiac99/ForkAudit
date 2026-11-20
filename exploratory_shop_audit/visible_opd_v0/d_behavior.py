"""Read-only comparison of the paired native evaluations at steps zero and ten."""
from collections import Counter, defaultdict
import numpy as np
from .acceptance import ALL_REQUIREMENTS
from .d_data import ARMS, PARENT, RUN
from .s23_cross import EVIDENCE
from .s22_report import tsv
from exploratory_shop_audit.grpo_forced_audit.common import read, save, stable_seed, digest_file

METRICS = ('reward', 'all_requirements', 'strict', 'length')
POOLS = ('T_opd_eval','V')
MODES = {'sample':range(4), 'greedy':(4,)}
PAIRS = [('P','R-beta'),('P','P-shuffled'),('P','R')] + [(a,'B') for a in ARMS]


def completed(arm, splits):
    root = RUN/'evaluation'/arm
    ranks = range(4) if arm=='B' else [ARMS.index(arm)]
    episodes = {}
    for rank in ranks:
        path = root/f'rank{rank}'
        meta = read(path/'complete.json')
        assert (meta['arm'],meta['rank'],meta['k'])==(arm,rank,0 if arm=='B' else 10)
        count = 0
        for p in sorted(path.glob('episodes_*.json')):
            for row in read(p):
                key = row['split'],row['goal'],row['rep']
                assert key not in episodes and sum(row['events'].values())==row['length']
                episodes[key] = row
                count += 1
        assert count==meta['count']
    expected = {(name,t['goal_index'],rep) for name in POOLS for t in splits[name] for rep in range(5)}
    assert episodes.keys()==expected
    return episodes


def readout(groups):
    rows = [e for group in groups for e in group]
    def scalar(e):
        return e['reward'],e['reason'] in ALL_REQUIREMENTS,e['reason']=='gold_purchase',e['length']
    per_task = np.array([np.mean([scalar(e) for e in group],axis=0) for group in groups])
    values = dict(zip(METRICS,per_task.mean(axis=0).tolist()))
    events = Counter(); tokens = Counter(); ent = Counter()
    for e in rows:
        events.update(e['events']); tokens.update(e['tokens']); ent.update(e['entropy_sum'])
    keys = ('search','open_listed_product','select_visible_option','buy')
    decision_n = tokens['click']+tokens['option']
    # Early completed chunks used a strict validity assertion before writing any episode.
    valid=sum(e.get('reward_valid',True) for e in rows)
    values.update(tasks=len(groups),episodes=len(rows),reward_valid_rate=valid/len(rows),invalid_rewards=len(rows)-valid,strict_ratio=(values['all_requirements']/values['strict']
        if values['strict'] else None),reward_types=dict(Counter(e['reason'] for e in rows)),
        legal_rate=1-events['illegal']/sum(events.values()),action_counts=events,
        funnel_episode_rate={k:sum(e['events'].get(k,0)>0 for e in rows)/len(rows) for k in keys},
        decision_token_count=decision_n,
        decision_entropy=(ent['click']+ent['option'])/decision_n if decision_n else None,
        entropy_by_class={k:{'tokens':tokens[k],'entropy':ent[k]/tokens[k] if tokens[k] else None}
                          for k in ('search','click','option')})
    return values,per_task


def cluster_draws(values, labels, seed):
    """Resample whole clusters and retain the original equal-task estimand."""
    names, idx = np.unique(labels,return_inverse=True)
    sums = np.zeros((len(names),values.shape[1]))
    np.add.at(sums,idx,values)
    weights = np.bincount(idx)
    draws = np.random.default_rng(seed).multinomial(len(names),np.full(len(names),1/len(names)),size=10000)
    return (draws@sums)/(draws@weights)[:,None]


def interval(point, draws):
    ci = np.quantile(draws,[.025,.975],axis=0)
    return {m:dict(estimate=float(point[i]),ci95=ci[:,i].tolist()) for i,m in enumerate(METRICS)}


def report():
    splits = read(PARENT/'asin_split_manifest.json')
    catalog = {name:splits[name] for name in POOLS}
    train_asins = {t['asin'] for name in ('T_cal','T_opd') for t in splits[name]}
    seen = [t for t in catalog['V'] if t['asin'] in train_asins]
    unseen = [t for t in catalog['V'] if t['asin'] not in train_asins]
    catalog.update(V_seen=seen,V_unseen=unseen)
    summary = {}; arrays = {}; table = []
    for arm in ('B',*ARMS):
        outcomes = completed(arm,splits)
        for pool,tasks in catalog.items():
            source = 'V' if pool.startswith('V') else pool
            for mode,reps in MODES.items():
                key = f'{arm}/{pool}/{mode}'
                if not tasks:
                    summary[key] = {'tasks':0,'status':'EMPTY_STRATUM'}
                    continue
                groups = [[outcomes[source,t['goal_index'],r] for r in reps] for t in tasks]
                row,data = readout(groups)
                summary[key] = row; arrays[arm,pool,mode] = data
                table.append(dict(arm=arm,checkpoint=0 if arm=='B' else 10,pool=pool,decode=mode,
                    **{k:row[k] for k in (*METRICS,'tasks','episodes','legal_rate','reward_valid_rate','invalid_rewards','strict_ratio','decision_entropy')}))
    paired = {}; gaps = {}
    for mode in MODES:
        for arm in ('B',*ARMS):
            train,valid = (summary[f'{arm}/{p}/{mode}'] for p in POOLS)
            gaps[f'{arm}/{mode}'] = {m:train[m]-valid[m] for m in ('reward','strict','all_requirements')}
        for left,right in PAIRS:
            for cluster in ('task','asin'):
                intervals = {}; pool_draws = {}
                for pool in POOLS:
                    tasks = catalog[pool]
                    d = arrays[left,pool,mode] - arrays[right,pool,mode]
                    labels = [t['goal_index' if cluster=='task' else 'asin'] for t in tasks]
                    draws = cluster_draws(d,labels,stable_seed(20261120,'stage_d_bootstrap',mode,pool,cluster))
                    intervals[pool] = interval(d.mean(0),draws)
                    pool_draws[pool] = draws
                point = np.array([intervals['T_opd_eval'][m]['estimate']-intervals['V'][m]['estimate'] for m in METRICS])
                intervals['difference_of_gaps'] = interval(point,pool_draws['T_opd_eval']-pool_draws['V'])
                paired[f'{left}-{right}/{mode}/{cluster}'] = intervals
    save(EVIDENCE/'behavior_report.json',dict(checkpoints={'0':'B shared by all arms','10':list(ARMS),
        'B_at_10':'same frozen policy, same seeds as B at zero'},summary=summary,paired=paired,gaps=gaps,
        independent_pool_bootstrap=True,entropy='full native vocabulary at temperature 1; token-weighted on click and option',
        manifest_sha256=digest_file(PARENT/'asin_split_manifest.json')))
    tsv(EVIDENCE/'behavior_summary.tsv',table)
    markdown(read(EVIDENCE/'training_report.json'),summary,paired,gaps)


def markdown(training,summary,paired,gaps):
    lines=['# 阶段 D 十步试点','',
        '四组使用恒定学习率 1e-6、冻结投影器、相同的任务顺序和随机种子方案。每步在本组当时策略生成的轨迹上冻结目标。ΔKL 为当步更新前后针对同一目标的 KL 下降量。','',
        '| 训练组 | KL 下降步数 / 10 | 累计 ΔKL | 累计对齐量 | 累计移动量 | 固定探针首尾 KL | search / click / 规格覆盖门 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for a in ARMS:
        row=training['arms'][a]
        cover=' / '.join('通过' if row['coverage_pass'][c] else '不足' for c in ('search','click','option'))
        lines.append(f"| {a} | {row['improving_updates']} | {row['sum_fixed_target_delta']:+.6f} | "
                     f"{row['sum_fixed_target_alignment']:+.6f} | {row['sum_fixed_target_movement']:.6f} | "
                     f"{row['start_to_end_probe_movement']:.6f} | {cover} |")
    lines.extend(['','[40 次更新的 FP32 分解与相邻探针移动量](training_steps.tsv) · [累计量、类别诊断、冻结系数和覆盖门](training_report.json)',
        '', '## 无特权行为评测','',
        '两个任务池各 200 题。采样为每题 4 条，温度 0.7；贪心为每题 1 条。0 步四组共享 B，B 在 10 步仍是同一冻结策略。下列全要求购买率以百分比计。','',
        '| 任务池 | 解码 | B@0 | R@10 | Rβ@10 | P@10 | P-shuffled@10 |','|---|---|---:|---:|---:|---:|---:|'])
    for pool in POOLS:
        for mode in MODES:
            rates=[summary[f'{a}/{pool}/{mode}']['all_requirements'] for a in ('B',*ARMS)]
            lines.append(f'| {pool} | {mode} | '+' | '.join(f'{100*x:.2f}%' for x in rates)+' |')
    lines.extend(['','两项主比较及 P−R 的采样全要求购买率差，单位为百分点。括号内为配对商品簇 bootstrap 95% 区间。','',
        '| 比较 | T_opd_eval | V |','|---|---:|---:|'])
    for a,b in PAIRS[:3]:
        row=paired[f'{a}-{b}/sample/asin']
        entries=[]
        for p in POOLS:
            m=row[p]['all_requirements'];lo,hi=m['ci95']
            entries.append(f"{100*m['estimate']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
        lines.append(f'| {a} − {b} | '+' | '.join(entries)+' |')
    lines.extend(['','[严格成功率、学生决策熵与漏斗](behavior_summary.tsv) · [终止类型、分层结果、训练与验证差距、任务簇和商品簇区间](behavior_report.json)',
        '', '本次停在第 10 步；V@10 仅作描述。'])
    (EVIDENCE/'STAGE_D.md').write_text('\n'.join(lines)+'\n')
