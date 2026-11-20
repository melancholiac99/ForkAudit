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
    values.update(tasks=len(groups),episodes=len(rows),strict_ratio=(values['all_requirements']/values['strict']
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
                    **{k:row[k] for k in (*METRICS,'tasks','episodes','legal_rate','strict_ratio','decision_entropy')}))
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
