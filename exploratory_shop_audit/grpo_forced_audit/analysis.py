"""Task-weighted saved-data statistics; privileged maxima are separate primitives."""
from collections import Counter,defaultdict
from itertools import combinations
import json
import math
import numpy as np
from .common import *
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.derive_environment import shown


def interval(values,name,level=.95):
    values=np.asarray(list(values),dtype=float)
    if not len(values):return {'estimate':None,'interval':None,'n_tasks':0,'level':level}
    rng=np.random.default_rng(stable_seed(SEED,name)); draws=np.array([np.mean(rng.choice(values,len(values),replace=True)) for _ in range(2000)])
    return {'estimate':float(values.mean()),'interval':np.quantile(draws,[(1-level)/2,1-(1-level)/2]).tolist(),'n_tasks':len(values),'level':level}


def action_kind(step):
    name,arg=parse_action_fixed(step['action']);public=step['before']['public']
    if name=='search':return 'search'
    if name=='click':
        if arg and arg.lower() in {str(p['asin']).lower() for p in public.get('products',[])}:return 'click'
        options=public.get('available_options',{})
        if arg and any(arg.lower()==str(v).lower() for values in options.values() for v in values):return 'spec'
    return 'other_or_terminal'


def page_key(step):return digest_value(step['before']['public'])


def get_maxima(run,episodes,steps,provider):
    path=run/'full_configuration_maxima.jsonl';cached={(r['task_id'],r['asin']):r for r in read_rows(path)}
    task_index={r['task_id']:r['goal_index'] for r in read_rows(run/'task_cleaning.jsonl')}
    required=set()
    for step in steps:
        for state in [step['before'],step['after']]:required|={(step['task_id'],asin) for asin in shown(state)}
    for tid,asin in sorted(required):
        if (tid,asin) in cached:continue
        result=provider(task_index[tid],asin)
        row={'task_id':tid,'asin':asin,**result};append(path,row);cached[(tid,asin)]=row
    return cached


def episode_metrics(episodes,steps,maxima):
    byid=defaultdict(list)
    for s in steps:byid[s['trajectory_id']].append(s)
    records=[]
    for ep in episodes:
        ss=sorted(byid[ep['trajectory_id']],key=lambda s:s['step']);tid=ep['task_id']
        seen=set();opened=set();ranks=[]
        for s in ss:
            for state in [s['before'],s['after']]:
                seen|=shown(state)
                if state['public'].get('product'):opened.add(state['public']['product']['asin'])
                purchase=ep['status'].get('purchase') or {}
                for p in state['public'].get('products',[]):
                    if p['asin']==purchase.get('asin'):ranks.append(p.get('rank'))
        row={k:ep.get(k) for k in ['task_id','trajectory_id','reward','reward_valid','length','termination_reason','purchase']}
        row.update(strict_success=ep.get('strict_success',ep['termination_reason']=='gold_purchase'),searches=sum(s['legal'] and parse_action_fixed(s['action'])[0]=='search' for s in ss),opened_products=len(opened),purchase_rank=ranks[0] if ranks else None,
            illegal_actions=sum(not s['legal'] for s in ss),actions=len(ss),generation_truncated=any(s.get('generation_truncated',False) or s.get('generation',{}).get('finish_reason')=='length' for s in ss))
        row['funnel']=None;reason=None
        if not ep.get('purchase'):reason='no_purchase'
        elif ep.get('reward_valid') is not True:reason='actual_reward_invalid'
        else:
            item=ep['status']['purchase']['asin'];sets=[seen,opened,{item}];upper=[]
            for group in sets:
                scores=[maxima[(tid,asin)]['best'] for asin in group if maxima[(tid,asin)]['best'] is not None]
                upper.append(max(scores) if scores else None)
            if any(x is None for x in upper):reason='no_valid_configuration_in_set'
            else:
                us,uo,uf=upper; gaps=[1-us,us-uo,uo-uf,uf-ep['reward']]
                assert all(g>=-1e-8 for g in gaps),gaps
                assert abs(sum(gaps)-(1-ep['reward']))<1e-8
                row['funnel']=gaps
            row['invalid_configuration_items']=sum(maxima[(tid,asin)]['valid_configurations']<maxima[(tid,asin)]['configurations'] for asin in seen)
        row['funnel_exclusion']=reason;records.append(row)
    return records


def summarize_records(records,name,level=.95):
    grouped=defaultdict(list)
    for r in records:grouped[r['task_id']].append(r)
    metrics={}
    fields=['reward','strict_success','purchase','length','searches','opened_products']
    for field in fields:metrics[field]=interval([np.mean([r[field] for r in rows]) for rows in grouped.values()],name+field,level)
    metrics['valid_rate']=interval([np.mean([r['reward_valid'] is True for r in rows]) for rows in grouped.values()],name+'valid',level)
    metrics['funnel_coverage']=interval([np.mean([r['funnel'] is not None for r in rows]) for rows in grouped.values()],name+'coverage',level)
    for i,field in enumerate(['L_search','L_click','L_stop','L_spec']):
        metrics[field]=interval([np.mean([r['funnel'][i] for r in rows if r['funnel'] is not None]) for rows in grouped.values() if any(r['funnel'] is not None for r in rows)],name+field,level)
    metrics['L_click_plus_spec']=interval([np.mean([r['funnel'][1]+r['funnel'][3] for r in rows if r['funnel'] is not None]) for rows in grouped.values() if any(r['funnel'] is not None for r in rows)],name+'L',.975)
    return {'metrics':metrics,'tasks':len(grouped),'episodes':len(records),'funnel_episodes':sum(r['funnel'] is not None for r in records),
            'termination_counts':dict(Counter(r['termination_reason'] for r in records)),'funnel_exclusions':dict(Counter(r['funnel_exclusion'] for r in records if r['funnel'] is None)),
            'illegal_action_rate':sum(r['illegal_actions'] for r in records)/max(1,sum(r['actions'] for r in records)),'records':records}


def evaluate_saved(run,episodes,steps,provider,name):
    maxima=get_maxima(run,episodes,steps,provider);records=episode_metrics(episodes,steps,maxima)
    split=read(run/'splits.json');tasks={t['task_id']:t for t in split['V']+split['T']+split['V_extra']}
    result={'all_unweighted':summarize_records(records,name)}
    for label in ['easy','hard','H_spec','H_click','H_search','H_unknown']:
        selected=[r for r in records if (tasks[r['task_id']]['easy'] if label=='easy' else not tasks[r['task_id']]['easy'] if label=='hard' else tasks[r['task_id']]['label']==label)]
        result[label]=summarize_records(selected,name+label)
    weights=split['population_weights'];result['population_weights']=weights;result['weighted_all']={}
    # Bootstrap within the deliberately sampled easy/hard strata, preserving weights.
    rng=np.random.default_rng(stable_seed(SEED,name,'weighted'))
    for field in ['reward','strict_success','purchase','length']:
        arrays={}
        for label in ['easy','hard']:
            groups=defaultdict(list)
            for r in result[label]['records']:groups[r['task_id']].append(r[field])
            arrays[label]=np.asarray([np.mean(v) for v in groups.values()])
        if any(not len(v) and weights[k]>0 for k,v in arrays.items()):result['weighted_all'][field]=None;continue
        estimate=sum(weights[k]*np.mean(v) for k,v in arrays.items() if len(v))
        draws=[sum(weights[k]*np.mean(rng.choice(v,len(v),replace=True)) for k,v in arrays.items() if len(v)) for _ in range(2000)]
        result['weighted_all'][field]={'estimate':float(estimate),'ci95':np.quantile(draws,[.025,.975]).tolist()}
    return result


def historical(run):
    from .interface import Engine
    from .cpu import maximum
    allowed={r['task_id'] for r in read_rows(run/'task_cleaning.jsonl') if r['keep']}
    episodes=[r for r in read_rows(PARENT/'episodes.jsonl') if r['phase']=='formal' and r['task_id'] in allowed]
    for ep in episodes:ep['strict_success']=ep['termination_reason']=='gold_purchase'
    ids={r['trajectory_id'] for r in episodes};steps=[r for r in read_rows(PARENT/'steps.jsonl') if r['trajectory_id'] in ids]
    engine=Engine(run);maxima=get_maxima(run,episodes,steps,lambda i,a:maximum(engine,i,a));records=episode_metrics(episodes,steps,maxima)
    by_task=defaultdict(list);by_id=defaultdict(list)
    for ep in episodes:by_task[ep['task_id']].append(ep)
    for s in steps:by_id[s['trajectory_id']].append(s)
    pair_records=[];variance=[];agreement=[]
    for tid,eps in by_task.items():
        contributions=defaultdict(float);n=len(eps)
        for ea,eb in combinations(eps,2):
            a=by_id[ea['trajectory_id']];b=by_id[eb['trajectory_id']]
            first=next((i for i,(sa,sb) in enumerate(zip(a,b)) if parse_action_fixed(sa['action'])!=parse_action_fixed(sb['action'])),min(len(a),len(b)))
            raw=next((i for i,(sa,sb) in enumerate(zip(a,b)) if sa['action']!=sb['action']),min(len(a),len(b)))
            label=action_kind(a[first]) if first<len(a) and first<len(b) else 'other_or_terminal' if len(a)!=len(b) else 'no_action_divergence_verify'
            contribution=(ea['reward']-eb['reward'])**2/(n*(n-1));contributions[label]+=contribution
            pair_records.append({'task_id':tid,'a':ea['trajectory_id'],'b':eb['trajectory_id'],'first_parsed_divergence':first,'first_raw_action_divergence':raw,'kind':label,'variance_contribution':contribution})
            for page in ['search_results','product_detail']:
                aa={};bb={}
                for seq,out in [(a,aa),(b,bb)]:
                    for s in seq:
                        if s['before']['public']['page_type']==page:out.setdefault(page_key(s),s)
                common=set(aa)&set(bb)
                for k in common:agreement.append({'task_id':tid,'page':page,'matched':True,'equal_action':parse_action_fixed(aa[k]['action'])==parse_action_fixed(bb[k]['action']),'equal_kind':action_kind(aa[k])==action_kind(bb[k]),'a':ea['trajectory_id'],'b':eb['trajectory_id']})
                if not common:agreement.append({'task_id':tid,'page':page,'matched':False,'equal_action':None,'a':ea['trajectory_id'],'b':eb['trajectory_id']})
        actual=float(np.var([ep['reward'] for ep in eps],ddof=1));assert abs(sum(contributions.values())-actual)<1e-10
        variance.append({'task_id':tid,'sample_variance':actual,'contributions':dict(contributions)})
    folder=run/'historical_reanalysis';folder.mkdir(exist_ok=True)
    save(folder/'metrics.json',{'interface':'historical_unchanged','episodes':len(episodes),'tasks':len(by_task),'funnel':summarize_records(records,'historical'),
        'page_pair_records':agreement,'first_divergence_records':pair_records,'variance_by_task':variance,'not_causal':True})
