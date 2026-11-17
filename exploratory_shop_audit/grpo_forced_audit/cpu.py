"""CPU-only task cleaning, exact script screening, and frozen splits."""
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import itertools
import json
import random
import sys
import time

from .common import *
from .interface import Engine
from exploratory_shop_audit.derive_environment import configurations


def clean(run):
    setup(run)
    engine = Engine(run)
    # Import the community's own published clean-set test helpers, never execute
    # its tests or open its final evaluation task file.
    spec = importlib.util.spec_from_file_location('community_clean_standard', COMMUNITY/'tests/test_evaluation_dataset.py')
    standard = importlib.util.module_from_spec(spec); spec.loader.exec_module(standard)
    products = read(SOURCE/'community_products.json')
    blind_ids=set(read(COMMUNITY/'src/shopping_grpo/resources/blind_final_task_ids.json')['task_ids'])
    goals=engine.server.goals
    assert len(products)==len(goals)
    identities=[identity(i,p,g) for i,(p,g) in enumerate(zip(products,goals))]
    assert all(p['asin']==g['asin'] for p,g in zip(products,goals))
    forbidden_hash={r['content_hash'] for r in identities if r['source_task_id'] in blind_ids or r['tag']!='train'}
    forbidden_instruction={r['instruction_hash'] for r in identities if r['source_task_id'] in blind_ids or r['tag']!='train'}
    rows=[]; seen=set()
    for i,(raw,goal,ident) in enumerate(zip(products,goals,identities)):
        reasons=[]; checks={}; result=None
        if raw['tag']!='train': reasons.append('upstream_eval')
        if i in blind_ids: reasons.append('blind_final_id')
        if ident['content_hash'] in forbidden_hash or ident['instruction_hash'] in forbidden_instruction: reasons.append('protected_content_overlap')
        if ident['content_hash'] in seen: reasons.append('exact_duplicate')
        seen.add(ident['content_hash'])
        if not reasons:
            product=engine.server.product_item_dict[goal['asin']]
            chosen,errors=standard.candidate_options_for_evaluation(product,goal['required_options_by_key'])
            price=standard.resolve_variant_price(product,chosen)
            result=standard.evaluate_purchase(deepcopy(product),deepcopy(goal),selected_options=chosen,price_resolution=price,rewards=engine.server.environment_config['reward'])
            checks={'gold_purchase':result.reward_type=='gold_purchase' and result.reward_valid and result.reward==1,
                    'unique_gold_axes':not goal['unresolved_option_requirements'] and errors['status']=='pass',
                    'variant_price':price['status']=='pass',
                    'explicit_budget':not standard.PRICE_HINT.search(goal['instruction_text']) or goal['price_upper'] is not None}
            reasons.extend(k for k,v in checks.items() if not v)
            # The same native click mapping must make the gold choices reachable.
            options=product['options']; mapping={v:a.lower() for a,vals in options.items() for v in vals}
            from web_agent_site.engine.reward_features import canonicalize_option_axis, normalize_option_text
            reachable={canonicalize_option_axis(a):{normalize_option_text(v) for v,b in mapping.items() if b==a.lower()} for a in options}
            reachable_gold=all(normalize_option_text(v) in reachable.get(canonicalize_option_axis(k),set()) for k,v in chosen.items())
            checks['native_gold_actions_reachable']=reachable_gold
            if not reachable_gold: reasons.append('native_gold_actions_unreachable')
        rows.append({**ident,'checks':checks,'failure_reasons':reasons,'keep':not reasons,
                     'gold_score':result.to_dict() if result is not None else None})
    with (run/'task_cleaning.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    retained=[r for r in rows if r['keep']]
    old={t['task_id'] for t in read(PARENT/'tasks.json')['formal']}
    summary={'total':len(rows),'train':sum(r['tag']=='train' for r in rows),'retained':len(retained),
        'failures':dict(Counter(x for r in rows for x in r['failure_reasons'])),
        'old_200':{'passed':sum(r['keep'] and r['task_id'] in old for r in rows),'failure_reasons':dict(Counter(x for r in rows if r['task_id'] in old for x in r['failure_reasons']))},
        'categories':{c:{'train':sum(r['category']==c and r['tag']=='train' for r in rows),'retained':sum(r['category']==c for r in retained)} for c in sorted({r['category'] for r in rows})},
        'cleaning_source':str(COMMUNITY/'tests/test_evaluation_dataset.py'),'cleaning_source_sha256':digest_file(COMMUNITY/'tests/test_evaluation_dataset.py'),
        'blind_ids_sha256':digest_file(COMMUNITY/'src/shopping_grpo/resources/blind_final_task_ids.json')}
    save(run/'cleaning_summary.json',summary); print(json.dumps(summary,ensure_ascii=False),flush=True)


def complete_configurations(product, fixed=None):
    """Exactly native reachable full-axis choices; no partial/invalid zero maxima."""
    required={k.lower() for k in product['options']}
    for options in configurations(product):
        if set(options)!=required: continue
        if fixed and any(options.get(k)!=v for k,v in fixed.items()): continue
        yield options


def maximum(engine, goal_index, asin, fixed=None):
    product=engine.server.product_item_dict[asin]; goal=engine.server.goals[goal_index]
    best=None; argmax=None; total=0; valid=0; invalid=Counter()
    for options in complete_configurations(product,fixed):
        # Evaluate native scoring once; no-price is not this experiment's primary.
        result=engine.evaluate(deepcopy(product),deepcopy(goal),selected_options=options,price_resolution=None,rewards=engine.server.environment_config['reward'])
        total+=1
        if result.reward_valid:
            valid+=1
            if best is None or result.reward>best: best=result.reward;argmax=options
        else: invalid[result.reward_type]+=1
    return {'best':best,'options':argmax,'configurations':total,'valid_configurations':valid,'invalid_reasons':dict(invalid),
            'complete_enumeration':True,'all_valid':total>0 and valid==total}


def buy_config(engine,key,goal_index,query,asin,options):
    engine.open(key,goal_index)
    result=engine.step(key,f'search[{query}]')
    result=engine.step(key,f'click[{asin}]')
    for value in options.values():
        if result['status']['done']: return result
        result=engine.step(key,f'click[{value}]')
    if not result['status']['done']: result=engine.step(key,'click[buy now]')
    engine.close(key)
    return result


def screen(run, shard=0, shards=1):
    engine=Engine(run); tasks=[r for r in read_rows(run/'task_cleaning.jsonl') if r['keep']]
    tasks=tasks[shard::shards]
    output=run/(f'script_screening12_{shard:02d}.jsonl' if shards>1 else 'script_screening.jsonl')
    done={r['task_id'] for r in read_rows(output)+read_rows(run/'screening_completed_before_parallel_resize.jsonl')}
    started=time.monotonic()
    for n,task in enumerate(tasks):
        if task['task_id'] in done: continue
        idx=task['goal_index']; query=engine.server.goals[idx]['instruction_text']; key='screen'
        engine.open(key,idx); result=engine.step(key,f'search[{query}]')
        products=result['state']['public'].get('products',[])
        scores=[]; s1=None;s2=None
        if products:
            asin=products[0]['asin']; product=engine.server.product_item_dict[asin]
            first=next(complete_configurations(product),None)
            if first is not None: s1=buy_config(engine,'s1',idx,query,asin,first)['status']
            for p in products:
                m=maximum(engine,idx,p['asin']); scores.append({'asin':p['asin'],'rank':p['rank'],**m})
            if scores[0]['options'] is not None:
                s2=buy_config(engine,'s2',idx,query,asin,scores[0]['options'])['status']
                if s2.get('purchase'): assert s2['reward']==scores[0]['best'] and s2['reward_valid']
        easy=bool(s1 and s1.get('termination_reason')=='gold_purchase')
        full1=bool(scores and scores[0]['best']==1)
        if easy: label='easy'
        elif full1 and s2 and s2.get('reward')==1: label='H_spec'
        elif any(p['best']==1 for p in scores[1:]): label='H_click'
        elif scores and all(p['all_valid'] for p in scores) and not full1: label='H_search'
        else: label='H_unknown'
        append(output,{'task_id':task['task_id'],'category':task['category'],'easy':easy,'label':label,
            's1':s1,'s2':s2,'query':query,'shown_scores':scores,'interface':INTERFACE})
        engine.close(key)
        if n%100==0: print(json.dumps({'stage':'screen','completed':n+1,'total':len(tasks),'seconds':time.monotonic()-started}),flush=True)
    rows=read_rows(output);save(run/(f'screening_summary_{shard:02d}.json' if shards>1 else 'screening_summary.json'),{'tasks':len(rows),'labels':dict(Counter(r['label'] for r in rows)),'cpu_seconds':time.monotonic()-started})


def stratified(rows,n,purpose):
    groups=defaultdict(list)
    for r in rows:groups[r['category']].append(r)
    rng=random.Random(stable_seed(SEED,purpose)); chosen=[]
    # Proportional stratification, largest remainders, deterministic within category.
    n=min(n,len(rows)); quotas={k:int(n*len(v)/max(1,len(rows))) for k,v in groups.items()}
    order=sorted(groups,key=lambda k:(-(n*len(groups[k])/max(1,len(rows))-quotas[k]),stable_seed(SEED,purpose,k)))
    for k in order[:n-sum(quotas.values())]:quotas[k]+=1
    for k in sorted(groups):
        subset=sorted(groups[k],key=lambda r:r['task_id']);rng.shuffle(subset);chosen+=subset[:quotas[k]]
    rng.shuffle(chosen);return chosen


def split(run):
    tasks={r['task_id']:r for r in read_rows(run/'task_cleaning.jsonl') if r['keep']}
    screens=read_rows(run/'script_screening.jsonl'); assert len(screens)==len(tasks)
    for r in screens:tasks[r['task_id']].update(label=r['label'],easy=r['easy'])
    rows=list(tasks.values()); hard=[r for r in rows if not r['easy']]; easy=[r for r in rows if r['easy']]
    vhard=stratified(hard,100,'V_hard'); veasy=stratified(easy,200-len(vhard),'V_easy')
    selected={r['task_id'] for r in vhard+veasy}
    if len(selected)<200:
        vhard+=stratified([r for r in hard if r['task_id'] not in selected],200-len(selected),'V_fill')
    V=vhard+veasy; used={r['task_id'] for r in V}
    extra_hard=stratified([r for r in hard if r['task_id'] not in used],50,'extra_hard')
    extra_easy=stratified([r for r in easy if r['task_id'] not in used],100-len(extra_hard),'extra_easy')
    extra=extra_hard+extra_easy;used|={r['task_id'] for r in extra}
    T=[r for r in rows if r['task_id'] not in used]
    E1=stratified(vhard,50,'E1'); e1ids={r['task_id'] for r in E1}
    # Audit shrinking is nested and keeps all E1 tasks first.
    E0=E1+stratified([r for r in vhard if r['task_id'] not in e1ids],200-len(E1),'E0_V')
    E0+=stratified([r for r in T if not r['easy']],200-len(E0),'E0_T')
    pilot=stratified([r for r in T if not r['easy']],12,'pilot_hard')+stratified([r for r in T if r['easy']],12,'pilot_easy')
    if len(pilot)<24:pilot+=stratified([r for r in T if r['task_id'] not in {p['task_id'] for p in pilot}],24-len(pilot),'pilot_fill')
    output={'seed':SEED,'T':T,'V':V,'V_extra':extra,'E0':E0,'E1':E1,'pilot':pilot,
        'population_weights':{'hard':len(hard)/len(rows),'easy':len(easy)/len(rows)},
        'task_cleaning_sha256':digest_file(run/'task_cleaning.jsonl'),'screening_sha256':digest_file(run/'script_screening.jsonl')}
    write_once(run/'splits.json',output)
    print(json.dumps({k:len(output[k]) for k in ['T','V','V_extra','E0','E1','pilot']}),flush=True)
