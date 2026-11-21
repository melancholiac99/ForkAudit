"""Score deterministic first result pages with the native environment on CPU."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
import sys
import statistics
from .e25_paths import PUBLIC,RUN
from exploratory_shop_audit.grpo_forced_audit.common import COMMUNITY,read,save,digest_file
from exploratory_shop_audit.grpo_forced_audit.interface import Engine
from exploratory_shop_audit.derive_environment import configurations

OUT=PUBLIC/'migration_2514'
DATA=RUN/'migration_2514/search'
POOLS=('V','T_opd_eval')
PRICE=re.compile(r'(?:[¥￥$]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?(?:\s*[-~至到]\s*\d+(?:\.\d+)?)?\s*(?:元|块钱|人民币|rmb|cny)|价格|价位|售价|预算|不超过|低于|高于|便宜|以下|以内)',re.I)


def page_judgments():
    from exploratory_shop_audit.visible_opd_v0.e25_paths import RUN
    parent=RUN.parent/'shop_visible_opd_v04_s16_20261120'
    engine=Engine(parent)
    tasks=read(OUT/'intervention_tasks.json')
    @lru_cache(maxsize=None)
    def qualifies(g,asin):
        prod=engine.server.product_item_dict[asin]
        goal=engine.server.goals[g]
        for options in configurations(prod):
            verdict=engine.evaluate(prod,goal,selected_options=options,price_resolution=None,
                                    rewards=engine.server.environment_config['reward'])
            if verdict.reward_valid and verdict.reward_type in ('gold_purchase','valid_alternative_purchase'):
                return True
        return False
    historical=read(DATA/'authenticated.json')
    observed_pages=defaultdict(set)
    for row in historical:
        if row['searches']:
            search=row['searches'][0]
            observed_pages[row['split'],row['goal'],search['query']].add(tuple(search['products']))
    old=OUT/'intervention_result_pages.json'
    cache={(r['split'],r['goal'],r['query']):r for r in read(old)} if old.exists() else {}
    results=[]
    for t in tasks:
        g=t['goal'];pool=t['split']
        for query in sorted(set(t['queries']['20'])|set(t['queries']['30'])):
            if (pool,g,query) in cache:
                saved=cache[pool,g,query]
                assert observed_pages[pool,g,query]=={tuple(saved['page_asins'])}
                results.append(saved);continue
            key=f'query_{g}';engine.open(key,g)
            try:
                after=engine.step(key,f'search[{query}]')
                assert after['legal'] and not after['status']['done']
                products=[p['asin'] for p in after['state']['public'].get('products',[])]
            finally:engine.close(key)
            assert observed_pages[pool,g,query]=={tuple(products)},(pool,g,query)
            good=[p for p in products if qualifies(g,p)]
            results.append(dict(split=pool,goal=g,query=query,page_asins=products,qualifying_asins=good,
                   contains_qualifying=bool(good),contains_gold=engine.server.goals[g]['asin'] in products))
    save(OUT/'intervention_result_pages.json',results)
    save(DATA/'pages_source.json',dict(sha256=digest_file(OUT/'intervention_tasks.json')))


def metrics(rows,goals):
    sys.path.insert(0,str(COMMUNITY/'environments/ShopSimulator/shop_env'))
    from web_agent_site.engine.search import search_tokens
    observed=[r for r in rows if r['searches'] is not None]
    first=[r for r in observed if r['searches']]
    lengths=[];overlaps=[];spec=[];price=[];gold_pages=[];targets=defaultdict(int);hitgold=0
    for row in observed:
        for target in row['other_click']:
            targets[target.casefold()]+=1
    for r in first:
        s=r['searches'][0];goal=goals[r['goal']];query=s['query']
        q=set(search_tokens(query));inst=set(search_tokens(goal['instruction_text']))
        if q:
            overlaps.append(len(q&inst)/len(q))
            specs=set(search_tokens(' '.join(str(v) for v in goal['goal_options'])))
            specs.update(search_tokens(' '.join(map(str,goal.get('expected_model') or []))))
            spec.append(len(q&specs)/len(q))
            terms=set(search_tokens(' '.join(m.group(0) for m in PRICE.finditer(query))))
            price.append(len(q&terms)/len(q))
        lengths.append(s['tokens']);has_gold=goal['asin'] in s['products'];gold_pages.append(has_gold)
        if has_gold and (r['purchased_asin'] or '').casefold()==goal['asin'].casefold():hitgold+=1
    types=defaultdict(int)
    for target,count in targets.items():
        if target in ('back to search','< prev','next >'):kind='navigation'
        elif target in ('description','features','reviews','attributes'):kind='product_detail_tab'
        else:kind='other_visible_target'
        types[kind]+=count
    def avg(v):return statistics.mean(v) if v else None
    return dict(total_episodes=len(rows),authenticated=len(observed),with_first_query=len(first),without_search=sum(r['legal_search_count']==0 for r in rows),
         length_tokens=avg(lengths),instruction_overlap=avg(overlaps),spec_fraction=avg(spec),price_fraction=avg(price),
         first_page_gold_count=sum(gold_pages),first_page_gold_rate=avg(gold_pages),
         bought_gold_with_gold_visible=hitgold,bought_gold_conditional=hitgold/sum(gold_pages) if sum(gold_pages) else None,
         other_click_target_types=dict(types),other_click_targets=dict(sorted(targets.items())))


def describe(historical=True,arm=None,k=None):
    goals=read(RUN.parent/'shop_visible_opd_v04_s16_20261120/snapshots/goals_frozen.json')
    if historical:
        recorded=read(DATA/'authenticated.json')
        checkpoints={step:[r for r in recorded if r['step']==step] for step in (0,10,20,30)}
        dest=OUT/'archived_search_metrics.json'
    else:
        from exploratory_shop_audit.env_worker import parse_action_fixed
        base=RUN/'formal_2512/evaluation'/f'{k:03}'/arm;records=[]
        for rank in range(4):
            sub=base/f'rank{rank}';assert (sub/'complete.json').exists()
            for f in sorted(sub.glob('search_*.json')):
                orig=read(sub/f.name.replace('search_','episodes_',1))
                traces=read(f)
                for r in orig:
                    actions=traces[f"{r['split']}_{r['goal']}_{r['rep']}"]
                    searches=[dict(tokens=a['arg_tokens'],query=a['query'],products=a['products']) for a in actions if a['event']=='search']
                    purchased=None
                    if actions[-1]['event']=='buy':
                        purchased=next(parse_action_fixed(a['action'])[1] for a in reversed(actions) if a['event']=='open_listed_product')
                    records.append(dict(split=r['split'],rep=r['rep'],goal=r['goal'],searches=searches,legal_search_count=len(searches),
                                        purchased_asin=purchased,other_click=[a['target'] for a in actions if a['event']=='other_click']))
        assert len(records)==2000
        checkpoints={k:records};dest=OUT/arm/'search'/f'{k:03}.json'
    stats={}
    for step,data in checkpoints.items():
        stats[step]={}
        for pool in POOLS:
            for mode in ('sample','greedy'):
                rows=[r for r in data if r['split']==pool and (r['rep']==4)==(mode=='greedy')]
                stats[step][f'{pool}/{mode}']=metrics(rows,goals)
    save(dest,dict(tokenization='native deduplicated Chinese bigrams and Latin/number tokens, query denominator; option/model terms from frozen goal, budget terms from explicit monetary expressions',results=stats))


if __name__=='__main__':
    if sys.argv[1:] == ['pages']:page_judgments()
    elif len(sys.argv)==3:describe(False,sys.argv[1],int(sys.argv[2]))
    else:describe()
