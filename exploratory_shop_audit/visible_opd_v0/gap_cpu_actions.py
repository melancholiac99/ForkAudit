"""Post hoc classification of executed training actions in Stage E and section 26."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import argparse
import gzip
import json
from pathlib import Path
import re

from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, read, save, digest_file

RUNS = ROOT/'.runtime/runs'
FORMAL = RUNS/'shop_gap26_20261123/formal'
E_FORMAL = RUNS/'shop_visible_opd_e25_20261121/formal_2512'
PARENT = RUNS/'shop_visible_opd_v04_s16_20261120'
OUT = Path(__file__).parent/'evidence/section26_20261123/posthoc_cpu'
CACHE = FORMAL.parent/'posthoc_cpu'
NAV = {'back to search','< prev','next >','description','features','reviews','attributes','buy now'}
TYPES = ('format_unparseable','action_type_unavailable','absent_product_id','unavailable_option','nonexistent_navigation','other')
ID = re.compile(r'[0-9]{8,12}',re.ASCII)
COMPOUND = re.compile(r'\]\s*(?:Action:\s*)?[A-Za-z_]\w*\[', re.I)


def legality(r):
    kind,arg=parse_action_fixed(r['action'])
    value=arg.lower() if arg is not None else None
    return bool(kind=='finish' or (kind=='search' and value) or
                (kind=='click' and value!='search' and value in {str(v).lower() for v in r['legal_clicks']}))


def classify(r,gold,public,options):
    kind,arg=parse_action_fixed(r['action'])
    page=r['page']
    if arg is None:
        return 'format_unparseable','format',{}
    if kind not in {'click','search','finish'}:
        return 'action_type_unavailable','format',{}
    if kind=='search':
        return 'other','search',{'detail':'empty_search'}
    assert kind=='click'
    value=arg.lower()
    listed={str(i['asin']).lower() for i in public.get('products',[])}
    current=(public.get('product') or {}).get('asin')
    if current:
        listed.add(current.lower())
    if ID.fullmatch(value):
        if value in listed:
            return 'action_type_unavailable','click',{'detail':'visible_but_not_clickable_id'}
        matches=[(SequenceMatcher(None,value,i,autojunk=False).find_longest_match().size,len(i)) for i in listed]
        overlap=any(2*n>=min(len(value),length) for n,length in matches)
        return 'absent_product_id','click',dict(gold=value==gold,partial=overlap,
                              longest=max((n for n,_ in matches),default=0),visible_ids=len(listed))
    if value in NAV or re.fullmatch(r'(back|prev(ious)?|next|home|checkout|buy)([ _<>-]*(page|now|to search|[<>]))?',value.strip(),re.I):
        return 'nonexistent_navigation','click',{}
    if not listed and not r['legal_clicks']:
        return 'action_type_unavailable','click',{}
    if COMPOUND.search(r['action']) or not value.strip():
        return 'other','click',{'detail':'compound_or_empty_click'}
    if value in options or (page=='product_detail' and public.get('available_options')):
        if page=='product_detail':
            return 'unavailable_option','option',{'detail':'catalog_value' if value in options else 'unrecognized_single_value_on_option_page'}
        return 'action_type_unavailable','option',{'detail':'option_off_detail_page'}
    return 'other','click',{'detail':'unrecognized_click'}


def tapes(root,arm,k):
    folder=root/'pilot'/f'{k:02}'/arm
    for rep in range(2):
        p=folder/f'rollout_{rep}.jsonl'
        yield p if p.is_file() else p.with_suffix('.jsonl.gz')


def rows(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        for line in stream:
            yield json.loads(line)


@lru_cache(maxsize=1)
def native_environment():
    from exploratory_shop_audit.grpo_forced_audit.interface import Engine
    env=Engine(PARENT)
    goals=read(PARENT/'snapshots/goals_frozen.json')
    options={str(v['value']).lower() for item in env.server.product_item_dict.values()
             for axis in (item.get('customization_options') or {}).values() for v in axis if isinstance(v,dict) and 'value' in v}
    # The native catalog may normalize option axes during ingestion.
    if not options:
        from exploratory_shop_audit.runtime import SOURCE
        options={str(v['value']).lower() for item in read(SOURCE/'community_products.json')
                 for axis in (item.get('customization_options') or {}).values() for v in axis if v.get('value') is not None}
    assert len(options)>100
    return env,goals,options


def audit_window(job):
    root,arm,first,end=job
    receipt = CACHE/'windows'/f'{arm}_{first:02}.json'
    version = digest_file(__file__)
    if receipt.is_file():
        saved = read(receipt)
        if (saved['code']==version and all(digest_file(RUNS/p)==h for p,h in saved['sources'].items())
                and (root!=FORMAL or all((CACHE/f'{arm}_{k:02}.json').exists() for k in range(first,end+1)))):
            return saved['window'],saved['sources'],saved['option_count']
    env,goals,options=native_environment()
    sources={}
    counts=Counter(); cross=Counter(); lengths=Counter(); detail=Counter()
    for k in range(first,end+1):
        annotations=[]
        for rep,path in enumerate(tapes(root,arm,k)):
            episodes=defaultdict(list)
            sources[str(path.relative_to(RUNS))]=digest_file(path)
            for r in rows(path):
                assert r['rep']==rep and r['step']==len(episodes[r['goal']])
                episodes[r['goal']].append(r)
            assert len(episodes)==32
            for g,seq in episodes.items():
                assert seq[-1]['done'] and sum(x['done'] for x in seq)==1
                flags=[legality(r) for r in seq]
                replay=not all(flags) or any('legal' not in r for r in seq)
                key=f'T_opd_D_{k}_{g}_{rep}'
                state=env.open(key,g) if replay else None
                for r,legal in zip(seq,flags):
                    if 'legal' in r:assert r['legal']==legal
                    if replay:
                        public=state['public']
                        assert r['page']==public['page_type'] and r['products']==[i['asin'] for i in public.get('products',[])]
                        actual=env.step(key,r['action'])
                        assert actual['legal']==legal and bool(actual['status']['done'])==r['done']
                        state=actual['state']
                    kind,arg=parse_action_fixed(r['action'])
                    if legal:
                        cls='option' if 'option' in r['classes'] else kind if kind in ('search','click') else 'format'
                        label='legal'
                    else:
                        label,cls,extra=classify(r,goals[g]['asin'].lower(),public,options)
                        if label=='absent_product_id':
                            cross[f"gold={int(extra['gold'])},partial={int(extra['partial'])}"]+=1
                            lengths[extra['longest']]+=1
                        if extra.get('detail'):detail[extra['detail']]+=1
                    counts[label]+=1
                    annotations.append([g,rep,r['step'],legal,cls])
                if replay:env.close(key)
        if root==FORMAL or (root/'pilot'/f'{k:02}'/arm/'collection.json').exists():
            saved=read(root/'pilot'/f'{k:02}'/arm/'collection.json')
            assert len(annotations)==saved['records'] and sum(r[3] for r in annotations)==saved['legal']
        if root==FORMAL:
            save(CACHE/f'{arm}_{k:02}.json',annotations)
    total=sum(counts.values());bad=total-counts['legal']
    window = dict(arm=arm,start=first,end=end,episodes=64*(end-first+1),actions=total,illegal=bad,
        illegal_fraction=bad/total,types={c:dict(n=counts[c],fraction_all=counts[c]/total,
        fraction_illegal=counts[c]/bad if bad else None) for c in TYPES},absent_id_cross={
        key:dict(n=c,fraction_absent=c/counts['absent_product_id'] if counts['absent_product_id'] else None)
        for key,c in ((f'gold={g},partial={v}',cross[f'gold={g},partial={v}']) for g in (0,1) for v in (0,1))},
        longest_digit_run=dict(lengths),details=dict(detail))
    print(arm,first,end,total,bad,flush=True)
    save(receipt, dict(code=version,window=window,sources=sources,option_count=len(options)))
    return window,sources,len(options)


def main(workers):
    CACHE.mkdir(exist_ok=True)
    jobs=[]
    for root,arms in ((FORMAL,('R-gap','P-gap')),(E_FORMAL,('P','R','R-matched'))):
        for arm in arms:
            steps=sorted(int(p.parent.parent.name) for p in (root/'pilot').glob('*/'+arm+'/optimizer.json'))
            assert steps==list(range(1,steps[-1]+1))
            jobs.extend((root,arm,k,min(k+4,steps[-1])) for k in range(1,steps[-1]+1,5))
    windows,sources=[],{}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for window,files,option_count in pool.map(audit_window,jobs):
            windows.append(window)
            assert not (sources.keys() & files.keys())
            sources.update(files)
    save(OUT/'illegal_actions.json',dict(windows=windows,source_sha256=sources,
         goals_sha256=digest_file(PARENT/'snapshots/goals_frozen.json'),catalog_option_values=option_count))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int, default=1)
    main(parser.parse_args().workers)
