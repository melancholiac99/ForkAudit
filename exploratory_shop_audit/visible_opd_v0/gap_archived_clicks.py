"""CPU-only §26 appendix: count attempts to click an absent gold ASIN in archived Stage E."""
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path

from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, digest_file, digest_value, read, save
from exploratory_shop_audit.grpo_forced_audit.interface import Engine


def main():
    runs=ROOT/'.runtime/runs'
    old=runs/'shop_visible_opd_e25_20261121'
    formal=old/'formal_2512'
    parent=runs/'shop_visible_opd_v04_s16_20261120'
    goals=read(parent/'snapshots/goals_frozen.json')
    engine=Engine(parent)
    out=Path(__file__).parent/'evidence/section26_20261123'
    files={}
    def track(path):
        files[str(path.relative_to(runs))]=digest_file(path)
        return path

    @lru_cache(maxsize=16384)
    def page_absent(g, prefix, command):
        gold=goals[g]['asin'].casefold()
        key=f'archive_{g}'
        state=engine.open(key,g,prefix)
        try:
            listed={p['asin'].casefold() for p in state['public'].get('products',[])}
            item=(state['public'].get('product') or {}).get('asin','').casefold()
            visible={v.casefold() for v in state['available']['clickables']}
            assert gold not in visible and not engine.step(key,command)['legal']
            return gold not in listed and gold!=item and gold not in state['observation'].casefold()
        finally:
            engine.close(key)

    def check(c,g,actions,illegal):
        c['episodes']+=1;c['actions']+=len(actions);c['illegal_actions']+=sum(illegal)
        hits=0
        for j,(action,bad) in enumerate(zip(actions,illegal,strict=True)):
            if not bad:continue
            kind,value=parse_action_fixed(action)
            if kind=='click' and (value or '').casefold()==goals[g]['asin'].casefold():
                c['illegal_click_gold']+=1
                if page_absent(g,tuple(actions[:j]),action):
                    c['absent_gold_clicks']+=1;hits+=1
        c['episodes_with_absent_gold_click']+=int(hits>0)

    training={}
    for arm in ('P','R','R-matched','P-frozen'):
        by_step={}
        for p in sorted((formal/'pilot').glob(f'*/{arm}/optimizer.json')):
            folder=p.parent;k=int(folder.parent.name)
            if arm=='P-frozen' and k==10:continue  # Imported D checkpoint, no Stage E update at 10.
            stat=Counter()
            for rep in range(2):
                path=folder/f'rollout_{rep}.jsonl'
                if not path.exists():path=path.with_suffix('.jsonl.gz')
                op=gzip.open if path.suffix=='.gz' else open
                traj=defaultdict(list)
                with op(track(path),'rt') as stream:
                    for line in stream:
                        r=json.loads(line);g=r['goal'];kind,value=parse_action_fixed(r['action'])
                        if 'legal' in r:legal=r['legal']
                        else:
                            allowed={v.casefold() for v in r['legal_clicks']}
                            legal=(kind=='finish' or (kind=='search' and bool(value)) or
                                   (kind=='click' and (value or '').casefold() in allowed and value.casefold()!='search'))
                            stat['actions_with_reconstructed_legality']+=1
                        assert r['rep']==rep and r['step']==len(traj[g])
                        traj[g].append((r['action'],not legal,r['done']))
                assert len(traj)==32
                for g,seq in traj.items():
                    assert seq[-1][2] and sum(s[2] for s in seq)==1
                    check(stat,g,[a for a,_,_ in seq],[b for _,b,_ in seq])
            by_step[str(k)]=dict(stat)
        training[arm]=by_step

    # Reuse action strings from any archived same-goal trace only after matching the
    # original ordered-action digest. Published §25 reconstructions have no new GPU work here.
    bank={}
    def add(g,events,expected=None):
        actions=[e['action'] for e in events];h=digest_value(actions)
        assert expected is None or h==expected
        bank[g,h]=events
    for variant in ('reconstruction','exact_retry'):
        for p in sorted((old/'migration_2514/search'/variant).glob('*/rank*/*.json')):
            if p.name=='complete.json':continue
            for r in read(track(p)):add(r['goal'],r['actions'],r['actions_sha'])
    base=formal/'evaluation'
    sources=sorted(base.glob('*/*/rank*/episodes_*.json'))
    for p in sources:
        other=p.with_name(p.name.replace('episodes_','search_',1))
        if not other.exists():continue
        traces=read(track(other))
        for r in read(p):
            key=f"{r['split']}_{r['goal']}_{r['rep']}"
            add(r['goal'],traces[key],r['actions_sha'])

    evaluation=defaultdict(Counter)
    for p in sources:
        arm=p.parents[1].name;k=int(p.parents[2].name)
        for r in read(track(p)):
            c=evaluation[f"{arm}@{k}/{r['split']}/"+('greedy' if r['rep']==4 else 'sample')]
            g=r['goal'];n=r['events'].get('illegal',0);events=bank.get((g,r['actions_sha']))
            if events is None:
                c['episodes']+=1;c['actions']+=r['length'];c['illegal_actions']+=n
                c['unresolved_episodes']+=int(n>0)
                c['unresolved_illegal_actions']+=n
                continue
            b=[e['event']=='illegal' for e in events]
            assert len(events)==r['length'] and sum(b)==n
            check(c,g,[e['action'] for e in events],b)
    def total(tables):
        c=sum((Counter(t) for t in tables),Counter())
        return {k:c[k] for k in ('episodes','actions','illegal_actions','illegal_click_gold','absent_gold_clicks',
                                 'episodes_with_absent_gold_click','unresolved_episodes','unresolved_illegal_actions')}
    totals={a:total(steps.values()) for a,steps in training.items()}
    eval_totals={a:total(v for k,v in evaluation.items() if k.split('@')[0]==a)
                 for a in ('B','P','P-frozen','R','R-matched')}
    save(out/'archived_illegal_gold_clicks.json',dict(
        definition='Illegal native click with argument exactly matching the task gold ASIN (case-insensitive). Replay the archived prefix, require the ASIN absent from currently listed products, current item, legal clickables and rendered current observation. Count each attempted action.',
        population='Stage E adopted formal run training tapes and archived baseline/formal evaluation episodes. The 10 inherited D updates of P-frozen are excluded from Stage E training. Shared prefixes across arms are included in each arm; per-arm totals are descriptive. No §26 evaluation was read.',
        missing='An episode with archived zero illegal actions is an exact zero. A missing action-string digest with a nonzero illegal count contributes unresolved_illegal_actions to the upper bound only.',
        training_summary=totals,evaluation_summary=eval_totals,training_by_update=training,evaluation_by_checkpoint_split_mode=evaluation,
        source_files=len(files),source_manifest_sha256=digest_value(files),goals_sha256=digest_file(parent/'snapshots/goals_frozen.json')))


if __name__=='__main__':
    main()
