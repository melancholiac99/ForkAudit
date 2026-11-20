"""CPU-only replay verifies terminal payments and counts native shopping actions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.interface import Engine
from exploratory_shop_audit.grpo_forced_audit.common import read_rows, save

def audit_batch(engine, folder):
    by_episode = defaultdict(list)
    for row in read_rows(folder/'steps.jsonl'):
        by_episode[row['mode'],row['goal'],row['rep']].append(row)
    results=[]
    for ep in read_rows(folder/'episodes.jsonl'):
        mode,goal,rep = ep['mode'],ep['goal'],ep['rep']
        session = f'vc_{goal}_{mode}_{rep}'
        state=engine.open(session,goal)
        events=Counter()
        for i,row in enumerate(by_episode.pop((mode,goal,rep))):
            page=state['public']
            assert row['page']==page['page_type'] and row['step']==i
            name,arg=parse_action_fixed(row['action'])
            clicked=(arg or '').lower()
            step=engine.step(session,row['action'])
            assert row['legal']==step['legal'] and bool(step['status']['done'])==(i+1==ep['length'])
            state=step['state']
            if not row['legal']:
                events['illegal']+=1
            elif name=='search':
                events['search']+=1
            elif name=='click':
                products={str(p['asin']).lower() for p in page.get('products',[])}
                options={str(v).lower() for vals in page.get('available_options',{}).values() for v in vals}
                if clicked in products: events['open_listed_product']+=1
                elif clicked in options: events['select_visible_option']+=1
                elif clicked=='buy now': events['buy']+=1
                else: events['other_click']+=1
            else:
                events['other_legal']+=1
        st=step['status']
        assert st['done'] and st['termination_reason']==ep['reason'] and st['reward']==ep['reward'] and st.get('reward_valid')==ep['valid']
        assert sum(events.values())==ep['length']
        engine.close(session)
        results.append(dict(mode=mode,goal=goal,rep=rep,actions=events))
    assert not by_episode
    return results

def run_audit(run,watch):
    dest=run/'cpu_replay_checks'
    dest.mkdir(exist_ok=True)
    engine=Engine(run)
    deadline=time.monotonic()+2250
    folders=[run/f'smoke_batch_{i}' for i in (0,60)]+[run/f'formal_batch_{i}' for i in range(0,1380,60)]
    for folder in folders:
        out=dest/(folder.name+'.json')
        if out.exists():continue
        while not folder.is_dir():
            if not watch or time.monotonic()>=deadline:raise RuntimeError(f'Missing completed batch {folder.name}')
            time.sleep(5)
        save(out,audit_batch(engine,folder))
    groups=defaultdict(list)
    for f in folders:
        for ep in json.loads((dest/(f.name+'.json')).read_text()):
            groups[ep['mode']+('_greedy' if ep['rep']==2 else '_sample')].append(ep['actions'])
    report={}
    for label,actions in groups.items():
        totals=Counter()
        for a in actions:totals.update(a)
        report[label]=dict(episodes=len(actions),steps=dict(totals),episodes_with_action={k:sum(k in a for a in actions) for k in totals})
    save(dest/'funnel.json',report)
    print('Verified native terminal outcomes for',sum(map(len,groups.values())),'episodes')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    run_audit(args.run,args.watch)
