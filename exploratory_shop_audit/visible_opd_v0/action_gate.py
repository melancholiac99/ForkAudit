"""Post hoc §17 audit on immutable per-response A.5 scores."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from exploratory_shop_audit.data import digest_file,write_once
from .packets import MATCH

REGISTRATION='54794325cdf4dbf264e9f540b8f0d8450eb4bc50'

def run_audit(parent, dest):
    old=json.loads((parent/'formal_report/packet_selection.json').read_text())
    hashes=json.loads((parent/'formal_report/input_hashes.json').read_text())
    data=defaultdict(list)
    seen=set()
    paths=sorted([*parent.glob('scores_*.jsonl'),*parent.glob('formal_scores_*.jsonl')])
    assert len(paths)==8
    for p in paths:
        assert digest_file(p)==hashes[p.name]
        for line in p.open():
            r=json.loads(line); key=(r['goal'],r['rep'],r['step'])
            assert key not in seen
            seen.add(key)
            for c in ('click','option'):
                inds=[i for i,v in enumerate(r['classes']) if v==c]
                if not inds:continue
                assert not {'click','option'}<=set(r['classes'])
                for m,ref in MATCH.items():
                    scores=r['scores']; valid=[];hits=0
                    if m in scores and ref in scores:
                        for i in inds:
                            if min(scores[g]['mass'][i] for g in ('none',m,ref))>=.25:
                                valid.append(i)
                                hits+=int(abs(scores[m]['content_d_y'][i])>.02 and scores[m]['content_kl'][i]>1e-5)
                    data[m,c].append((key[0],len(inds),len(valid),hits))
    assert len(seen)==1290
    def stats(items):
        use=[x for x in items if x[2]]
        a=len(use); n=sum(x[2] for x in use); hits=sum(x[3] for x in use)
        return dict(total_actions=len(items),eligible_actions=a,eligible_tokens=n,token_hits=hits,
                    eligible_tasks=len({x[0] for x in use}),hit_actions=sum(x[3]>0 for x in use),
                    hit_tasks=len({x[0] for x in use if x[3]}),multi_hit_actions=sum(x[3]>1 for x in use),
                    mean_tokens=sum(x[1] for x in items)/len(items) if items else None,
                    one_hit_per_action_conditional_ceiling=a/n if n else None,
                    token_counts=dict(Counter(x[1] for x in items)),eligible_token_counts=dict(Counter(x[2] for x in items)),
                    hits_per_eligible_action=dict(Counter(x[3] for x in use)))
    decisions={}; eligible=[]
    for m in MATCH:
        by={c:stats(data[m,c]) for c in ('click','option')}
        s=stats(data[m,'click']+data[m,'option'])
        previous=old['decisions'][m]
        assert (s['token_hits'],s['eligible_tokens'])==(previous['hits'],previous['positions'])
        passed=(s['eligible_tokens']>=100 and s['eligible_tasks']>=20 and s['hit_actions']>=.1*s['eligible_actions'] and s['hit_tasks']>=10)
        if passed and previous['behavior_gate']=='improved' and m!='evidence':eligible.append(m)
        decisions[m]=dict(overall=s,by_class=by,content_gate=passed,behavior_gate=previous['behavior_gate'],lift=previous['lift'])
    winner=max(eligible,key=lambda m:decisions[m]['lift']['estimate'],default=None)
    result=dict(prereg_commit=REGISTRATION,post_hoc=True,gpu_hours=0,original_verdict=old['status'],original_sha256=digest_file(parent/'formal_report/packet_selection.json'),selected=winner,placeholder=MATCH.get(winner),status='SELECTED' if winner else 'PRIVILEGE_NOT_USEFUL',decisions=decisions,evidence_limit='未构成有效证据包测试',structural_limit='Conditional on at most one hit per action. Multiple divergent or shifted tokens are possible; tokenization alone gives no universal ceiling below 100%.',scores={p.name:hashes[p.name] for p in paths})
    dest.mkdir(parents=True,exist_ok=True)
    write_once(dest/'packet_selection_v2.json',result)
    print(result['status'],winner,[(m,d['overall']['hit_actions'],d['overall']['eligible_actions']) for m,d in decisions.items()])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('dest',type=Path);a=p.parse_args();run_audit(a.parent,a.dest)
