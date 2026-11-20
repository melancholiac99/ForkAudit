"""Pair the real pilot cells, summarize costs and stop before packet selection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from exploratory_shop_audit.grpo_forced_audit.common import save, read_rows, read
from .packets import POLICIES, MATCH
from .section16 import behavior_result

def stats(arr):
    if not len(arr):return dict(n=0)
    x=np.asarray(arr,dtype=float)
    if not np.isfinite(x).all():raise ValueError('nonfinite measurement')
    return dict(n=len(x),mean=float(x.mean()),q10_q50_q90=np.quantile(x,[.1,.5,.9]).tolist())

def paired(goals,d):
    x=np.array([d[g] for g in goals])
    rng=np.random.default_rng(20261120)
    ci=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975])
    return dict(estimate=float(x.mean()),ci95=ci.tolist(),clusters=len(x))

def report(run,formal=False):
    dest=run/'formal_report' if formal else run
    dest.mkdir(exist_ok=True)
    tasks=read(run/'asin_split_manifest.json')['T_pre'][:100 if formal else 8]
    goals=[t['goal_index'] for t in tasks]
    assert len(set(t['asin'] for t in tasks))==len(tasks)
    revisions={(r['goal'],r['rep'],r['step']):r['classes'] for r in read_rows(run/'parser_repair.jsonl')}
    eps={}; steps=[]; tape={}; seen=set()
    for folder in sorted(run.glob('smoke_batch_*'))+(sorted(run.glob('formal_batch_*')) if formal else []):
        for e in read_rows(folder/'episodes.jsonl'):
            key=(e['goal'],e['rep'],e['mode'])
            assert key not in eps
            eps[key]=e
        steps.extend(read_rows(folder/'steps.jsonl'))
        for r in read_rows(folder/'score_inputs.jsonl'):
            tape[r['goal'],r['rep'],r['step']]=r
    assert set(eps)=={(g,r,m) for g in goals for r in range(3) for m in POLICIES}
    success=('gold_purchase','valid_alternative_purchase')
    behavior={}
    for mode in POLICIES:
        for decode,reps in [('sample',(0,1)),('greedy',(2,))]:
            rows=[eps[g,r,mode] for g in goals for r in reps]
            ss=[s for s in steps if s['mode']==mode and s['rep'] in reps]
            group=dict(n=len(rows),reward=stats([e['reward'] for e in rows]),all_requirements=np.mean([e['reason'] in success for e in rows]),strict=np.mean([e['reason']==success[0] for e in rows]),reward_types=dict(Counter(e['reason'] for e in rows)),valid_actions=np.mean([s['legal'] for s in ss]),steps=len(ss),packet_status=dict(Counter(s['packet_status'] or 'available' for s in ss)),pages=dict(Counter(s['page'] for s in ss)),actions=dict(Counter(s['action'].split('[')[0] for s in ss)))
            for ref in ['none']+(['placeholder'] if mode=='full' else []):
                diff={g:np.mean([int(eps[g,r,mode]['reason'] in success)-int(eps[g,r,ref]['reason'] in success) for r in reps]) for g in goals}
                group['all_requirements_vs_'+ref]=paired(goals,diff)
            behavior[mode+'_'+decode]=group
    save(dest/'preflight_utility_report.json',behavior)
    bins=defaultdict(lambda:defaultdict(list));coverage=defaultdict(set); template=defaultdict(list)
    empty=[];match=defaultdict(list);counts=defaultdict(list)
    eligible=defaultdict(list); shifted=defaultdict(set)
    for p in sorted(run.glob('scores_*.jsonl'))+(sorted(run.glob('formal_scores_*.jsonl')) if formal else []):
        with p.open() as f:
            for line in f:
                row=json.loads(line);key=row['goal'],row['rep'],row['step']
                original=tape[key]
                assert key not in seen and row['actual']==original['response'] and row['classes']==original['classes']
                if key in revisions:
                    row['classes']=revisions[key]
                    assert len(row['classes'])==len(row['actual'])
                seen.add(key);empty.append(row['empty_max'])
                for m,view in original['views'].items():
                    if view['counts']:
                        counts[m+'/total'].append(view['length'])
                        for field,n in view['counts'].items():counts[m+'/'+field].append(n)
                for i,c in enumerate(row['classes']):
                    if row['rep']==2:match[c].append(row['greedy_match'][i])
                    for m,o in row['scores'].items():
                        k='/'.join([m,row['page'],c]);coverage[k].add(row['goal'])
                        for metric in ['mass','delta_y','d_y','kl','content_d_y','content_kl']:
                            if metric in o:bins[k][metric].append(o[metric][i])
                        if m in MATCH and MATCH[m] in row['scores']:
                            ref=row['scores'][MATCH[m]]
                            if c in ('click','option') and min(row['scores']['none']['mass'][i],o['mass'][i],ref['mass'][i])>=.25:
                                hit=abs(o['content_d_y'][i])>.02 and o['content_kl'][i]>1e-5
                                eligible[m].append((row['goal'],hit))
                                if hit:shifted[m].add(row['goal'])
                            template[k].append(o['kl'][i]-ref['kl'][i])
                            if m!='full' and 'placeholder' in row['scores']:
                                template[k+'/vs_common_placeholder_KL_difference'].append(o['kl'][i]-row['scores']['placeholder']['kl'][i])
    assert seen==set(tape) and seen
    metrics={}
    for k,vals in bins.items():
        summary={m:stats(v) for m,v in vals.items()}
        for m in ('delta_y','d_y','content_d_y'):
            if m in vals:summary[m]['above_0.02']=float((np.abs(vals[m])>.02).mean())
        metrics[k]=dict(tasks=len(coverage[k]),metrics=summary)
    save(dest/'privilege_delta_report.json',metrics)
    save(dest/'template_effect_report.json',{k:stats(v) for k,v in template.items()})
    save(dest/'preflight_content_coverage.json',{m:dict(positions=len(eligible[m]),tasks=len({g for g,h in eligible[m]}),positions_above_threshold=sum(h for g,h in eligible[m]),tasks_above_threshold=len(shifted[m])) for m in MATCH})
    save(dest/'packet_token_counts.json',{k:stats(v) for k,v in counts.items()})
    save(dest/'score_alignment.json',dict(scored_steps=len(seen),response_tokens=sum(len(t['response']) for t in tape.values()),max_empty_error=max(empty),greedy_hf_vllm_agreement={k:stats(v) for k,v in match.items()}))
    cost=sum(e['gpu_hours'] for e in read_rows(run/'preflight_cost_ledger.jsonl'))
    if formal:
        decisions={}; passed=[]
        for m in MATCH:
            items=eligible[m]; lift=behavior[m+'_sample']['all_requirements_vs_none']
            cover=len(items)>=100 and len({g for g,h in items})>=20
            signal=cover and np.mean([h for g,h in items])>=.1 and len(shifted[m])>=10
            verdict='inconclusive'
            behavior_gate=behavior_result(lift['estimate'],lift['ci95'])
            if cover and (not signal or behavior_gate=='no_improvement'):verdict='negative'
            if signal and behavior_gate=='improved':
                verdict='eligible';passed.append(m)
            decisions[m]=dict(status=verdict,behavior_gate=behavior_gate,positions=len(items),tasks=len({g for g,h in items}),shifted_tasks=len(shifted[m]),hits=sum(h for g,h in items),lift=lift)
        winner=max(passed,key=lambda m:decisions[m]['lift']['estimate']) if passed else None
        status='SELECTED' if winner else ('PREFLIGHT_INCONCLUSIVE' if any(x['status']=='inconclusive' for x in decisions.values()) else 'PRIVILEGE_NOT_USEFUL')
        save(dest/'packet_selection.json',dict(status=status,selected=winner,decisions=decisions,templates=read(run/'teacher_packet_manifest.json'),pre_hash=read(run/'splits_hash.json')['T_pre']))
    else:
        save(dest/'packet_selection.json',dict(status='SMOKE_ONLY_AWAITING_FORMAL_BUDGET',selected=None))
    totals=dict(tasks=len(tasks),episodes=len(eps),steps=len(steps),student_steps=len(seen),gpu_hours=cost)
    if not formal:totals['remaining_92_linear_gpu_hours']=cost*92/8
    save(dest/('summary.json' if formal else 'smoke_summary.json'),totals)
    print(read(dest/('summary.json' if formal else 'smoke_summary.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);report(p.parse_args().run)
