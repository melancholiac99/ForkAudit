"""Publish independent arm milestones and finish after all result and search receipts exist."""
import csv
import json
import statistics
import os
import subprocess
from pathlib import Path
from .e25_paths import PUBLIC,RUN,SERIAL
from exploratory_shop_audit.grpo_forced_audit.common import ROOT,read,save,digest_file

OUT=PUBLIC/'migration_2514'
LOCAL=RUN/'migration_2514'
ARMS=('R-matched','P-frozen','R')
POOLS=('T_opd_eval','V')


def curves(arms,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    def add(arm,k,mon,behavior):
        for pool in POOLS:
            for mode in ('sample','greedy'):
                x=behavior['summary'][f'{arm}/{pool}/{mode}']
                rows.append(dict(arm=arm,step=k,pool=pool,decode=mode,episodes=x['episodes'],
                    successes=round(x['all_requirements']*x['episodes']),rate=x['all_requirements'],kl=mon[arm]['stats']['all']['kl']))
    root=RUN/SERIAL
    add('B',0,read(root/'monitor/000.json'),read(root/'behavior/000.json'))
    for k in (10,20,30):add('P',k,read(root/'monitor'/f'{k:03}.json'),read(root/'behavior'/f'{k:03}.json'))
    for arm in arms:
        for f in sorted((OUT/arm/'behavior').glob('*.json')):
            k=int(f.stem);add(arm,k,read(OUT/arm/'monitor'/f'{k:03}.json'),read(f))
    dest.mkdir(exist_ok=True,parents=True)
    with (dest/'checkpoint_curves.tsv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for mode in ('sample','greedy'):
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharey='row',layout='constrained')
        for row,pool in enumerate(POOLS):
            for col,axis in enumerate(('step','kl')):
                ax=axes[row,col]
                for arm in ('P',*arms):
                    data=[r for r in rows if r['arm'] in ('B',arm) and r['pool']==pool and r['decode']==mode]
                    ax.plot([r[axis] for r in data],[100*r['rate'] for r in data],'o-',label=arm)
                base=next(r for r in rows if r['arm']=='B' and r['pool']==pool and r['decode']==mode)
                ax.axhline(100*base['rate'],ls=':',color='0.4',label='B@0')
                ax.set(xlabel='Training update' if axis=='step' else 'BF16 KL(base || checkpoint)',ylabel=f'{pool} all requirements (%)')
                if row==0 and col==0:ax.legend()
                ax.grid(alpha=.2)
        fig.savefig(dest/f'{mode}_curves.png',dpi=140);plt.close(fig)


def stop_record(baseline,rates,early_stop):
    legal=[dict(update=k,rate=rate,drop_pp=100*(baseline-rate),below_threshold=baseline-rate>=.1-1e-12)
           for k,rate in rates]
    triggers=[right['update'] for left,right in zip(legal,legal[1:]) if left['below_threshold'] and right['below_threshold']]
    assert triggers==([legal[-1]['update']] if early_stop else [])
    return dict(status='EARLY_STOP' if early_stop else 'COMPLETED',baseline_legal_rate=baseline,checkpoint_legal_rates=legal)


def milestone(arm):
    endpoint=read(OUT/arm/'endpoint.json')
    end=endpoint['step']
    for k in range(10,end+1,10):
        for name in ('monitor','behavior','search'):
            assert (OUT/arm/name/f'{k:03}.json').is_file()
    from .e25_layout import Layout
    from .e2513_adopt import verify_state
    l=Layout(SERIAL);receipt=read(l.folder(end,arm)/'optimizer.json')
    assert verify_state(l.state(end,arm),receipt)
    baseline=endpoint['adoption']['initial_training_legal_rate']
    rates=[(k,read(l.folder(k,arm)/'train_statistics.json')['legal_rate']) for k in range(10,end+1,10)]
    vals=[]
    start=11 if arm=='P-frozen' else 6
    for k in range(start,end+1):
        path=l.folder(k,arm)
        vals.append(dict(update=k,optimizer=read(path/'optimizer.json'),measurement=read(path/'measurement.json'),
                         train_statistics=read(path/'train_statistics.json')))
    save(OUT/arm/'training.json',dict(first_new_update=start,terminal_update=end,steps=vals,
        **stop_record(baseline,rates,endpoint['early_stop'])))
    curves([arm],OUT/arm)
    return OUT/arm


def utilization():
    tape=LOCAL/'gpu_utilization.jsonl'
    if not tape.exists():return
    samples=[json.loads(line) for line in tape.read_text().splitlines() if line.endswith(']]')]
    if not samples:return
    cutoff=(OUT/'R-matched/endpoint.json').stat().st_mtime
    summary={}
    for label,group in [('whole_recorded_interval',samples),('before_matched_endpoint',[s for s in samples if s[0]<=cutoff]),
                        ('after_matched_endpoint',[s for s in samples if s[0]>cutoff])]:
        if not group:continue
        by_gpu=[]
        for rank in range(8):
            readings=[s[1][rank] for s in group]
            assert all(r[0]==rank for r in readings)
            pct=[r[1] for r in readings]
            by_gpu.append(dict(rank=rank,mean_utilization_pct=statistics.mean(pct),median_utilization_pct=statistics.median(pct),
                               sample_fraction_at_zero=pct.count(0)/len(pct),peak_mib=max(r[2] for r in readings)))
        summary[label]=dict(samples=len(group),first_unix=group[0][0],last_unix=group[-1][0],gpus=by_gpu)
    save(OUT/'final/gpu_utilization.json',dict(interval_seconds=15,coverage='Polling began during R-matched update 9',periods=summary))


def comparison_table(outcomes):
    def pp(metric):
        low,high=metric['ci95']
        return f"{100*metric['estimate']:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"
    base=read(RUN/SERIAL/'behavior/030.json')
    unseen=base['summary']['P/V_unseen/sample']['tasks']
    seen=base['summary']['P/V_seen/sample']['tasks']
    lines=['','## Paired all-requirements purchase rates','',
        f"Contrasts use the four sampled episodes per task and ASIN-cluster bootstrap intervals. V contains {unseen} unseen-ASIN tasks and {seen} seen-ASIN tasks. P comparisons between trained arms use their last shared checkpoint.",
        '', '| Comparison | Role | Checkpoints (left, right) | T_opd_eval difference, pp [95% CI] | V-unseen difference, pp [95% CI] |',
        '|---|---|---:|---:|---:|']
    checkpoints={'P-R-matched':min(30,outcomes['R-matched']),'P-B':30,'P-R':min(30,outcomes['R']),'R-B':outcomes['R']}
    for pair,k in checkpoints.items():
        left,right=pair.split('-',1)
        arm='R' if pair=='R-B' else right
        record=base if pair=='P-B' else read(OUT/arm/'behavior'/f'{k:03}.json')
        assert record['endpoints'][left]==k and record['endpoints'][right]==(0 if right=='B' else k)
        values=[pp(record['paired'][f'{pair}/sample/asin/{s}']['all_requirements']) for s in ('T_opd_eval','V_unseen')]
        role='Primary' if pair in ('P-R-matched','P-B') else 'Secondary'
        lines.append(f"| {pair} | {role} | {k}, {record['endpoints'][right]} | {values[0]} | {values[1]} |")
    old=read(PUBLIC.parent/'section23_20261120/behavior_report.json')['paired']['P-P-shuffled/sample/asin']
    lines.extend(['',f"At update 10 in stage D, P minus P-shuffled was {pp(old['T_opd_eval']['all_requirements'])} pp on T_opd_eval and {pp(old['V']['all_requirements'])} pp on V. The stage D record includes the R-beta and shuffled controls. [Stage D paired outcomes](../../../section23_20261120/behavior_report.json)."])
    return lines


def final():
    from .e2514_gate import migration_verified
    migration_verified()
    for arm in ARMS:
        assert (OUT/arm/'training.json').is_file()
    needed=('search_selection.json','archived_search_metrics.json','intervention_tasks.json','intervention_result_pages.json',
        'search_swap_report.json','tcal_closing_bracket.json','a5_first_query.json','verify_group_A.json','verify_group_B.json',
        'search_coverage/handoff/coverage.json','search_coverage/final/coverage.json')
    for name in needed:assert (OUT/name).is_file()
    for k in (0,10,20,30):
        with (OUT/'search_archive'/f'checkpoint_{k:03}.tsv').open() as f:
            assert sum(1 for r in csv.DictReader(f,delimiter='\t'))==2000
    curves(ARMS,OUT/'final')
    utilization()
    outcomes={a:read(OUT/a/'endpoint.json')['step'] for a in ARMS}
    report=read(OUT/'search_swap_report.json')['summary'];selection=read(OUT/'search_selection.json')['counts']
    lines=['# Stage E migration endpoints','', 'Single seed. P ended at update 30. P-frozen is a mechanism diagnostic and its first ten updates used the old layout. Updates 1–5 of R and R-matched used the adopted layout.','',
        '| Arm | Status | Terminal update | T_opd_eval sample | V sample | BF16 movement |','|---|---|---:|---:|---:|---:|']
    with (OUT/'final/checkpoint_curves.tsv').open() as f:t=list(csv.DictReader(f,delimiter='\t'))
    for a,k in {'B':0,'P':30,**outcomes}.items():
        rows={x['pool']:x for x in t if x['arm']==a and x['step']==str(k) and x['decode']=='sample'}
        status=read(OUT/a/'training.json')['status'] if a in ARMS else {'B':'Baseline','P':'Stopped at migration'}[a]
        lines.append(f"| {a} | {status} | {k} | {rows['T_opd_eval']['successes']}/800 | {rows['V']['successes']}/800 | {float(rows['V']['kl']):.6f} |")
    for a in ARMS:
        train=read(OUT/a/'training.json')
        if train['status']=='EARLY_STOP':
            lower,upper=train['checkpoint_legal_rates'][-2:]
            lines.extend(['',f"{a} met §25.7. Its baseline training legal-action rate was {100*train['baseline_legal_rate']:.2f}%. The rates at updates {lower['update']} and {upper['update']} were {100*lower['rate']:.2f}% and {100*upper['rate']:.2f}%."])
    lines.extend(comparison_table(outcomes))
    lines.extend(['','## First-query intervention','','Rates below use the same selected tasks on both sides and four fresh completions per distinct query and recipient.','','| Set | Selected | Both sources available | P@30 gain (pp) | P@20 loss (pp) | P@30 recovery fraction |','|---|---:|---:|---:|---:|---:|'])
    for key,v in report.items():
        if not key.endswith('/both_sources'):continue
        ef=v['effects'];frac=ef['P30_recovery_fraction']['estimate']
        lines.append(f"| {key.rsplit('/',1)[0]} | {v['selected_tasks']} | {v['paired_tasks']} | {100*ef['P30_gain_good_minus_own']['estimate']:+.2f} | {100*ef['P20_loss_own_minus_bad']['estimate']:+.2f} | {frac if frac is not None else 'NA'} |")
    lines.extend(['','[Sampling curves](sample_curves.png) · [Greedy curves](greedy_curves.png) · [Checkpoint table](checkpoint_curves.tsv).',
                  '', '## Projector refresh on the P probes','',
                  '| Versions | Cosine | Mean squared difference | All-position BF16 KL |','|---|---:|---:|---:|'])
    for k in (10,20):
        m=read(OUT/'P/monitor'/f'{k:03}.json')['P']
        lines.append(f"| {m['versions'][0]} → {m['versions'][1]} | {m['projector_cosine']:.4f} | {m['projector_mse']:.6f} | {m['stats']['all']['kl']:.6f} |")
    b=read(OUT/'tcal_closing_bracket.json')['overall']
    lines.extend(['','## Search label and teacher trajectory checks','',
        f"On T_cal, the exact closing-bracket candidate occurred at {b['n']} search-argument positions across {b['tasks']} tasks. Its mean content correction was {b['mean'][0]:.4f}. The held-out predictor mean was {b['mean'][1]:.4f}.",
        '', '| A.5 packet | Decoding | First searches / episodes | Mean argument tokens | Queries containing gold ASIN |', '|---|---|---:|---:|---:|'])
    for a in read(OUT/'a5_first_query.json')['results']:
        lines.append(f"| {a['mode']} | {a['decode']} | {a['with_first_search']}/{a['episodes']} | {a['mean_tokens']:.2f} | {a['gold_in_query']} |")
    lines.extend(['','The archive tables distinguish verified action sequences from unrecovered episodes. Argument token counts come from regenerated native tokens or the archived total for an unambiguous single search. Each distinct task/query result page is scored once with the native purchase rules. The fresh sampling report includes task-cluster intervals, both task definitions and coverage.','',
        '[Archived first searches](../search_archive) · [Checkpoint search summaries](../archived_search_metrics.json) · [First result pages](../intervention_result_pages.json) · [Fresh continuation outcomes](../search_swap_report.json) · [Closing bracket by position](../tcal_closing_bracket.json).'])
    extra=read(OUT/'search_archive/crosscheck_at_10.json')
    lines.extend(['', f"The two migration update-10 evaluations independently authenticated {extra['newly_authenticated']} of the {extra['originally_unrecovered']} historical trajectories originally marked as missing. {extra['intervention_source_rows']} were source trials for the sealed intervention. [Additional first searches and token counts](../search_archive/crosscheck_at_10.json)."])
    augmented=OUT/'search_coverage/final'
    more=read(augmented/'coverage.json')
    assert digest_file(augmented/'search_swap_report.json')==more['outcomes_sha256']
    lines.extend(['','## Coverage sensitivity after the migration evaluations','',
        f"Full action digests recovered {more['recovered_original_source_trials']} additional source trials. The expanded analysis includes {more['fresh_episodes_including_prior']} fresh continuations including the original 2,664.",
        '', '| Selected tasks | P@30 recovery | Task-cluster 95% interval |','|---|---:|---:|'])
    for pool in POOLS:
        value=read(augmented/'search_swap_report.json')['summary'][pool+'/all/both_sources']['effects']['P30_recovery_fraction']
        interval=value['ci95']
        ci='NA' if interval is None else f"{interval[0]:.3f} to {interval[1]:.3f}"
        point='NA' if value['estimate'] is None else f"{value['estimate']:.3f}"
        lines.append(f"| {pool} | {point} | {ci} |")
    lines.extend(['','[Expanded fresh-control outcomes](../search_coverage/final/search_swap_report.json) · [Authenticated source records](../search_coverage/final/archived_queries.json) · [Additional sampling at the P-frozen handoff](../search_coverage/handoff/coverage.json).'])
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'final/completion.json',dict(endpoints=outcomes,original_p=30,cpu_checks=needed,task_selection=selection,
         all_public_sha256={str(p.relative_to(OUT)):digest_file(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'final/completion.json'}))


if __name__=='__main__':
    import sys
    if sys.argv[1]=='curves':curves([],OUT/'P')
    elif sys.argv[1]=='final':final()
    else:milestone(sys.argv[1])
