"""Publish independent arm milestones and finish after all result and search receipts exist."""
import csv
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


def milestone(arm):
    end=read(OUT/arm/'endpoint.json')['step']
    for k in range(10,end+1,10):
        for name in ('monitor','behavior','search'):
            assert (OUT/arm/name/f'{k:03}.json').is_file()
    from .e25_layout import Layout
    from .e2513_adopt import verify_state
    l=Layout(SERIAL);receipt=read(l.folder(end,arm)/'optimizer.json')
    assert verify_state(l.state(end,arm),receipt)
    vals=[]
    start=11 if arm=='P-frozen' else 6
    for k in range(start,end+1):
        path=l.folder(k,arm)
        vals.append(dict(update=k,optimizer=read(path/'optimizer.json'),measurement=read(path/'measurement.json'),
                         train_statistics=read(path/'train_statistics.json')))
    save(OUT/arm/'training.json',dict(first_new_update=start,steps=vals))
    curves([arm],OUT/arm)
    return OUT/arm


def final():
    from .e2514_gate import migration_verified
    migration_verified()
    for arm in ARMS:
        assert (OUT/arm/'training.json').is_file()
    needed=('search_selection.json','archived_search_metrics.json','intervention_tasks.json','intervention_result_pages.json',
        'search_swap_report.json','tcal_closing_bracket.json','a5_first_query.json','verify_group_A.json','verify_group_B.json')
    for name in needed:assert (OUT/name).is_file()
    for k in (0,10,20,30):
        with (OUT/'search_archive'/f'checkpoint_{k:03}.tsv').open() as f:
            assert sum(1 for r in csv.DictReader(f,delimiter='\t'))==2000
    curves(ARMS,OUT/'final')
    outcomes={a:read(OUT/a/'endpoint.json')['step'] for a in ARMS}
    report=read(OUT/'search_swap_report.json')['summary'];selection=read(OUT/'search_selection.json')['counts']
    lines=['# Stage E migration endpoints','', 'Single seed. P ended at update 30. P-frozen is a mechanism diagnostic and its first ten updates used the old layout. Updates 1–5 of R and R-matched used the adopted layout.','',
        '| Arm | Terminal update | T_opd_eval sample | V sample | BF16 movement |','|---|---:|---:|---:|---:|']
    with (OUT/'final/checkpoint_curves.tsv').open() as f:t=list(csv.DictReader(f,delimiter='\t'))
    for a,k in {'B':0,'P':30,**outcomes}.items():
        rows={x['pool']:x for x in t if x['arm']==a and x['step']==str(k) and x['decode']=='sample'}
        lines.append(f"| {a} | {k} | {rows['T_opd_eval']['successes']}/800 | {rows['V']['successes']}/800 | {float(rows['V']['kl']):.6f} |")
    lines.extend(['','## First-query intervention','','Rates below use the same selected tasks on both sides and four fresh completions per distinct query and recipient.','','| Set | Selected | Both sources available | P@30 gain (pp) | P@20 loss (pp) | P@30 recovery fraction |','|---|---:|---:|---:|---:|---:|'])
    for key,v in report.items():
        if not key.endswith('/both_sources'):continue
        ef=v['effects'];frac=ef['P30_recovery_fraction']['estimate']
        lines.append(f"| {key.rsplit('/',1)[0]} | {v['selected_tasks']} | {v['paired_tasks']} | {100*ef['P30_gain_good_minus_own']['estimate']:+.2f} | {100*ef['P20_loss_own_minus_bad']['estimate']:+.2f} | {frac if frac is not None else 'NA'} |")
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'final/completion.json',dict(endpoints=outcomes,original_p=30,cpu_checks=needed,task_selection=selection,
         all_public_sha256={str(p.relative_to(OUT)):digest_file(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'final/completion.json'}))


if __name__=='__main__':
    import sys
    if sys.argv[1]=='curves':curves([],OUT/'P')
    elif sys.argv[1]=='final':final()
    else:milestone(sys.argv[1])
