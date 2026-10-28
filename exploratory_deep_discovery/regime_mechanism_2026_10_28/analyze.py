"""Five-arm causal information completion; outcome success and controllability.

Primary: broadcast minus native action variance within original Fork stratum,
with mean success. Quadratic quantities use distinct-repeat U-statistics.
Same-budget G-test profiles are descriptive, never updated frozen labels.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from metrics import bootstrap, completion_effect, outcome_components
from protocol import ARMS, CONDITIONS, read, write
from run import result_key, validate_record


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def g_stat(counts, n):
    s=np.asarray(counts,float); f=n-s
    p=s.sum(axis=-1,keepdims=True)/(s.shape[-1]*n)
    return 2*np.sum(xlogy(s,s/np.maximum(n*p,1e-20))+
                    xlogy(f,f/np.maximum(n*(1-p),1e-20)),axis=-1)


def profile(x, seed):
    n,k=x.shape
    assert np.all((x==0)|(x==1))
    counts=x.sum(0).astype(int); total=int(counts.sum())
    if total in [0,n*k]:
        return dict(profile='Constant-failure' if total==0 else 'Constant-success',p_het=1.)
    rng=np.random.default_rng(seed)
    null=rng.multivariate_hypergeometric([n]*k,total,size=10000)
    p=(1+np.sum(g_stat(null,n)>=g_stat(counts,n)-1e-12))/10001
    return dict(profile='Detected-Fork' if p<.10 else 'Live-no-detected-Fork',p_het=float(p))


def scenarios(y0,y1,selective):
    one=y0.copy(); one[:,selective]=y1[:,selective]
    return {'native':y0,'broadcast':y1,'selective':one,'lottery20':.8*y0+.2*y1}


def collect(out,partial):
    m=read(out/'run_manifest.json'); points=m['design']['points']
    points=[p for p in points if p['id'] in m['config']['point_ids']]
    rows=[]; arms=[]; block_rows=[]; profiles=[]
    complete=0; reused=0; steps=0; tokens=0
    arrays={}
    for p in points:
        blocks=[]
        for rep in range(m['config']['reps']):
            records={}
            for ai,a in enumerate(ARMS):
                for ci,c in enumerate(CONDITIONS):
                    key=result_key(p,a,c,rep); path=out/'results'/f'{key}.json'
                    if not path.exists():
                        continue
                    r=read(path); validate_record(r,m['run_fingerprint'],key)
                    assert r['dry']==m['config']['dry']
                    records[ai,ci]=r
                    complete+=1; reused+=int('source_record' in r)
                    steps+=r['generated_steps']; tokens+=r['generated_tokens']
            if len(records)==10:
                block=np.array([[int(records[ai,ci]['won']) for ai in range(5)] for ci in range(2)])
                blocks.append(block)
                row=dict(point_id=p['id'],old_regime=p['old_regime'],rep=rep)
                row.update({f'{a}_{c}':int(records[ai,ci]['won'])
                            for ai,a in enumerate(ARMS) for ci,c in enumerate(CONDITIONS)})
                block_rows.append(row)
        if len(blocks)<2:
            continue
        blocks=np.asarray(blocks,float); y0,y1=blocks[:,0,:],blocks[:,1,:]
        arrays[p['id']]=(p,y0,y1)
        sc=scenarios(y0,y1,ARMS.index(p['selective_arm']))
        row=dict(point_id=p['id'],task_id=p['task_id'],task_type=p['task_type'],
                 old_regime=p['old_regime'],repeats=len(blocks),selective_arm=p['selective_arm'])
        row.update(completion_effect(y0,y1))
        for name,x in sc.items():
            row.update({f'{name}_{k}':v for k,v in outcome_components(x).items()})
            if name!='lottery20':
                pr=profile(x,20261028+p['index'])
                profiles.append(dict(point_id=p['id'],old_regime=p['old_regime'],condition=name,**pr))
        row['selective_minus_lottery_C']=row['selective_action_variance']-row['lottery20_action_variance']
        row['selective_minus_lottery_success']=row['selective_mean_success']-row['lottery20_mean_success']
        row['broadcast_minus_native_rms_plugin']=float(np.std(y1.mean(0))-np.std(y0.mean(0)))
        row['native_target_OR']=int(any(p['arms'][a]['target_word_in_feedback'] for a in ARMS))
        row['broadcast_target_OR']=1
        rows.append(row)
        for ai,a in enumerate(ARMS):
            arms.append(dict(point_id=p['id'],old_regime=p['old_regime'],arm=a,
                action=p['arms'][a]['action'],native_q=float(y0[:,ai].mean()),
                completed_q=float(y1[:,ai].mean()),lift=float((y1[:,ai]-y0[:,ai]).mean()),
                target_word_in_feedback=p['arms'][a]['target_word_in_feedback'],
                clue_object_in_feedback=p['arms'][a]['clue_object_in_feedback'],
                clue=p['clue']['text'],source=p['arms'][a]['action']==p['source_action']))
    expected=len(points)*10*m['config']['reps']
    if complete!=expected and not partial:
        raise ValueError(f'Incomplete {complete}/{expected}; --allow-partial uses separate outputs')
    return m,rows,arms,block_rows,profiles,arrays,dict(complete=complete,expected=expected,
        reused=reused,new=complete-reused,generated_steps=steps,generated_tokens=tokens)


def make_plot(rows,arms,summary,path,dry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Dead':'#777777','Indifferent':'#2879ad','Fork':'#d77521'}
    fig,axs=plt.subplots(1,3,figsize=(14,4.3))
    for regime in colors:
        subset=[r for r in rows if r['old_regime']==regime]
        for r in subset:
            xs=[r['native_mean_success'],r['broadcast_mean_success']]
            ys=[r['native_action_variance'],r['broadcast_action_variance']]
            axs[0].annotate('',xy=(xs[1],ys[1]),xytext=(xs[0],ys[0]),
                            arrowprops=dict(arrowstyle='->',color=colors[regime],alpha=.45,lw=1))
        axs[0].scatter([r['native_mean_success'] for r in subset],
                       [r['native_action_variance'] for r in subset],s=20,
                       color=colors[regime],label=f'Old {regime}')
    axs[0].axhline(0,color='gray',lw=.6)
    endpoint_y=[r[k] for r in rows for k in ['native_action_variance','broadcast_action_variance']]
    axs[0].set_xlim(-.04,1.04)
    axs[0].set_ylim(min(endpoint_y)-.01,max(endpoint_y)+.01)
    axs[0].set(xlabel='Uniform five-arm success probability',ylabel='Unbiased action variance C',
               title='Native → shared information')
    axs[0].legend(fontsize=8)
    for i,regime in enumerate(colors):
        g=summary['groups'].get(regime)
        if not g:
            continue
        for offset,key,label,alpha in [(-.17,'native_action_variance','Native',.5),
                                       (.17,'broadcast_action_variance','Shared',1)]:
            v=g[key];lo,hi=v['ci95']
            axs[1].bar(i+offset,v['mean'],width=.32,color=colors[regime],alpha=alpha,
                       label=label if i==0 else None)
            axs[1].errorbar(i+offset,v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],
                            fmt='none',color='black',capsize=3)
    axs[1].axhline(0,color='gray',lw=.6);axs[1].set_xticks(range(3),list(colors))
    axs[1].set(ylabel='Unbiased C (task bootstrap 95% CI)',title='Keep old strata fixed')
    if axs[1].get_legend_handles_labels()[0]:
        axs[1].legend(fontsize=8)
    g=summary['groups']['all']
    names=['native','lottery20','selective','broadcast']
    for i,name in enumerate(names):
        v=g[f'{name}_action_variance'];lo,hi=v['ci95']
        axs[2].errorbar(i,v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',capsize=4)
    axs[2].axhline(0,color='gray',lw=.6)
    axs[2].set_xticks(range(4),['Native','20% lottery','One fixed arm','All arms'],rotation=18)
    axs[2].set(ylabel='Unbiased C',title='Allocation scenarios from measured cells')
    fig.suptitle('CPU DRY RUN — NOT SCIENTIFIC RESULTS' if dry else
                 'Exploratory: can information alter five-arm credit structure?')
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=190)
    fig.savefig(path.with_suffix('.svg'));plt.close(fig)

    # All selected cases, including reversals and physical/execution failures.
    fig,axes=plt.subplots(int(np.ceil(len(rows)/4)),4,figsize=(14,2.5*np.ceil(len(rows)/4)),squeeze=False)
    for ax,row in zip(axes.flat,rows):
        ar=[a for a in arms if a['point_id']==row['point_id']]
        ax.plot(range(5),[a['native_q'] for a in ar],'-o',label='Native',color='#4477aa')
        ax.plot(range(5),[a['completed_q'] for a in ar],'-o',label='Shared clue',color='#ee7733')
        ax.set_title(row['point_id']+' · old '+row['old_regime'],fontsize=9)
        ax.set_xticks(range(5),['factual','alt1','alt2','alt3','alt4'],fontsize=7)
        ax.set_ylim(-.05,1.05);ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('All five-arm success profiles (16 repeats per cell; exploratory)' if not dry
                 else 'CPU DRY RUN — NOT SCIENTIFIC RESULTS')
    fig.tight_layout()
    fig.savefig(path.parent/(path.stem+'_all_profiles.png'),dpi=170)
    fig.savefig(path.parent/(path.stem+'_all_profiles.svg'))
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run_dir',type=Path)
    ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--no-plot',action='store_true')
    args=ap.parse_args()
    m,rows,arms,blocks,profiles,arrays,cost=collect(args.run_dir,args.allow_partial)
    if not rows:
        print('No complete five-arm blocks with at least two repeats yet.');return
    assert len({r['task_id'] for r in rows})==len(rows)
    fields=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k!='repeats']
    subsets={'all':rows,**{g:[r for r in rows if r['old_regime']==g] for g in ['Dead','Indifferent','Fork']}}
    groups={g:{f:bootstrap([r[f] for r in rs]) for f in fields} for g,rs in subsets.items() if rs}
    transitions={}
    for name in ['broadcast','selective']:
        table={}
        for r in rows:
            p0=next(p['profile'] for p in profiles if p['point_id']==r['point_id'] and p['condition']=='native')
            p1=next(p['profile'] for p in profiles if p['point_id']==r['point_id'] and p['condition']==name)
            key=p0+' → '+p1;table[key]=table.get(key,0)+1
        transitions[name]=table
    summary=dict(status='CPU DRY RUN' if m['config']['dry'] else 'exploratory',
        complete=cost['complete']==cost['expected'],run_fingerprint=m['run_fingerprint'],cost=cost,
        estimands='Uniform original five-arm mean success and Var_a(q); unbiased quadratic estimates; '
                  'task bootstrap preserves all conditions and arms.',
        primary_group='Fork (original frozen label)',groups=groups,descriptive_transitions=transitions,
        limitations=['Enriched sample, not prevalence.',
          'Selective and lottery are composed from measured cells, not extra independent experiments.',
          'New profiles use 16/arm and a one-stage G test; not the original frozen two-stage labels.',
          'Non-rejection does not establish equality; negative unbiased C estimates are retained.',
          'Location clues need not eliminate physical or downstream execution differences.'])
    stem='summary' if summary['complete'] else 'partial_summary'
    write(args.run_dir/f'{stem}.json',summary)
    csv_write(args.run_dir/f'{stem}_points.csv',rows);csv_write(args.run_dir/f'{stem}_arms.csv',arms)
    csv_write(args.run_dir/f'{stem}_blocks.csv',blocks);csv_write(args.run_dir/f'{stem}_profiles.csv',profiles)
    def cell(v,scale=1):
        lo,hi=v['ci95'];return f"{scale*v['mean']:.4f} [{scale*lo:.4f}, {scale*hi:.4f}]"
    lines=['# Five-arm information completion — '+summary['status'],'',
        f"Records {cost['complete']}/{cost['expected']}; reused {cost['reused']}; new {cost['new']}.",'',
        '| Old stratum | n | Native success | Shared success | Native C | Shared C | C reduction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g,v in groups.items():
        lines.append(f"| {g} | {v['native_C']['n']} | "+' | '.join(cell(v[k]) for k in
          ['native_mean_success','broadcast_mean_success','native_C','completed_C','C_reduction'])+' |')
    lines+=['','C1-C0 = 2 Cov(q0, lift) + Var(lift). Same-repeat noise is removed.',
            'Frozen historical labels are unchanged. Profiles are descriptive, not proof of zero effect.','',
            '## Information allocation (same collected cells)','',
            '| Scenario | Mean success | C |','|---|---:|---:|']
    for name in ['native','lottery20','selective','broadcast']:
        lines.append(f"| {name} | {cell(groups['all'][name+'_mean_success'])} | "
                     f"{cell(groups['all'][name+'_action_variance'])} |")
    lines+=['','Selective: clue only on the metadata-fixed arm. Lottery: clue with probability 0.2 '
             'independently of action. Both have expected injection frequency 1/5 under uniform arms.',
             'These are derived intervention protocols; no additional independent rollouts were run.',
             '', '## Same-budget profiles (descriptive)', '',json.dumps(transitions,indent=2),
             '', '## Every arm', '', '| Point | Old regime | Arm | Action | Native q | Shared q |',
             '|---|---|---|---|---:|---:|']
    for a in arms:
        lines.append(f"| {a['point_id']} | {a['old_regime']} | {a['arm']} | {a['action']} | "
                     f"{a['native_q']:.3f} | {a['completed_q']:.3f} |")
    (args.run_dir/f'{stem}.md').write_text('\n'.join(lines)+'\n')
    if not args.no_plot:
        make_plot(rows,arms,summary,args.run_dir/f'{stem}_figure',m['config']['dry'])
    print('\n'.join(lines[:12]))


if __name__=='__main__':
    main()
