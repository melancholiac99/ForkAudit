"""Complete-case factorial readout and raw-trace integrity checks. No fitting."""
import csv
from pathlib import Path
import numpy as np
from protocol import HERE, ROOT, ARMS, read, write, file_hash, seed_for, repair_action, common
from metrics import outcome_components, bootstrap

CONDS = ['native', 'completed', 'repair', 'repair_completed']

def csv_write(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    d = read(HERE/'design.json')
    out = HERE/'runs/main_6x8'
    manifest = read(out/'run_manifest.json')
    assert manifest['config']['reps'] == 8 and not manifest['config']['dry']
    controls = {(r['point'],r['arm'],r['condition'],r['rep']):r for r in d['controls']}
    records=[]; cells=[]; points=[]; arrays={}; sources=[]; violations=[]
    for p in d['points']:
        arrays[p['id']] = {}
        for cond in CONDS:
            x=np.zeros((8,5))
            for ai, arm in enumerate(ARMS):
                cell=[]
                for rep in range(8):
                    if cond in ['native','completed']:
                        source=controls[(p['id'],arm,cond,rep)]
                        path=ROOT/source['path']; assert file_hash(path)==source['sha256']
                    else:
                        path=out/'results'/f"{p['id']}__{arm}__{cond}__r{rep:03d}.json"
                    r=read(path)
                    assert not r['dry'] and r['status']=='complete' and isinstance(r['won'], bool)
                    assert r['point_id']==p['id'] and r['arm']==arm and r['rep']==rep
                    assert r['forced_action']==p['arms'][arm]['action']
                    ts=r['transitions']; repairs=[t for t in ts if t.get('repair')]
                    generated=[t for t in ts if not t['forced'] and not t.get('repair')]
                    assert len(repairs)<=1 and r['final_episode_step']<=50
                    assert ts[0]['forced'] and not any(t['forced'] for t in ts[1:])
                    assert [t['step'] for t in ts]==list(range(p['turn'],r['final_episode_step']+1))
                    assert r['generated_steps']==len(generated)
                    if cond.startswith('repair'):
                        assert r['run_fingerprint']==manifest['run_fingerprint']
                        assert r['repair_applied']==bool(repairs)
                        for t in generated:
                            assert t['seed']==seed_for(p['index'],arm,rep,t['step']-1)
                        for t in repairs:
                            assert t['action'].startswith(f"move {p['arms'][arm]['repair_object']} to ")
                            assert t['observation_after'].startswith('You move ')
                    cue_expected=cond in ['completed','repair_completed']
                    assert (p['clue']['text'] in ts[0]['shown_observation_after'])==cue_expected
                    assert sum(p['clue']['text'] in t['shown_observation_after'] for t in ts)==int(cue_expected)
                    pickup=f"You pick up the {p['clue']['object']} from "
                    actual_take=any(t['observation_after'].startswith(pickup) for t in ts)
                    row=dict(point=p['id'],old_regime=p['old_regime'],arm=arm,condition=cond,rep=rep,
                        success=int(r['won']),repair_applied=int(bool(repairs)),
                        repair_step=repairs[0]['step'] if repairs else '',
                        target_pickup=int(actual_take),generated_steps=len(generated),
                        generated_tokens=r['generated_tokens'], final_step=r['final_episode_step'],
                        source=str(path.relative_to(ROOT)),source_sha256=file_hash(path))
                    sources.append(dict(path=row['source'],sha256=row['source_sha256']))
                    records.append(row);cell.append(row);x[rep,ai]=int(r['won'])
                cells.append(dict(point=p['id'],old_regime=p['old_regime'],arm=arm,
                    action=p['arms'][arm]['action'],condition=cond,
                    repair_object=p['arms'][arm]['repair_object'],
                    q=float(x[:,ai].mean()),successes=int(x[:,ai].sum()),n=8,
                    repair_rate=float(np.mean([r['repair_applied'] for r in cell])),
                    pickup_rate=float(np.mean([r['target_pickup'] for r in cell]))))
            arrays[p['id']][cond]=x
            m=outcome_components(x)
            points.append(dict(point=p['id'],old_regime=p['old_regime'],condition=cond,**m))
    assert len(records)==960
    contrasts=[]
    for p in d['points']:
        values={c:outcome_components(arrays[p['id']][c]) for c in CONDS}
        row=dict(point=p['id'],old_regime=p['old_regime'])
        for metric in ['mean_success','action_variance']:
            v={c:values[c][metric] for c in CONDS}
            row.update({f'{metric}_info_without_repair':v['completed']-v['native'],
                f'{metric}_info_with_repair':v['repair_completed']-v['repair'],
                f'{metric}_repair_without_info':v['repair']-v['native'],
                f'{metric}_repair_with_info':v['repair_completed']-v['completed'],
                f'{metric}_interaction':v['repair_completed']-v['repair']-v['completed']+v['native']})
        contrasts.append(row)
    groups={}
    for group in ['all','Dead','Fork']:
        chosen=[r for r in contrasts if group=='all' or r['old_regime']==group]
        groups[group]={k:bootstrap([r[k] for r in chosen]) for k in chosen[0] if k not in ['point','old_regime']}
    # No-op branches are a useful fixed control; they receive identical intervention rules.
    noop=[r for r in cells if r['repair_object'] is None]
    noop_summary={c:float(np.mean([r['q'] for r in noop if r['condition']==c])) for c in CONDS}
    summary=dict(status='complete exploratory mechanism experiment',n_points=6,n_control=480,n_new=480,
        point_condition_metrics=points,point_contrasts=contrasts,group_contrasts=groups,
        noop_branch_means=noop_summary,
        new_repair_activation=sum(r['repair_applied'] for r in records if r['condition'].startswith('repair')),
        cost=read(out/'progress.json'),
        caveat='Six purposefully selected cases; old labels unchanged; no population inference. '
               'C is an unclipped unbiased cross-repeat estimate. New and control seeds differ. '
               'Repair includes real one-step/history costs; no learned-policy improvement claimed.')
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    csv_write(dest/'records.csv',records);csv_write(dest/'cells.csv',cells)
    csv_write(dest/'points.csv',points);csv_write(dest/'contrasts.csv',contrasts)
    write(dest/'summary.json',summary)
    write(dest/'integrity.json',dict(records=960,new=480,reused=480,all_checks_passed=True,
        final_step_max=max(r['final_step'] for r in records),sources=sources))
    # A figure retains the full action vectors rather than relying on class flips.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    colors=['#777777','#e69f00','#0072b2','#009e73']
    for ax,p in zip(axes.flat,d['points']):
        for cond,color in zip(CONDS,colors):
            ax.plot(range(5),arrays[p['id']][cond].mean(0),marker='o',label=cond,color=color)
        ax.set_title(f"{p['id']} (old {p['old_regime']})")
        ax.set_xticks(range(5),ARMS);ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
        ax.set_ylabel('Success probability (8 / cell)')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4)
    fig.suptitle('Information and one-time execution repair: all six action profiles',y=1.02)
    fig.tight_layout()
    fig.savefig(dest/'action_profiles.png',dpi=180,bbox_inches='tight')
    fig.savefig(dest/'action_profiles.svg',bbox_inches='tight')
    print('Complete:',summary['cost'],flush=True)
    for row in contrasts: print(row,flush=True)

if __name__=='__main__':
    main()
