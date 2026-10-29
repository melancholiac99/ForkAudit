"""Read saved teacher probabilities and independent replay outcomes separately."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'exploratory_deep_discovery/opd_opsd_2026_10_29/prepared_v1'
CONDITIONS=['S','H','A','X']


def jl(path): return [json.loads(x) for x in path.read_text().splitlines()]


def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def cluster_ci(rows,key,rng):
    tasks=sorted({r['task_id'] for r in rows})
    totals=np.array([sum(r[key] for r in rows if r['task_id']==t) for t in tasks])
    counts=np.array([sum(r['task_id']==t for r in rows) for t in tasks])
    picks=rng.integers(len(tasks),size=(4000,len(tasks)))
    means=totals[picks].sum(1)/counts[picks].sum(1)
    return dict(n_points=len(rows),n_tasks=len(tasks),mean=float(totals.sum()/counts.sum()),
                ci95=np.quantile(means,[.025,.975]).tolist())


def readout(probs,q):
    v={c:float(probs[c]@q) for c in CONDITIONS}
    v.update(H_minus_S=v['H']-v['S'],A_minus_H=v['A']-v['H'],
             A_minus_X=v['A']-v['X'],X_minus_H=v['X']-v['H'])
    return v


def main():
    prep=json.loads((OUT/'preparation.json').read_text())
    scores=jl(OUT/'scores.jsonl')
    assert len(scores)==prep['contexts']==576
    assert len({r['request_id'] for r in scores})==576
    requests={r['request_id']:r for r in jl(OUT/'requests.jsonl')}
    groups={}
    for s in scores:
        assert s['request_sha256']==requests[s['request_id']]['request_sha256']
        p=np.array(s['candidate_prob'])
        assert p.shape==(5,) and np.isclose(p.sum(),1) and np.all(p>=0)
        groups.setdefault(s['point_id'],{})[s['condition']]=s
    points={r['point_id']:r for r in jl(OLD/'evaluation/points_and_outcomes.jsonl')}
    contexts={r['point_id']:r for r in jl(OLD/'contexts/student_visible.jsonl')}
    five=ROOT/'exploratory_deep_discovery/regime_mechanism_2026_10_28'
    mechanism=json.loads((five/'results/run_manifest.json').read_text())['design']['points']
    mechanism={p['id']:p for p in mechanism}
    blocks=list(csv.DictReader((five/'results/summary_blocks.csv').open()))
    devq={}
    for pid in mechanism:
        rows=[r for r in blocks if r['point_id']==pid]
        assert len(rows)==16
        devq[pid]={condition:np.array([np.mean([int(r[f'a{i}_{condition}']) for r in rows])
                                      for i in range(5)]) for condition in ['native','completed']}
    values=[];behaviors=[];cases=[]
    for pid,sc in groups.items():
        p=points[pid];assert set(sc)==set(CONDITIONS)
        probs={c:np.array(sc[c]['candidate_prob']) for c in CONDITIONS}
        legal=np.array([c['legal_before'] for c in contexts[pid]['candidates']],bool)
        assert legal.sum()>=4
        legal_probs={c:(probs[c]*legal)/(probs[c]*legal).sum() for c in CONDITIONS}
        reg=mechanism[pid]['old_regime'] if p['cohort']=='development' else p['regime']
        base=dict(point_id=pid,task_id=p['task_id'],cohort=p['cohort'],regime=reg,
                  masked=p['masked'],all_legal=bool(legal.all()),task_type=p['task_type'])
        qs={}
        if p['cohort']=='main' and p['primary_value_eligible']:
            qs['bank56']=np.array(p['primary_q'])
        if p['cohort']=='main' and 'gold' in p['banks']:
            qs['gold']=np.array([r['successes']/r['n'] for r in p['banks']['gold']])
        if p['cohort']=='development':qs=devq[pid]
        for pool,q in qs.items():
            v=readout(probs,q);lv=readout(legal_probs,q)
            values.append(dict(**base,pool=pool,**v,**{'legal_'+k:x for k,x in lv.items()},
                               uniform=float(q.mean()),factual=float(q[0])))
        b=dict(**base)
        for c in CONDITIONS:
            b[c+'_chosen_mass']=float(probs[c][0]);b[c+'_illegal_mass']=float(probs[c][~legal].sum())
            b[c+'_candidate_log_mass']=sc[c]['candidate_log_mass']
            b[c+'_entropy']=float(-np.sum(probs[c]*np.log(np.maximum(probs[c],1e-300))))
            b[c+'_argmax']=int(np.argmax(probs[c]))
            if p['cohort']=='development':
                visible=np.array([mechanism[pid]['arms'][f'a{i}']['clue_object_in_feedback'] for i in range(5)],float)
                b[c+'_visible_mass']=float(probs[c]@visible)
        b['A_H_total_variation']=float(abs(probs['A']-probs['H']).sum()/2)
        b['A_X_total_variation']=float(abs(probs['A']-probs['X']).sum()/2)
        b['H_S_total_variation']=float(abs(probs['H']-probs['S']).sum()/2)
        behaviors.append(b)
        cases.append(dict(**base,actions=[c['action'] for c in contexts[pid]['candidates']],
             q={k:v.tolist() for k,v in qs.items()},probabilities={k:v.tolist() for k,v in probs.items()},
             sequence_logp={c:sc[c]['sequence_logp'] for c in CONDITIONS}))
    # CSV has a single schema; no missing outcomes are filled with zeros.
    with (OUT/'values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
    dump('behaviors.json',behaviors);dump('cases.json',cases)
    rng=np.random.default_rng(2026102913)
    summaries={}
    keys=CONDITIONS+['H_minus_S','A_minus_H','A_minus_X','X_minus_H']
    for cohort,pools in [('main',['bank56','gold']),('development',['native','completed'])]:
        for pool in pools:
            for subset in ['all','all_legal','Fork','Indifferent','Dead','Masked']:
                rows=[r for r in values if r['cohort']==cohort and r['pool']==pool and
                    (subset=='all' or (subset=='all_legal' and r['all_legal']) or
                     (subset=='Masked' and r['masked']) or r['regime']==subset)]
                if not rows:continue
                summaries[f'{cohort}/{pool}/{subset}']={k:cluster_ci(rows,k,rng) for k in keys}
                if subset=='all':
                    summaries[f'{cohort}/{pool}/legal_restricted']={
                        k:cluster_ci(rows,'legal_'+k,rng) for k in keys}
    diag={}
    for cohort in ['main','development']:
        for subset in ['all','Fork','Indifferent','Dead']:
            rows=[r for r in behaviors if r['cohort']==cohort and (subset=='all' or r['regime']==subset)]
            if not rows:continue
            ks=[c+'_chosen_mass' for c in CONDITIONS]+['H_S_total_variation','A_H_total_variation','A_X_total_variation']
            if cohort=='development':ks += [c+'_visible_mass' for c in CONDITIONS]
            diag[f'{cohort}/{subset}']={k:cluster_ci(rows,k,rng) for k in ks}
    dump('summary.json',dict(status='exploratory actual frozen-model teacher scoring; no training',
         groups=summaries,behavior=diag,scope='fixed realized thought and five complete candidate commands'))
    input_paths=[OLD/'evaluation/points_and_outcomes.jsonl',OLD/'contexts/student_visible.jsonl',
        five/'results/summary_blocks.csv',five/'results/run_manifest.json',OUT/'scores.jsonl']
    dump('evaluation_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axs,['development/native/Fork','development/completed/Fork','main/bank56/all_legal'],
                            ['Mechanism Fork: native (8 tasks)','Same Fork: completed (8 tasks)','Main: all-legal Live points']):
        g=summaries[key]
        for i,c in enumerate(CONDITIONS):
            s=g[c];lo,hi=np.array(s['ci95'])*100;m=100*s['mean']
            ax.bar(i,m,color=['#777777','#4477aa','#228833','#cc6677'][i],alpha=.85)
            ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],color='black',capsize=3)
        ax.set(xticks=range(4),xticklabels=['Student','Hindsight','Aligned','Shuffled'],
               ylabel='Expected success under candidate rule (%)',title=title,ylim=(0,100))
        ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('Actual teacher action preferences evaluated with existing independent continuations')
    fig.tight_layout();fig.savefig(OUT/'teacher_results.png',dpi=180);fig.savefig(OUT/'teacher_results.svg')
    for key in ['development/native/Fork','development/completed/Fork','main/bank56/all',
                'main/bank56/all_legal','main/gold/all']:
        print(key,json.dumps(summaries.get(key,{})),flush=True)


if __name__=='__main__':main()
