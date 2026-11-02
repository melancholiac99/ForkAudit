"""Independent reconstruction of identity/value statistics and decision flags."""
import argparse
import json
from pathlib import Path
import numpy as np
import exp1 as E
import execution_checks as X


def statistic(values,tids):
    values=np.asarray(values,float);tids=np.asarray(tids);valid=np.isfinite(values)
    v=values[valid];t=tids[valid]
    if not len(v):return dict(n=0,estimate=None,ci95=None)
    unique=sorted(set(t));n=len(unique)
    sums=np.array([sum(v[t==g]) for g in unique]);counts=np.array([sum(t==g) for g in unique])
    weights=np.random.default_rng(20261102).multinomial(n,[1/n]*n,size=4000)
    dist=np.array([float(np.dot(w,sums)/np.dot(w,counts)) for w in weights])
    return dict(n=len(v),n_trajectories=n,estimate=float(sum(v)/len(v)),ci95=np.quantile(dist,[.025,.975]).tolist())


def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'/{i}')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert np.isfinite(a) and np.isfinite(b) and abs(a-b)<1e-10,(path,a,b)
    else:assert a==b,(path,a,b)


def verify(repo,out,arms,name):
    jobs,meta,man=E.rows_and_meta(out);target=out/name;run=E.load_json(target/'run.json')
    assert run['complete'] and run['contexts']==3860 and run['scores_sha256']==E.digest(target/'scores.jsonl')
    assert run['hardware_sha256']==X.fp_sha(run['runtime'])
    X.check_attention_run(run,E.digest(E.HERE/'EXECUTION_AMENDMENT_20261102.txt'))
    original=E.load_json(repo/f'exploratory_P4/e2_{name}/run.json')
    assert run['model_sha256']==original['model_sha256'] and run['revision']==original['revision']
    assert run['backend_sha256']==original['scorer_sha256']
    scores=X.load_scores(target/'scores.jsonl',jobs,run['hardware_sha256'],complete=True)
    X.check_sessions(target,scores,jobs,run['hardware_sha256'])
    rebuilt,pointmeta,hashes=E.build_dataset(repo,arms)
    assert rebuilt==jobs and pointmeta==meta and hashes==man['input_sha256']
    data=E.read_parquet(repo,arms);q={}
    for i in range(len(data['traj_id'])):
        pid=f'{data["traj_id"][i]}_t{int(data["turn"][i]):02d}'
        q.setdefault(pid,{})[int(data['j'][i])]={k:data[k][i] for k in ('q_12','q_3456','q_gold')}
    result=E.load_json(target/'analysis.json');assert result['points']==386
    reconstructed={};checked=0
    for ctx in ('S','F'):
        for group in ('all','I_all','U_all','q12_factual_worse','q12_factual_worse_ge_1_3',
                      'I_q12_factual_worse_ge_1_3','U_q12_factual_worse_ge_1_3',
                      'q12_factual_not_worse','gold_q12_factual_worse','gold_q12_factual_worse_ge_1_3'):
            metrics={k:[] for k in E.PP_MEASURES+E.NAT_MEASURES};tids=[]
            for m in meta:
                pid=m['point_id'];alt=m['swap_arm'];gap=m['q12_gap']
                best=sorted(range(1,5),key=lambda j:(-q[pid][j]['q_12'],j))[0]
                assert alt==best and abs(gap-(q[pid][alt]['q_12']-q[pid][0]['q_12']))<1e-12
                if group[:2] in ('I_','U_') and m['layer']!=group[0]:continue
                if group.endswith('ge_1_3') and gap<1/3-1e-8:continue
                if group in ('q12_factual_worse','gold_q12_factual_worse') and gap<=0:continue
                if group=='q12_factual_not_worse' and gap>0:continue
                which='q_gold' if group.startswith('gold_') else 'q_3456'
                value=[q[pid][j][which] for j in range(5)]
                if any(x is None or not np.isfinite(x) for x in value):
                    assert which=='q_gold';continue
                p={};lp={}
                for c in ('0','E','X','PE','PX'):
                    row=scores[pid+'__'+ctx+c];lp[c]=np.array(row['sequence_logp'])
                    raw=np.exp(lp[c]-lp[c].max());p[c]=raw/raw.sum()
                identity_nat=.5*(lp['E'][0]-lp['E'][alt]-lp['X'][0]+lp['X'][alt])
                placebo_nat=.5*(lp['PE'][0]-lp['PE'][alt]-lp['PX'][0]+lp['PX'][alt])
                identity=.5*(p['E'][0]-p['X'][0]+p['X'][alt]-p['E'][alt])
                placebo=.5*(p['PE'][0]-p['PX'][0]+p['PX'][alt]-p['PE'][alt])
                v={c:sum(p[c][j]*value[j] for j in range(5)) for c in p}
                fields=dict(p_exec_true_minus_swap=p['E'][0]-p['X'][0],p_alt_swap_minus_true=p['X'][alt]-p['E'][alt],
                    identity_switch_effect=identity,placebo_switch_effect=placebo,identity_minus_placebo=identity-placebo,
                    value_true_minus_swap=v['E']-v['X'],value_true_minus_no_label=v['E']-v['0'],value_swap_minus_no_label=v['X']-v['0'],
                    exec_argmax_no_label=float(np.argmax(p['0'])==0),exec_argmax_true_label=float(np.argmax(p['E'])==0),
                    exec_argmax_swap_label=float(np.argmax(p['X'])==0),identity_switch_logodds=identity_nat,
                    placebo_switch_logodds=placebo_nat,identity_minus_placebo_logodds=identity_nat-placebo_nat,
                    identity_switch_logodds_clip10=min(10,max(-10,identity_nat)))
                for k,x in fields.items():metrics[k].append(float(x))
                tids.append(m['traj'])
            stats={k:statistic(v,tids) for k,v in metrics.items()}
            stats['identity_switch_logodds']['median']=float(np.median(metrics['identity_switch_logodds'])) if tids else None
            compare(stats,result['metrics'][ctx+'/'+group],ctx+'/'+group);checked+=len(stats)
            reconstructed[ctx+'/'+group]=stats
    assert set(reconstructed)==set(result['metrics'])
    f=reconstructed['F/all'];bad=reconstructed['F/q12_factual_worse_ge_1_3']
    def pos(x):return x['ci95'] is not None and x['ci95'][0]>0
    def neg(x):return x['ci95'] is not None and x['ci95'][1]<0
    decision=dict(primary_model=name=='32b',identity_logodds_positive=pos(f['identity_switch_logodds']),
        identity_probability_positive=pos(f['identity_switch_effect']),value_harm_vs_no_label=neg(bad['value_true_minus_no_label']),
        value_harm_vs_swapped_label=neg(bad['value_true_minus_swap']),bootstrap_seed=20261102,nboot=4000)
    compare(decision,result['decisions'])
    assert f['identity_switch_logodds']['n']==386
    old={r['request_id']:r for r in map(json.loads,(repo/f'exploratory_P4/e2_{name}/scores.jsonl').read_text().splitlines())}
    for ctx in ('S','F'):
        tv=[];agree=[]
        for m in meta:
            pid=m['point_id'];a=np.array(old[pid+'__'+ctx]['candidate_prob']);b=np.array(scores[pid+'__'+ctx+'0']['candidate_prob'])
            tv.append(float(sum(abs(a-b))/2));agree.append(a.argmax()==b.argmax())
        expected={'n':len(tv),'mean_tv':float(np.mean(tv)),'share_tv_gt_0.005':float(np.mean(np.array(tv)>.005)),
                  'argmax_agreement':float(np.mean(agree))}
        compare(expected,result['cross_run_stability'][ctx+'0_vs_original_'+ctx])
    E.write_json(target/'verification.json',dict(passed=True,model=name,points=386,contexts=3860,
                 statistics_independently_verified=checked,decisions=decision,scores_sha256=run['scores_sha256'],
                 analysis_sha256=E.digest(target/'analysis.json'),verifier_sha256=E.digest(__file__)))
    print('Independent verification passed',name,checked,'statistics',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arms',type=Path,required=True);p.add_argument('--name',choices=['32b','7b'],required=True);a=p.parse_args()
    verify(a.repo,a.out,a.arms,a.name)
