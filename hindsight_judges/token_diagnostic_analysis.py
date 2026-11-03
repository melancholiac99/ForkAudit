"""Aggregation and independent artifact checks for historical response diagnostics."""
from pathlib import Path
import gzip,json
from collections import defaultdict
import numpy as np
import exp1c as E
import exp2_common as C
import exp3_token_diagnostic as T

def rank_average(x):
    x=np.asarray(x,float);return np.array([1+sum(x<v)+(sum(x==v)-1)/2 for v in x],float)

def ranking_metrics(reward,q):
    r=np.asarray(reward,float);q=np.asarray(q,float);assert r.shape==q.shape==(5,) and np.isfinite(r).all()
    assert np.ptp(q)>0
    a,b=rank_average(r),rank_average(q)
    rho=float(np.corrcoef(a,b)[0,1]) if np.ptp(a)>0 else None
    pairs=[]
    for i in range(5):
        for j in range(i):
            if q[i]==q[j]:continue
            pairs.append(.5 if r[i]==r[j] else float((r[i]-r[j])*(q[i]-q[j])>0))
    return dict(spearman=rho,pairwise_concordance=float(np.mean(pairs)),
                top_proxy_value_gain=float(q[r==r.max()].mean()-q.mean()),
                factual_minus_alternative_proxy=float(r[0]-r[1:].mean()))

def collect(args,strict=False):
    out=Path(args.out);groups,meta=T.prepared(out,args.limit)
    folder=out/args.name/('full' if not args.limit else f'first{args.limit}')
    run=C.read_json(folder/'run.json');assert run['complete'] and run['points']==len(groups)
    assert run['groups_hash']==meta['groups_hash'] and run['source']==T.source_fingerprint()
    assert C.sha_file(folder/'manifest.json')==run['manifest_sha256']
    man=C.read_json(folder/'manifest.json');assert set(man)=={g['id'] for g in groups}
    gates=C.read_json(folder/'numerical_gate.json')
    assert len(gates)==3 and all(g['passed'] and g['max_tv']<.005 for g in gates)
    cached=out/('student_cache' if not args.limit else f'student_cache_first{args.limit}')
    cm=C.read_json(cached/'run.json');assert cm['complete'] and C.sha_file(cached/'manifest.json')==run['student_cache_manifest_sha256']
    cman=C.read_json(cached/'manifest.json');summaries={};tokens=0;responses=0
    for g in groups:
        p=folder/'records'/(g['id']+'.json.gz');assert C.sha_file(p)==man[g['id']]
        record=json.loads(gzip.decompress(p.read_bytes()));assert record['group_sha256']==g['sha256']
        seen=set();rr={r['arm']:r for r in g['responses']}
        cache=None
        if strict:
            cp=cached/(g['id']+'.npz');assert C.sha_file(cp)==cman[g['id']]['file_sha256']
            cache=np.load(cp,allow_pickle=False)
        for row in record['rows']:
            key=(row['condition'],row['arm']);assert key not in seen;seen.add(key)
            response=rr[row['arm']];assert row['token_ids']==response['ids'] and row['kinds']==response['kinds']
            values={k:np.asarray(v,float) for k,v in row['values'].items()};n=len(row['token_ids'])
            assert set(values)=={'teacher_logp','student_logp','delta','reverse_kl','forward_kl'}
            assert all(len(v)==n and np.isfinite(v).all() for v in values.values())
            assert np.max(abs((values['teacher_logp'].astype(np.float32)-values['student_logp'].astype(np.float32))-values['delta']))<1e-6
            assert min(values['reverse_kl'].min(),values['forward_kl'].min())>=-1e-5
            if strict:
                assert cache['ids'+str(row['arm'])].tolist()==response['ids']
                assert np.max(abs(cache['lp'+str(row['arm'])]-values['student_logp']))<1e-4
            for kind in ('all',)+T.KINDS:
                mask=np.array([kind=='all' or k==kind for k in row['kinds']]);got=row['summary'][kind]
                assert got['n']==int(mask.sum())
                for name,v in values.items():
                    assert abs(float(v[mask].sum())-got[name]['sum'])<1e-9
                    if mask.any():assert abs(float(v[mask].mean())-got[name]['mean'])<1e-9
                    else:assert got[name]['mean'] is None
            summaries[(g['id'],*key)]=row['summary'];tokens+=n;responses+=1
        if cache is not None:cache.close()
        assert seen=={(c,r['arm']) for c in T.CONDS for r in g['responses']}
    assert responses==run['responses']
    return groups,summaries,folder,dict(points=len(groups),responses=responses,scored_token_positions=tokens,
                                         model=args.name,cohort=args.cohort,numerical_gates=gates)

def ci(v,c):return E.mean_ci(v,c,seed=T.SEED,nboot=T.NBOOT)

def calculate(groups,s,cohort):
    metrics={};clusters=[g['cluster'] for g in groups]
    def stat(key,vals):
        keep=[i for i,v in enumerate(vals) if v is not None and np.isfinite(v)]
        metrics[key]=ci([vals[i] for i in keep],[clusters[i] for i in keep])
        metrics[key]['undefined_or_missing']=len(vals)-len(keep)
    for condition in T.CONDS:
        # Per-point averaging first: E4's five responses have equal weight within a state.
        for kind in ('all',)+T.KINDS:
            for field in ('delta','reverse_kl','forward_kl'):
                for agg in ('sum','mean'):
                    vals=[]
                    for g in groups:
                        vv=[s[(g['id'],condition,r['arm'])][kind][field][agg] for r in g['responses']]
                        vals.append(float(np.mean(vv)) if all(v is not None for v in vv) else None)
                    stat(f'{condition}_{kind}_{field}_{agg}',vals)
    for condition in ('L','LN','F'):
        for kind in ('all',)+T.KINDS:
            for agg in ('sum','mean'):
                vals=[]
                for g in groups:
                    diffs=[]
                    for r in g['responses']:
                        a=s[(g['id'],condition,r['arm'])][kind]['delta'][agg]
                        b=s[(g['id'],'M',r['arm'])][kind]['delta'][agg]
                        diffs.append(a-b if a is not None and b is not None else None)
                    vals.append(float(np.mean(diffs)) if all(v is not None for v in diffs) else None)
                stat(f'{condition}_minus_M_{kind}_delta_{agg}',vals)
    if cohort=='p4':
        qgap=np.array([g['q'][0]-np.mean(g['q'][1:]) for g in groups]);bad=qgap<0
        for condition in T.CONDS:
            for agg in ('sum','mean'):
                vals=np.array([s[(g['id'],condition,0)]['all']['delta'][agg] for g in groups])
                for qname,gap in [('q3456_advantage',qgap),('q3456_q12_comparator',np.array([g['q'][0]-g['q'][g['q12_comparator']] for g in groups]))]:
                    metrics[f'{condition}_delta_{agg}_slope_vs_{qname}']=E.slope_ci(gap,vals,clusters,seed=T.SEED,nboot=T.NBOOT)
                keep=[i for i,g in enumerate(groups) if all(v is not None for v in g['qgold'])]
                if len(keep)>3:
                    gold=[groups[i]['qgold'][0]-np.mean(groups[i]['qgold'][1:]) for i in keep]
                    metrics[f'{condition}_delta_{agg}_slope_vs_gold_advantage']=E.slope_ci(gold,vals[keep],[clusters[i] for i in keep],seed=T.SEED,nboot=T.NBOOT)
                metrics[f'{condition}_delta_{agg}_positive_on_bad_factual']=ci((vals[bad]>0).astype(float).tolist(),np.array(clusters)[bad].tolist())
    else:
        perpoint={}
        for condition in T.CONDS:
            for agg in ('sum','mean'):
                rows=[ranking_metrics([s[(g['id'],condition,j)]['all']['delta'][agg] for j in range(5)],g['q']) for g in groups]
                perpoint[(condition,agg)]=rows
                for name in rows[0]:stat(f'{condition}_delta_{agg}_{name}',[v[name] for v in rows])
        for condition in ('L','LN','F'):
            for agg in ('sum','mean'):
                for name in ('spearman','pairwise_concordance','top_proxy_value_gain','factual_minus_alternative_proxy'):
                    vals=[a[name]-b[name] if a[name] is not None and b[name] is not None else None
                          for a,b in zip(perpoint[(condition,agg)],perpoint[('M',agg)])]
                    stat(f'{condition}_minus_M_delta_{agg}_{name}',vals)
    return metrics

def analyze(args):
    groups,s,folder,counts=collect(args)
    metrics=calculate(groups,s,args.cohort)
    result=dict(**counts,bootstrap=dict(seed=T.SEED,nboot=T.NBOOT,cluster='trajectory' if args.cohort=='p4' else 'game'),metrics=metrics,
        interpretation='Historical teacher-forced signal diagnostic; not a measured OPD gradient or training benefit.',
        limitations=['Historical response text was canonically re-encoded; original generated token IDs/probabilities unavailable.',
                     'L/LN share action-dependent factual future across targets; within-group ranking must be compared with independent replay Q.',
                     'Sum delta is a sampled-token density-ratio proxy, not exact reverse-KL loss.',
                     'E4 selection is conditional on nonconstant noisy three-replay values; not a full-population estimate.'])
    C.save_json(folder/'analysis.json',result)
    lines=[result['interpretation'],json.dumps(counts),*result['limitations']]
    for k,v in metrics.items():
        if 'slope_vs_q3456_advantage' in k or 'spearman' in k or k.endswith('minus_M_all_delta_sum'):
            lines.append(k+' '+json.dumps(v))
    (folder/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def verify(args):
    groups,s,folder,counts=collect(args,strict=True)
    analysis=C.read_json(folder/'analysis.json')
    assert all(analysis[k]==v for k,v in counts.items())
    # Independent reconstruction of the central paired L-vs-M full-reply effect.
    vals=np.array([np.mean([s[(g['id'],'L',r['arm'])]['all']['delta']['sum']-
                           s[(g['id'],'M',r['arm'])]['all']['delta']['sum'] for r in g['responses']]) for g in groups])
    names=sorted({g['cluster'] for g in groups});ix={n:i for i,n in enumerate(names)}
    idx=np.array([ix[g['cluster']] for g in groups]);tot=np.bincount(idx,weights=vals);n=np.bincount(idx)
    w=np.random.default_rng(T.SEED).multinomial(len(names),np.full(len(names),1/len(names)),size=T.NBOOT)
    dist=(w@tot)/(w@n);expected=np.percentile(dist,[2.5,97.5])
    got=analysis['metrics']['L_minus_M_all_delta_sum']
    assert abs(got['estimate']-float(vals.mean()))<1e-9 and np.max(abs(np.array(got['ci95'])-expected))<1e-9
    for path,digest in C.read_json(Path(args.out)/'source_hashes.json').items():assert C.sha_file(path)==digest,path
    C.save_json(folder/'verification.json',dict(passed=True,**counts,independent_L_minus_M_check=True,
        token_alignment_and_full_record_integrity=True,original_sources_unchanged=True,analysis_sha256=C.sha_file(folder/'analysis.json')))
    print('VERIFIED',args.cohort,args.name,counts)
