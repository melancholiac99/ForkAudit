"""Point-first trajectory-cluster analysis and independent raw-score verification."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import exp1c as E
import exp2_common as C
import exp5_token_slot as S
import token_slot_backend as K

SEED=20261106;NBOOT=4000
PHASES=('all','pre_k','at_or_after_k','no_k')


@lru_cache(maxsize=128)
def design(clusters):
    names=sorted(set(clusters));index=np.array([names.index(c) for c in clusters])
    w=np.random.default_rng(SEED).multinomial(len(names),[1/len(names)]*len(names),size=NBOOT)
    return index,w,len(names)


def ci(values,clusters):
    good=[i for i,v in enumerate(values) if v is not None and np.isfinite(v)]
    if not good:return dict(n=0,n_trajectories=0,estimate=None,ci95=None)
    v=np.array([values[i] for i in good]);labels=tuple(clusters[i] for i in good);ix,w,n=design(labels)
    sums=np.bincount(ix,weights=v,minlength=n);counts=np.bincount(ix,minlength=n)
    dist=(w@sums)/(w@counts)
    return dict(n=len(v),n_trajectories=n,estimate=float(v.mean()),ci95=np.percentile(dist,[2.5,97.5]).tolist())


def slope_ci(x,y,clusters):
    x=np.array(x,float);y=np.array(y,float)
    if len(x)<3 or np.var(x)<1e-12:return None
    ix,w,n=design(tuple(clusters))
    summaries=np.array([np.bincount(ix,weights=z,minlength=n) for z in (np.ones(len(x)),x,y,x*x,x*y)])
    totals=summaries.sum(1);boot=w@summaries.T
    def slopes(a):
        count,sx,sy,sxx,sxy=a.T;den=sxx-sx*sx/count;num=sxy-sx*sy/count
        return np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-12)
    dist=slopes(boot);valid=np.isfinite(dist)
    return dict(estimate=float(slopes(totals[None,:])[0]),ci95=np.percentile(dist[valid],[2.5,97.5]).tolist() if valid.any() else None,
                undefined_draws=int((~valid).sum()))


def avg(v):return float(np.mean(v)) if len(v) else None


def prob(ll):
    x=np.asarray(ll,float);p=np.exp(x-x.max());return p/p.sum()


def load(args):
    groups,meta=S.prepared(args.out,args.limit);dest=S.folder(args);run=C.read_json(dest/'run.json')
    assert run['complete'] and run['n']==len(groups) and run['fingerprint']['source']==S.fingerprint()
    assert C.sha_file(dest/'manifest.json')==run['manifest_sha256'] and C.sha_file(dest/'numerical_gate.json')==run['gate_sha256']
    gates=C.read_json(dest/'numerical_gate.json');assert gates and all(x['passed'] and x['tv']<.005 for x in gates)
    man=C.read_json(dest/'manifest.json');assert set(man)=={g['id'] for g in groups}
    records=[]
    for g in groups:
        path=dest/'records'/(g['id']+'.json.gz');assert C.sha_file(path)==man[g['id']]
        r=S.read(path);assert r['group_sha256']==g['sha256'] and r['fingerprint_sha256']==K.digest(run['fingerprint'])
        assert r['q']==g['q'] and r['k']==S.stable_k(r['commitment_curve'])
        assert r['positions']==S.native_positions(g,r['k'])
        records.append(r)
    return groups,records,dest,meta,run,gates


def calculate(records):
    clusters=[r['cluster'] for r in records];probe={};native={};evidence={};paired={}
    names=list(next(n for r in records for n in r['native'])['targets'])
    for phase in PHASES:
        for name in names:
            point_values=[];factual=[]
            for r in records:
                rows=[p for p in r['probes'] if phase=='all' or p['phase']==phase]
                pp=[prob([a['sequence_logp'][name] for a in p['arms']]) for p in rows]
                point_values.append(avg([float(p@np.asarray(r['q'])*100) for p in pp]));factual.append(avg([p[0] for p in pp]))
            probe[f'{phase}_{name}_value_pp']=ci(point_values,clusters)
            probe[f'{phase}_{name}_p_f']=ci(factual,clusters)
        for reference in (K.BASELINE,'T0','T3'):
            diffs=[]
            for r in records:
                vals=[]
                for p in r['probes']:
                    if phase!='all' and p['phase']!=phase:continue
                    a=prob([arm['sequence_logp'][K.MAIN] for arm in p['arms']]);b=prob([arm['sequence_logp'][reference] for arm in p['arms']])
                    vals.append(float((a-b)@np.asarray(r['q'])*100))
                diffs.append(avg(vals))
            paired[f'{phase}_T2_minus_{reference}_value_pp']=ci(diffs,clusters)
        for slot in ('A','E'):
            vv=[]
            for r in records:
                q=np.asarray(r['q']);q=q-q.mean();den=float(q@q);slopes=[]
                for p in r['probes']:
                    if phase!='all' and p['phase']!=phase:continue
                    delta=np.array([a['raw_sequence_logp'][slot+'F']-a['raw_sequence_logp'][slot+'M'] for a in p['arms']]);delta-=delta.mean()
                    if den:slopes.append(float(q@delta/den))
                vv.append(avg(slopes))
            evidence[f'{phase}_{slot}_within_point_centered_slope']=ci(vv,clusters)
        for kind in ('all','think','action','other'):
            selected=[[p for p in r['native'] if (phase=='all' or p['phase']==phase) and (kind=='all' or p['kind']==kind)] for r in records]
            for name in names:
                for field in ('logp','proxy','logp_minus_AF','kl_target_AF','kl_AF_target','tv_AF','actual_token_rank'):
                    values=[avg([p['targets'][name][field] for p in rows]) for rows in selected]
                    native[f'{phase}_{kind}_{name}_{field}']=ci(values,clusters)
                for lexical in K.RESTART_STRINGS:
                    native[f'{phase}_{kind}_{name}_{lexical}_mass']=ci([avg([p['targets'][name]['lexical_mass'][lexical] for p in rr]) for rr in selected],clusters)
                vals=[avg([p['targets'][name]['proxy'] for p in rr]) for rr in selected];keep=[i for i,v in enumerate(vals) if v is not None]
                if len(keep)>2:
                    gap=[records[i]['q'][0]-np.mean(records[i]['q'][1:]) for i in keep]
                    native[f'{phase}_{kind}_{name}_proxy_slope_vs_qadvantage']=slope_ci(gap,[vals[i] for i in keep],[clusters[i] for i in keep])
            for field in ('logp','kl_target_AF','kl_AF_target'):
                paired[f'{phase}_{kind}_T2_minus_T1_{field}']=ci([avg([p['targets'][K.MAIN][field]-p['targets'][K.BASELINE][field] for p in rr]) for rr in selected],clusters)
    gain=paired['pre_k_T2_minus_T1_b1_value_pp'];lp=paired['all_think_T2_minus_T1_logp']
    kls=[paired['all_think_T2_minus_T1_'+x] for x in ('kl_target_AF','kl_AF_target')]
    enough=gain['n']>=30 and gain['n_trajectories']>=10
    candidate=bool(enough and gain['estimate']>=3 and gain['ci95'][0]>0 and lp['ci95'] and lp['ci95'][0]>=-.1
                   and all(x['ci95'] and x['ci95'][1]<=0 for x in kls))
    return dict(points=len(records),clusters=len(set(clusters)),no_k=sum(r['k'] is None for r in records),
        no_k_fraction=sum(r['k'] is None for r in records)/len(records),
        probe=probe,native=native,paired=paired,evidence=evidence,
        decision=dict(enough_pre_k=enough,training_design_candidate=candidate,main_gain=gain,native_logp_guard=lp,
                      native_kl_guards=kls,rule='>=30 pre-k points/10 clusters; gain>=3pp, lower CI>0; both KL upper CIs<=0; logp lower CI>=-0.1nat',
                      conclusion='Minimum gate met; not demonstrated training benefit.' if candidate else 'Do not claim T2 superiority or training readiness under the fixed gate.'),
        bootstrap=dict(draws=NBOOT,seed=SEED,unit='trajectory after within-point averaging'),
        limitations=['Native positions and counterfactual action probes are separate estimands.',
                     'Sampled token density ratio is not a measured OPD gradient.',
                     'Lexical restart token sets are incomplete and can match legitimate continuations.',
                     'CIs exploratory without multiplicity correction; no-k excluded only from pre/post strata.'])


def analyze(args):
    groups,records,dest,meta,run,gates=load(args);result=calculate(records)
    result.update(source=S.fingerprint(),name=args.name,pilot=bool(args.limit),gates=gates,
        manifest_sha256=run['manifest_sha256'],runtime=run['fingerprint']['runtime'],
        seconds=sum(r['seconds'] for r in records),native_positions=sum(len(r['native']) for r in records),
        probe_positions=sum(len(r['probes']) for r in records),model_calls=sum(r['model_calls'] for r in records))
    if args.limit:result['decision']['conclusion']='Pilot only: correctness/timing; no scientific decision.'
    C.save_json(dest/'analysis.json',result)
    lines=['P4 CAUSAL TOKEN-SLOT DIAGNOSTIC',str({k:result[k] for k in ('name','pilot','points','no_k','seconds','native_positions','probe_positions')}),str(result['decision'])]
    for key,value in result['paired'].items():lines.append(key+' '+str(value))
    (dest/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:3]))


def verify(args):
    groups,records,dest,meta,run,gates=load(args);analysis=C.read_json(dest/'analysis.json')
    assert analysis['source']==S.fingerprint() and analysis['manifest_sha256']==run['manifest_sha256']
    # Every stored multi-token action sum must reconstruct from token-level normalized targets.
    values=[];labels=[]
    for g,r in zip(groups,records):
        for p in r['native']:
            assert p['y']==g['response']['ids'][p['t']]
            for target,z in p['targets'].items():
                assert np.isfinite([z['logp'],z['proxy'],z['kl_target_AF'],z['kl_AF_target']]).all()
                assert abs(z['proxy']-(z['logp']-p['student_logp']))<1e-6
        before=[]
        for p in r['probes']:
            for arm in p['arms']:
                for name,total in arm['sequence_logp'].items():
                    assert abs(total-sum(t['target_logp'][name] for t in arm['tokens']))<1e-8
                for t in arm['tokens']:
                    raw=t['raw_logp'];d=(raw['EF']-raw['EM'])-(raw['AF']-raw['AM'])
                    for name,lp in t['target_logp'].items():
                        if name=='T0':expected=raw['AF']
                        elif name=='T3':expected=raw['EF']
                        else:
                            bits=name.split('_');beta=float(bits[1][1:])
                            if bits[0]=='T1':change=raw['AF']-raw['AM']
                            else:
                                change=max(-5,min(5,d)) if bits[2]=='clip5' else d
                                if bits[3]=='top128' and not t['top128_contains_y']:change=0.
                            expected=raw['AF']+beta*change-t['log_normalizers'][name]
                        assert abs(expected-lp)<1e-4,(name,expected,lp)
            if p['phase']=='pre_k':
                a=prob([x['sequence_logp'][K.MAIN] for x in p['arms']]);b=prob([x['sequence_logp'][K.BASELINE] for x in p['arms']])
                before.append(float(np.dot(a-b,r['q'])*100))
        if before:values.append(sum(before)/len(before));labels.append(r['cluster'])
        for request in r['requests'].values():
            # All native/probe inputs carry an explicit consumed token prefix and no unconsumed target field.
            assert 'y' not in request and 'target' not in request
            assert request['slot'] in ('A','E') and request['evidence'] in ('M','F')
    reported=analysis['paired']['pre_k_T2_minus_T1_b1_value_pp']
    if values:
        names=sorted(set(labels));ix=np.array([names.index(c) for c in labels]);sums=np.bincount(ix,weights=values);counts=np.bincount(ix)
        w=np.random.default_rng(SEED).multinomial(len(names),[1/len(names)]*len(names),size=NBOOT)
        dist=(w@sums)/(w@counts)
        assert abs(reported['estimate']-np.mean(values))<1e-9
        assert np.allclose(reported['ci95'],np.percentile(dist,[2.5,97.5]),atol=1e-9,rtol=0)
    else:assert reported['n']==0
    # Independent NumPy full-vocabulary normalization audit at deterministic saved positions.
    audited=0
    for path in sorted((dest/'vocab_audits').glob('*.npz')):
        arrays=np.load(path);af,am,ef,em=[arrays[k].astype(np.float64) for k in ('AF','AM','EF','EM')]
        for lp in (af,am,ef,em):assert abs(np.exp(lp).sum()-1)<1e-5
        d=np.clip((ef-em)-(af-am),-5,5);z=af+d;m=z.max();lp=z-(m+np.log(np.exp(z-m).sum()));y=int(arrays['y'])
        rid=path.name.split('_native')[0].split('_probe')[0];r=next(r for r in records if r['id']==rid)
        got=r['native'][0]['targets'][K.MAIN]['logp'] if '_native' in path.name else r['probes'][0]['arms'][0]['tokens'][0]['target_logp'][K.MAIN]
        assert abs(float(lp[y])-got)<1e-4;audited+=1
    assert audited>=2
    C.save_json(dest/'verification.json',dict(passed=True,source=S.fingerprint(),points=len(records),
        normalized_action_sequences_reconstructed=True,independent_primary_bootstrap=True,vocabulary_audits=audited,
        analysis_sha256=C.sha_file(dest/'analysis.json'),manifest_sha256=run['manifest_sha256']))
    print('VERIFIED',args.name,len(records),'points')
