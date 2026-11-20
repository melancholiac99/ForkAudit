"""Compare the same detached target and support under the two forward precisions."""
import csv
from pathlib import Path

import numpy as np
import torch
from .s21_data import SmokeBatch, summarize
from .s21_report import merge
from .s22_measure import ARMS, SCALES, RUN, PREVIOUS
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def checkpoint(precision,key):
    src=RUN/precision/key
    manifests=[read(src/f'manifest_{i}.json') for i in range(4)]
    for name in ('weight_sha256','update_norm','seed','precision','checkpoint'):
        assert len({m[name] for m in manifests})==1
    shards=[]
    for i,m in enumerate(manifests):
        p=src/f'logits_{i}.pt'
        assert m['rank']==i and digest_file(p)==m['logits_sha256']
        shards.append(torch.load(p,weights_only=True))
    all_logits=merge(shards)
    assert len(all_logits)==282 and sum(len(v) for v in all_logits.values())==4448
    return all_logits,manifests


def fit_curve(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    design=np.column_stack((np.ones(len(x)),x))
    intercept,slope=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y-design@np.array([intercept,slope])
    sst=((y-y.mean())**2).sum()
    return dict(x=x.tolist(),y=y.tolist(),intercept=float(intercept),slope=float(slope),
                residuals=residual.tolist(),r_squared=float(1-residual@residual/sst) if sst>0 else None)


def gate(measured,null):
    threshold=float(np.quantile(null,.95,method='linear'))
    return dict(alignment=measured,null_q95=threshold,above_by=measured-threshold,passed=bool(measured>threshold))


def tsv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def report():
    batch=SmokeBatch();every=list(range(len(batch.rows)))
    old=read(PREVIOUS/'step_size_diagnostics.json')
    dest=RUN/'report';dest.mkdir(exist_ok=True)
    bases={};lineage={};real=[];null=[];probe=[];scores={};fits={}
    for precision in ('bf16','fp32'):
        bases[precision],lineage[precision+'/initial']=checkpoint(precision,'initial')
    assert lineage['bf16/initial'][0]['weight_sha256']==lineage['fp32/initial'][0]['weight_sha256']
    original=merge([torch.load(PREVIOUS/'zero'/f'logits_{k}.pt',weights_only=True)['before'] for k in range(4)])
    assert all(torch.equal(original[i],bases['bf16'][i]) for i in every)
    def record(a,q,precision,key,kind,scale,seed,desired,actual,end):
        v=summarize(batch,bases[precision],end,q,every)
        u=summarize(batch,bases[precision],end,q,batch.probes)
        metadata=dict(arm=a,precision=precision,kind=kind,multiplier=scale,seed=seed,expected_norm=desired,measured_norm=actual)
        (real if kind=='trained' else null).append({**metadata,**v})
        probe.append({**metadata,**u})
        return v
    for a in ARMS:
        q=batch.targets(a)
        alignments=[];movements=[];null_align=[]
        for mult in SCALES:
            key=f'{a}_{mult}';norm=old['geometry'][str(float(mult))]['norms'][a]
            end,meta=checkpoint('fp32',key);lineage['fp32/'+key]=meta
            v=record(a,q,'fp32',key,'trained',mult,None,norm,meta[0]['update_norm'],end)
            alignments.append(v['alignment']);movements.append(v['movement'])
            archived=merge([torch.load(PREVIOUS/'grid'/key/f'logits_{k}.pt',weights_only=True)['end'] for k in range(4)])
            bv=record(a,q,'bf16',key,'trained',mult,None,norm,norm,archived)
            for metric in ('delta_kl','alignment','movement'):
                assert abs(bv[metric]-old['endpoints'][a][str(float(mult))][metric])<1e-11
        seeds=set()
        for j in range(5):
            key=f'null_{a}_{j}';pair=[]
            for precision in ('bf16','fp32'):
                end,meta=checkpoint(precision,key);lineage[precision+'/'+key]=meta
                pair.append(meta[0]['weight_sha256']);seed=meta[0]['seed'];seeds.add(seed)
                norm=old['geometry']['1.0']['norms'][a]
                assert abs(meta[0]['update_norm']/norm-1)<1e-5
                v=record(a,q,precision,key,'random',1,seed,norm,meta[0]['update_norm'],end)
                if precision=='fp32':null_align.append(v['alignment'])
            assert pair[0]==pair[1]
        assert len(seeds)==5
        scores[a]=gate(alignments[0],null_align)
        fits[a]=dict(alignment_vs_multiplier=fit_curve(SCALES,alignments),
                     movement_vs_multiplier_squared=fit_curve(np.square(SCALES),movements))
    passed=all(v['passed'] for v in scores.values())
    decision=dict(prior_section21_status='NOT_ESTABLISHED',status='RETAIN_LR' if passed else 'PAUSE',
                  selected_lr=1e-6 if passed else None,by_arm=scores,null_per_arm=5,
                  quantile_method='linear empirical q95, strict greater-than',stage_D_updates=0,
                  V_reads=0,T_opd_eval_reads=0)
    tsv(dest/'real_precision_comparison.tsv',real);tsv(dest/'null_controls.tsv',null)
    tsv(dest/'fixed_probe.tsv',probe)
    save(dest/'lr_decision.json',decision);save(dest/'step_size_fits.json',fits)
    save(dest/'measurement_lineage.json',dict(source_run=str(RUN),parent_run=str(PREVIOUS),shards=lineage))
    save(dest/'measurement_checks.json',dict(tasks=32,trajectories=64,rows=len(every),positions=sum(len(r['answer']) for r in batch.rows),
        real_fp32_cells=12,null_paired_directions=len(null)//2,fixed_probe_records=len(batch.probes),
        frozen_targets=True,source_weights_dtype='bfloat16',fp32_compute='FP32 MATH SDPA, TF32 disabled',
        max_identity_residual=max(abs(r['identity_residual']) for r in real+null+probe),
        max_direct_cached_fp32_error=max((e for ms in lineage.values() for m in ms for e in m['probe_direct_max_error'].values()),default=None),
        cosine_source_sha256=digest_file(PREVIOUS/'step_size_diagnostics.json'),
        measured_scoring_worker_gpu_hours=sum(m['seconds'] for ms in lineage.values() for m in ms)/3600))
if __name__=='__main__':report()
