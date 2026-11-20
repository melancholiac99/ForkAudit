"""Read-only summaries for the paired learning-rate grid and confirmation smoke."""
import argparse
import hashlib
import itertools
import zipfile

import numpy as np
import torch
from .s21_data import SmokeBatch, select_multiplier, summarize
from .s21_store import RUN, ORIGINAL, ARMS, GRID, unpack_member
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, digest_value, read, save


def merge(shards):
    result={}
    for part in shards:
        assert not result.keys() & part.keys()
        result.update(part)
    return result


def load(path):
    return torch.load(path,weights_only=True,map_location='cpu')


def adapter_geometry(paths,states,initial):
    dots=np.zeros((len(ARMS),len(ARMS)),dtype=np.float64)
    hashes={a:hashlib.sha256() for a in ARMS}
    zips=[zipfile.ZipFile(p) for p in paths]
    try:
        for name,b in initial.items():
            vectors=[]
            for arm,z in zip(ARMS,zips):
                arr=unpack_member(z,name,initial)
                hashes[arm].update(name.encode());hashes[arm].update(arr.tobytes())
                vectors.append(torch.from_numpy(arr).double().reshape(-1)-b.double().reshape(-1))
            mat=torch.stack(vectors)
            dots+=(mat@mat.T).numpy()
    finally:
        for z in zips:z.close()
    for a,h in hashes.items():assert h.hexdigest()==states[a]
    norms=np.sqrt(dots.diagonal())
    return dict(norms=dict(zip(ARMS,norms.tolist())),pairwise_cosines={
        ARMS[i]+' / '+ARMS[j]:float(dots[i,j]/(norms[i]*norms[j])) if norms[i]*norms[j]>0 else None
        for i,j in itertools.combinations(range(len(ARMS)),2)})


def original(batch,initial):
    p=RUN/'original'
    summary={};paths=[];states={}
    for a in ARMS:
        info=read(p/f'arm_{a}.json')
        old=read(ORIGINAL/f'arm_{a}.json')
        scores=[]
        for phase in ('before','after'):
            local=[load(p/f'{a}_{phase}_{k}.pt') for k in range(4)]
            scores.append({j:local[k][i] for j,(_,k,i) in enumerate(batch.items)})
        vals=summarize(batch,*scores,batch.targets(a),range(len(batch.rows)))
        assert abs(vals['kl_before']-info['before']['task_weighted_kl'])<1e-11
        assert abs(vals['kl_after']-info['after']['task_weighted_kl'])<1e-11
        summary[a]=dict(decomposition=vals,original_before=old['before']['task_weighted_kl'],
            original_after=old['after']['task_weighted_kl'],gradient_norm=info['gradient_norm'],
            original_gradient_norm=old['gradient_norm'],initial_hash_match=old['initial_adapter_sha256']==info['initial_adapter_sha256'],
            final_hash_match=info['matches_original_final_hash'])
        paths.append(p/f'{a}.zip');states[a]=info['final_adapter_sha256']
    save(RUN/'original_replay_report.json',dict(arms=summary,geometry=adapter_geometry(paths,states,initial)))


def run(mode):
    batch=SmokeBatch()
    initial=load(RUN/'original/initial.pt')
    if mode=='original':
        original(batch,initial)
        return
    zero_files=[load(RUN/'zero'/f'logits_{k}.pt') for k in range(4)]
    start=merge([s['before'] for s in zero_files]);repeat=merge([s['after'] for s in zero_files])
    every=list(range(len(batch.rows)))
    assert sorted(start)==sorted(repeat)==every
    unchanged=all(torch.equal(start[i],repeat[i]) for i in every)
    targets={a:batch.targets(a) for a in ARMS}
    endpoints={a:{} for a in ARMS};intermediate={};geometry={}
    for a in ARMS:
        v=summarize(batch,start,repeat,targets[a],every)
        v['error_envelope']=max(abs(v[k]) for k in ('delta_kl','alignment','movement','identity_residual'))
        endpoints[a]['0.0']=v
    scales=GRID[1:] if mode=='grid' else (read(RUN/'selection.json')['selected_multiplier'],)
    for scale in scales:
        label=str(scale)
        paths=[];states={};intermediate[label]={}
        for a in ARMS:
            p=RUN/mode/f'{a}_{scale:g}'
            info=read(p/'complete.json');assert info['minibatch_sha256']==digest_value(batch.batches)
            assert info['grad_norms'] and min(info['grad_norms'])>0 and info['optimizer_steps']==5
            files=[load(p/f'logits_{k}.pt') for k in range(4)]
            end=merge([f['end'] for f in files])
            assert sorted(end)==every
            endpoints[a][label]=summarize(batch,start,end,targets[a],every)
            paths.append(p/'adapter.zip');states[a]=info['final_sha256']
            steps=[];probes=[merge([f['probes'][i] for f in files]) for i in range(len(batch.batches)+1)]
            for i,sub in enumerate(batch.batches):
                before,after=[merge([f['steps'][i][j] for f in files]) for j in range(2)]
                assert set(sub)==set(before)==set(after)
                if i==0:assert all(torch.equal(before[j],start[j]) for j in sub)
                assert set(probes[i])==set(probes[i+1])==set(batch.probes)
                steps.append(dict(optimizer_step=i+1,rows=len(sub),gradient_norm=info['grad_norms'][i],
                    fixed_minibatch=summarize(batch,before,after,targets[a],sub),
                    fixed_probe=summarize(batch,probes[i],probes[i+1],targets[a],batch.probes)))
            intermediate[label][a]=steps
            if mode=='rerun':
                previous=RUN/'grid'/f'{a}_{scale:g}'
                baseline=[load(previous/f'logits_{k}.pt')['end'] for k in range(4)]
                equivalent=all(torch.equal(x,baseline[k][j]) for k,f in enumerate(files) for j,x in f['end'].items())
                endpoints[a][label].update(grid_final_adapter_match=states[a]==read(previous/'complete.json')['final_sha256'],
                    grid_final_logits_match=equivalent)
        geometry[label]=adapter_geometry(paths,states,initial)
    geometry['0.0']=dict(norms={a:0. for a in ARMS},
        pairwise_cosines={a+' / '+b:None for a,b in itertools.combinations(ARMS,2)})
    report=dict(endpoints=endpoints,geometry=geometry,optimization_steps=intermediate,
        grid=([0.]+list(scales)),zero_logits_identical=unchanged,seed=20261120,rows=len(every),positions=sum(len(r['answer']) for r in batch.rows),
        task_weighting='equal tasks > equal trajectories > equal response positions',batches=list(map(len,batch.batches)),
        minibatch_sha256=digest_value(batch.batches),fixed_probe_records=len(batch.probes),probe_sha256=digest_value([batch.keys[i] for i in batch.probes]),
        V_reads=0,T_opd_eval_reads=0)
    save(RUN/('step_size_diagnostics.json' if mode=='grid' else 'new_smoke_report.json'),report)
    if mode=='grid':
        chosen=select_multiplier(endpoints)
        save(RUN/'selection.json',dict(selected_multiplier=chosen,learning_rate=None if chosen is None else chosen*1e-6,
            rule='Try 1, 1/2, 1/4, 1/8; require all four weighted alignments strictly above their measured zero-scale error.',
            status='SELECTED' if chosen is not None else 'NO_ELIGIBLE_SCALE',
            error={a:v['0.0']['error_envelope'] for a,v in endpoints.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['original','grid','rerun'])
    run(p.parse_args().mode)
