"""Sequential four-card §26 preflight, with separate clocks and a training-only budget."""
import json
import os
import statistics
import subprocess
import sys
import time

from .gap_score import PUBLIC, RUN
from .e25_paths import RUN as STAGE_E
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save

PREFIX='exploratory_shop_audit.visible_opd_v0.'


def run_phase(label,cmd):
    receipt=RUN/'timing'/f'{label}.json'
    if receipt.exists():
        return
    start=time.monotonic()
    with (RUN/f'{label}.log').open('a') as log:
        subprocess.run(cmd,stdout=log,stderr=log,check=True)
    save(receipt,dict(seconds=time.monotonic()-start,command=cmd))


def multi(module,*args):
    return ['torchrun','--standalone','--nproc-per-node=4','-m',PREFIX+module,*args]


def summarize():
    wall=lambda name:read(RUN/'timing'/f'{name}.json')['seconds']
    scored=lambda p: [read(RUN/p/f'score_{i}.json') for i in range(4)]
    first=scored('initial'); hist=scored('timing_refresh')
    history_steps=5
    stage=STAGE_E/'formal_2512'
    # Source clocks for identical native sampler, LoRA trainer and four-rank probe only.
    costs={'collect':[],'train':[],'probe':[]}
    for k in range(10,31):
        for a in ('P','R'):
            loc=stage/'pilot'/f'{k:02}'/a
            parts=[loc/'native'/f'rank{r}'/'complete.json' for r in range(4)]
            if all(p.exists() for p in parts):
                costs['collect'].append(max(read(p)['seconds'] for p in parts))
            for name,key in [('train','train'),('common','probe')]:
                src=stage/'stages'/a/f'{k:03}_{name}.json'
                if src.exists():
                    costs[key].append(read(src)['elapsed'])
    page=[read(RUN/'page_timing'/f'rank{i}'/'timing.json') for i in range(4)]
    assert all(p['completed']>0 for p in page)
    # Measure free interactions separately; the common-page benchmark has four sampled + greedy.
    common=wall('pages')
    init_cost=wall('initial_fit')+wall('initial_calibrate')
    refit=wall('refresh_fit')+wall('refresh_calibrate')
    two=max(p['seconds'] for p in hist)/history_steps
    counts=[min(k,20) for k in range(5,50,5)]
    # These measured clocks include the same mixture of search, click, option and other prefixes.
    # Direct actor nomination is necessary on 4/5 steps, measured separately in the fixed-C check.
    check=[read(RUN/'checks'/f'rank{i}.json') for i in range(4)]
    extra=read(RUN/'actor_timing.json')
    scorer_start=extra['startup_seconds']
    per_step_score=two+scorer_start+.8*extra['current_forward_per_update_seconds']
    page_setup=max(v['start_seconds']+v['close_seconds'] for v in page)
    common_one=page_setup+max(v['continuation_seconds']*50/8 for v in page)
    # Free eval is timed from sampled+greedy unconstrained interaction in this preflight.
    free_one=page_setup+max(v['free_seconds']*50/8 for v in page)
    updates=100
    secs=dict(initial_scorer=max(p['seconds'] for p in first)+scorer_start,
              initial_fit=init_cost,training_sampling=updates*statistics.median(costs['collect']),
              training_scoring=updates*per_step_score,
              optimizer=updates*statistics.median(costs['train']),
              refit_scoring=sum(counts)*two+len(counts)*scorer_start,
              refit_training=sum(refit*k/history_steps for k in counts),
              common_probe=10*statistics.median(costs['probe']),
              free_behavior=(2+2*(4+3))*free_one,common_behavior=3*2*common_one)
    # Teacher-forced full-action mass is measured separately with the same public pages.
    mass=read(RUN/'click_timing.json')
    secs['legal_click_mass']=6*(mass['startup_seconds']+mass['wall_for_200_tasks'])
    total=sum(secs.values())
    save(PUBLIC/'preflight_budget.json',dict(
        measured_tcal=dict(tasks=sum(len(s['goals']) for s in first),records=sum(s['records'] for s in first),
                           seconds=max(s['seconds'] for s in first),fit_with_calibration_seconds=init_cost),
        measured_refresh=dict(source='Stage E P@10 timing proxy, archived steps 6–10, next block ASIN excluded',
                              tasks=sum(len(s['goals']) for s in hist),records=sum(s['records'] for s in hist),
                              positions=sum(s['positions'] for s in hist),rescore_seconds=max(s['seconds'] for s in hist),
                              warm_fit_with_calibration_seconds=refit),native_page_timing=page,
        step_actor_nomination=extra,click_mass_timing=mass,archived_stage_timing=costs,seconds_by_component=secs,
        projected_gpu_hours=total*4/3600,budget_gpu_hours_25pct=1.25*total*4/3600,
        serial_wall_hours_25pct=1.25*total/3600,assumptions=dict(updates_per_arm=50,
        history_windows=counts,free_pool_model_pairs=16,common_pool_model_pairs=6,
        terminal='both reach update 50; early stops change actual cost',throughput='linear extrapolation from measured five-update refresh and 8 tasks per rank; offline calibration/process startup conservatively scale with window')))
    save(PUBLIC/'initialization_check.json',dict(passed=True,rows=check,
         classes=read(RUN/'initial/calibration.json'),initial_score_manifest=[digest_file(RUN/'initial'/f'score_{r}.json') for r in range(4)],
         support='fixed h and C for actor replacement; nominal C from current online actor',
         source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


if __name__=='__main__':
    if sys.argv[1:] == ['check']:
        from .e25_probe import configure
        from .gap_targets import init_check
        init_check(configure())
    elif sys.argv[1:] == ['timings']:
        for tag,name,prev in [('initial','initial',[]),('refresh','timing_refresh',['initial'])]:
            run_phase(tag+'_fit',multi('gap_fit','fit',name,*prev))
            run_phase(tag+'_calibrate',[sys.executable,'-m',PREFIX+'gap_fit','calibrate',name])
        run_phase('initialization',multi('gap_preflight','check'))
        run_phase('pages',multi('gap_eval'))
        run_phase('clicks_and_nomination',multi('gap_clicks','bench'))
        clocks=[read(RUN/'page_timing'/f'rank{i}'/'hf_timing.json') for i in range(4)]
        save(RUN/'click_timing.json',dict(startup_seconds=max(a['startup_seconds'] for a in clocks),
             wall_for_200_tasks=max(a['click_seconds'] for a in clocks)*50/8,by_rank=clocks))
        save(RUN/'actor_timing.json',dict(startup_seconds=max(a['startup_seconds'] for a in clocks),
             current_forward_per_update_seconds=max(a['actor_seconds'] for a in clocks),
             records=sum(a['actor_records'] for a in clocks)))
        summarize()
