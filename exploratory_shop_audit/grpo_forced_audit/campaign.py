"""Isolated concurrent stages under one cumulative GPU-hour budget."""
import argparse
import os
import shutil
import subprocess
import time
from .common import *
from .budget import execute,spent,uncapped
from .train import prepare_data,launch
from .report import platform,final_report


def gpu_stage(run,stage,module,args=(),limit=None,gpu='0,1,2,3'):
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',SHOP_GPU_IDS=str(gpu),SHOP_GPU_ID=str(gpu).split(',')[0],CUDA_VISIBLE_DEVICES=str(gpu))
    execute(run,stage,[CPU_PYTHON,'-m',__package__+'.'+module,*args,'--run',run],env,limit)


def checkpoint(run,n):return run/'training/checkpoints'/f'global_step_{n}'


def checkpoint_complete(run,n):
    folder=checkpoint(run,n)/'actor'
    return all((folder/p).exists() for p in ['live_fingerprint_rank_0.json','lora_adapter/adapter_model.safetensors','lora_adapter/adapter_config.json'])


def freeze_budget(run):
    ledger=read_rows(run/'cost_ledger.jsonl');finished={r['stage']:r['gpu_hours'] for r in ledger if r['event']=='end'}
    # Includes model reload/validation overhead: conservative cost extrapolation.
    per_update=finished['train_0010']/9
    audit_per_task=finished['audit_base_throughput']/5
    remaining=24-spent(run);eval_reserve=max(2.,finished['train_0001']*3)
    e1_reserve=max(.5,audit_per_task*50*1.25)
    chosen=next((n for n in [200,100,50] if 1.25*audit_per_task*n+e1_reserve+eval_reserve+per_update*20*1.25<remaining),50)
    max_updates=min(100,10+10*int(max(0,remaining-eval_reserve-e1_reserve-audit_per_task*chosen*1.25)/(10*per_update*1.25)))
    max_updates=max(10,max_updates)
    if uncapped(run):chosen,max_updates=200,100
    data={'time':utc(),'gpu_hours_spent':spent(run),'per_update_hours_conservative':per_update,'audit_hours_per_task_conservative':audit_per_task,
        'E0_tasks':chosen,'E1_tasks':50,'max_updates':max_updates,'final_repeats':4,'final_evaluation_reserve_hours':eval_reserve,
        'training_reserve_remaining_hours':max(0,remaining-eval_reserve-e1_reserve-audit_per_task*chosen*1.25),
        'gpu_hour_limit':None if uncapped(run) else 24,
        'reason':'Researcher removed GPU-hour cap: use existing largest E0=200 and original maximum 100 updates, retaining plateau rule.' if uncapped(run) else 'Cost only, using first 10 updates and first 5 frozen E0 tasks; no reward-dependent scale choice.'}
    write_once(run/'budget_freeze.json',data);return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--wait-for-cpu',action='store_true');a=p.parse_args();run=a.run.resolve();setup(run)
    if a.wait_for_cpu:
        while not (run/'splits.json').exists():time.sleep(5)
    assert read(run/'preflight_report.json')['passed']
    assert (run/'historical_reanalysis/metrics.json').exists()
    prepare_data(run)
    (run/'rollouts').mkdir(exist_ok=True)
    if not (run/'pilot_summary.json').exists():gpu_stage(run,'pilot','rollout',['pilot'],1)
    assert read(run/'pilot_summary.json')['passed']
    existing=[int(p.name.split('_')[-1]) for p in (run/'training/checkpoints').glob('global_step_*') if checkpoint_complete(run,int(p.name.split('_')[-1]))]
    latest=max(existing,default=0)
    if latest==0:launch(run,1,limit_hours=2);latest=1
    assert (run/'weight_updates.jsonl').exists()
    # Each phase owns all four devices; never run an audit on a training rank.
    if not (run/'audit_base/metrics.json').exists():gpu_stage(run,'audit_base_throughput','audit',['--count','5'],1)
    if latest<10:launch(run,10,checkpoint(run,latest),limit_hours=4);latest=10
    assert read(run/'training/restore_rank_0.json')['passed']
    budget=read(run/'budget_freeze.json') if (run/'budget_freeze.json').exists() else freeze_budget(run)
    gpu_stage(run,'audit_base','audit',['--count',str(budget['E0_tasks'])])
    final=latest
    for update in range(final+10,budget['max_updates']+1,10):
        if not uncapped(run) and spent(run)+budget['final_evaluation_reserve_hours']+max(.5,budget['audit_hours_per_task_conservative']*budget['E1_tasks']*1.25)>=24:break
        launch(run,update,checkpoint(run,final))
        final=update
        if platform(run):break
    final_adapter=checkpoint(run,final)/'actor/lora_adapter'
    assert (final_adapter/'adapter_model.safetensors').exists()
    save(run/'final_checkpoint.json',{'update':final,'path':str(final_adapter),'sha256':digest_file(final_adapter/'adapter_model.safetensors')})
    for label,adapter in [('base','base'),('final',str(final_adapter))]:
        gpu_stage(run,'evaluate_'+label,'rollout',['evaluate','--checkpoint',adapter,'--label',label,'--repeats',str(budget['final_repeats'])])
    gpu_stage(run,'audit_post','audit',['--checkpoint',str(final_adapter),'--label','final','--count',str(budget['E1_tasks'])])
    final_report(run)
    save(run/'campaign_status.json',{'status':'complete','time':utc(),'gpu_hours':spent(run),'training_updates':final,'no_next_experiment':True})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        # Keep traceback and all partial samples; never pick a better retry.
        import traceback
        append(DEFAULT_RUN/'errors.jsonl',{'time':utc(),'stage':'campaign','error':traceback.format_exc()})
        raise
