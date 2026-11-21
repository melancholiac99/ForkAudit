"""Saved exact trajectories under the same prompt/interface as native training."""
from collections import defaultdict
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .common import *
from .checks import environment
from exploratory_shop_audit.runtime import RPC,make_input,parse_generation
from exploratory_shop_audit.run_pilot import public_state


class InferencePool:
    """Independent TP1 engines; one controller owns all environment and log writes."""
    def __init__(self,cmd,run,ids):
        self.ids=ids;self.pool=ThreadPoolExecutor(max_workers=len(ids));self.models=[]
        def start(gpu):
            # Each TP1 engine creates its own rendezvous even when the controllers use torchrun.
            rank_env = {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK',
                        'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'}
            env = {k:v for k,v in os.environ.items() if k not in rank_env and not k.startswith('TORCHELASTIC_')}
            env.update(PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=str(gpu),VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='2')
            return RPC(cmd,run/f'inference_gpu_{gpu}.log',env,timeout=900)
        futures=[self.pool.submit(start,gpu) for gpu in ids]
        try:
            for future in futures:self.models.append(future.result())
        except BaseException:
            for future in futures:
                try:
                    model=future.result()
                    if model not in self.models:self.models.append(model)
                except Exception:pass
            self.close();raise
    def call(self,requests):
        groups=[list(range(i,len(requests),len(self.models))) for i in range(len(self.models))]
        futures=[self.pool.submit(model.call,requests=[requests[i] for i in group]) if group else None for model,group in zip(self.models,groups)]
        result=[None]*len(requests)
        for group,future in zip(groups,futures):
            if future is None:continue
            rows=future.result();assert len(rows)==len(group)
            for i,row in zip(group,rows):result[i]=row
        return result
    def close(self):
        try:
            for model in self.models:model.close()
        finally:self.pool.shutdown(wait=True)


class Runner:
    def __init__(self,run,checkpoint='base',*,parallel_start=False):
        from transformers import AutoTokenizer
        self.run=run; self.checkpoint=checkpoint
        self.tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
        engine_name='collection_engine_gpu_'+os.environ.get('SHOP_GPU_ID','0')
        cmd=[GPU_PYTHON,'-m',__package__+'.inference_worker','--model',MODEL]
        if checkpoint!='base':cmd+=['--adapter',checkpoint]
        self.gpu_ids=[int(x) for x in os.environ.get('SHOP_GPU_IDS',os.environ.get('SHOP_GPU_ID','0')).split(',')]
        if parallel_start:
            def start_environment():
                started=time.monotonic()
                engine=environment(run,engine_name)
                return engine,(started,time.monotonic())
            with ThreadPoolExecutor(max_workers=1) as startup:
                pending = startup.submit(start_environment)
                model_start=time.monotonic()
                try:
                    self.model = InferencePool(cmd, run, self.gpu_ids)
                    model_end=time.monotonic()
                except BaseException:
                    if pending.exception() is None:
                        pending.result()[0].close()
                    raise
                try:
                    self.engine,(env_start,env_end) = pending.result()
                    self.startup_timing=dict(environment_seconds=env_end-env_start,
                        model_seconds=model_end-model_start,
                        overlap_seconds=max(0.,min(model_end,env_end)-max(model_start,env_start)))
                except BaseException:
                    self.model.close()
                    raise
        else:
            self.engine=environment(run,engine_name)
            self.model=InferencePool(cmd,run,self.gpu_ids)
        self.checkpoint_hash=digest_file(Path(checkpoint)/'adapter_model.safetensors') if checkpoint!='base' else 'Qwen2.5-7B-Instruct:a09a35458c702b33eeacc393d103063234e8bc28'
    def close(self):
        try:self.model.close()
        finally:self.engine.close()
    def trajectories(self,folder,jobs,temperature=.7):
        folder.mkdir(parents=True,exist_ok=True)
        completed={r['trajectory_id']:r for r in read_rows(folder/'episodes.jsonl')}
        steps=defaultdict(list)
        for r in read_rows(folder/'steps.jsonl'):steps[r['trajectory_id']].append(r)
        batch_size=16*len(self.gpu_ids)
        for offset in range(0,len(jobs),batch_size):
            active=[]
            for job in jobs[offset:offset+batch_size]:
                key=job['trajectory_id']
                if key in completed:continue
                prior=steps[key];prefix=job.get('prefix',[])+[r['action'] for r in prior]
                history=job.get('history',[])+[(r['before']['observation'],r['generation']['text']) for r in prior]
                if prior and prior[-1]['done']:
                    append(folder/'episodes.jsonl',prior[-1]['episode_result']);continue
                state=self.engine.call(cmd='open',key=key,goal_index=job['goal_index'],prefix=prefix)
                active.append({'job':job,'key':key,'state':state,'prefix':prefix,'history':history,'prior':len(prior)})
            while active:
                inputs=[make_input(self.tokenizer,e['state']['task'],e['history'],e['state']['observation']) for e in active]
                requests=[{'token_ids':inp['token_ids'],'temperature':temperature,'seed':stable_seed(SEED,e['key'],len(e['prefix']))} for e,inp in zip(active,inputs)]
                outputs=self.model.call(requests=requests)
                append(self.run/'inference_costs.jsonl',{'time':utc(),'checkpoint_sha':self.checkpoint_hash,'requests':len(requests),'input_tokens':sum(len(r['token_ids']) for r in requests),'output_tokens':sum(len(r['token_ids']) for r in outputs)})
                keep=[]
                for e,inp,req,gen in zip(active,inputs,requests,outputs):
                    action=parse_generation(gen['text']);out=self.engine.call(cmd='step',key=e['key'],action=action);status=out['status']
                    row={'run_id':self.run.name,**e['job'],'checkpoint_sha':self.checkpoint_hash,'interface_version':INTERFACE,
                        'step':len(e['prefix']),'state_id':e['key']+':'+str(len(e['prefix'])),'seed':req['seed'],
                        'input':inp,'generation':gen,'action':action,'legal':out['legal'],'before':public_state(e['state']),
                        'after':public_state(out['state']),'reward':status.get('reward',0),'reward_valid':status.get('reward_valid'),
                        'termination_reason':status.get('termination_reason'),'done':status['done'],'status':status}
                    e['prefix'].append(action);e['history'].append((e['state']['observation'],gen['text']));e['state']=out['state']
                    if status['done']:
                        ep={k:row[k] for k in ['run_id','task_id','goal_index','trajectory_id','checkpoint_sha','interface_version','seed','reward','reward_valid','termination_reason','status']}
                        ep.update(length=len(e['prefix']),continuation_length=len(e['prefix'])-len(e['job'].get('prefix',[])),strict_success=status.get('termination_reason')=='gold_purchase',purchase=bool(status.get('purchase')),split=e['job'].get('split'),decode='greedy' if temperature==0 else 'sample')
                        row['episode_result']=ep
                        append(folder/'steps.jsonl',row);append(folder/'episodes.jsonl',ep)
                        self.engine.call(cmd='close',key=e['key'])
                    else:append(folder/'steps.jsonl',row);keep.append(e)
                active=keep
        return read_rows(folder/'episodes.jsonl')


def jobs(tasks,stage,checkpoint,repeats=1,decode='sample'):
    return [{'task_id':t['task_id'],'goal_index':t['goal_index'],'split':stage,
             'trajectory_id':f'{stage}:{checkpoint}:{decode}:{t["task_id"]}:{r}'} for t in tasks for r in range(repeats)]


def pilot(run):
    split=read(run/'splits.json');episodes=read_rows(run/'pilot/episodes.jsonl')
    if len(episodes)<8*len(split['pilot']):
        runner=Runner(run)
        try:episodes=runner.trajectories(run/'pilot',jobs(split['pilot'],'pilot','base',8))
        finally:runner.close()
    steps=read_rows(run/'pilot/steps.jsonl');grouped=defaultdict(list)
    for ep in episodes:grouped[ep['task_id']].append(ep['reward'])
    assert len(episodes)==8*len(split['pilot']) and all(len(v)==8 for v in grouped.values())
    legal=float(np.mean([s['legal'] for s in steps]));purchase=float(np.mean([e['purchase'] for e in episodes]));mixed=int(sum(np.ptp(v)>1e-8 for v in grouped.values()))
    summary={'episodes':len(episodes),'tasks':len(grouped),'legal_action_rate':legal,'purchase_rate':purchase,
             'reward_valid_rate':float(np.mean([e['reward_valid'] is True for e in episodes])),
             'group_std':[float(np.std(v,ddof=1)) for v in grouped.values()],'same_score_groups':len(grouped)-mixed,
             'all_full_groups':sum(all(r==1 for r in v) for v in grouped.values()),'all_non_full_groups':sum(all(r<1 for r in v) for v in grouped.values()),
             'effective_advantage_groups':mixed,'passed':legal>.1 and purchase>.05 and mixed>0,
             'engineering_gate':'Stop only for near-total interaction failure or no nonconstant reward group; no SFT fallback.'}
    save(run/'pilot_summary.json',summary)
    assert summary['passed'],summary


def evaluate(run,checkpoint,label,repeats=4):
    split=read(run/'splits.json');runner=Runner(run,checkpoint)
    try:
        for decode,temperature,n in [('greedy',0,1),('sample',.7,repeats)]:
            runner.trajectories(run/'evaluation'/label/decode,jobs(split['V'],'V',label,n,decode),temperature)
    finally:runner.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['pilot','evaluate']);p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--checkpoint',default='base');p.add_argument('--label',default='base');p.add_argument('--repeats',type=int,default=4);a=p.parse_args()
    if a.command=='pilot':pilot(a.run)
    else:
        from .priority_evaluation import run_pending
        run_pending(a.run)
        evaluate(a.run,a.checkpoint,a.label,a.repeats)

if __name__=='__main__':main()
