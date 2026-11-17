"""Community environment adapter for the pinned, unmodified verl-agent GRPO loss."""
import atexit
from collections import defaultdict
import functools
import json
import os
from pathlib import Path
import time

import numpy as np

from .common import *
from .checks import environment
from exploratory_shop_audit.runtime import make_input, parse_generation
from exploratory_shop_audit.run_pilot import public_state


def run_dir():return Path(os.environ['SHOP_GRPO_RUN'])


class Manager:
    def __init__(self,config,split):
        self.config=config;self.split=split;self.stage=split;self.checkpoint=0;self.decode='sample';self.chunk=0
        self.rpc=None; self.keys=[]
    def reset(self,kwargs):
        if self.rpc is None:
            self.rpc=environment(run_dir(),f'native_{self.split}');atexit.register(self.rpc.close)
        if self.keys:self.rpc.call(cmd='batch',requests=[{'cmd':'close','key':k} for k in self.keys])
        assert kwargs is not None
        self.tasks=[dict(k) for k in kwargs]; self.history=[[] for _ in kwargs]; self.actual=[None for _ in kwargs]
        counts=defaultdict(int);self.keys=[];self.repeats=[]
        for task in self.tasks:
            tid=task['task_id'];rep=counts[tid];counts[tid]+=1;self.repeats.append(rep)
            self.keys.append(f'{self.stage}:{self.checkpoint}:{self.decode}:{tid}:{rep}')
        self.states=self.rpc.call(cmd='batch',requests=[{'cmd':'open','key':k,'goal_index':t['goal_index']} for k,t in zip(self.keys,self.tasks)])
        self.chunk+=1
        return self.obs(),[{} for _ in self.tasks]
    def obs(self):
        return {'text':[s['observation'] for s in self.states],'image':None,'anchor':[s['observation'] for s in self.states]}
    def step(self,text_actions):
        requests=[];indices=[]; rewards=np.zeros(len(self.tasks));dones=np.array([s['done'] for s in self.states]);infos=[{'is_action_valid':True,'won':False} for _ in self.tasks]
        for i,text in enumerate(text_actions):
            if dones[i]:continue
            requests.append({'cmd':'step','key':self.keys[i],'action':parse_generation(text)});indices.append(i)
        results=self.rpc.call(cmd='batch',requests=requests)
        for i,result in zip(indices,results):
            before=self.states[i];text=text_actions[i];step=len(self.history[i]); status=result['status'];reward=float(status.get('reward',0))
            row={'run_id':run_dir().name,'task_id':self.tasks[i]['task_id'],'split':self.split,'checkpoint':self.checkpoint,
                'interface_version':INTERFACE,'trajectory_id':self.keys[i],'state_id':self.keys[i]+f':{step}',
                'step':step,'seed':stable_seed(SEED,self.keys[i],step),'action':requests[indices.index(i)]['action'],
                'generation_text':text,'input':self.actual[i],'before':public_state(before),'after':public_state(result['state']),
                'reward':reward,'reward_valid':status.get('reward_valid'),'legal':result['legal'],'done':status['done'],
                'termination_reason':status.get('termination_reason'),'status':status,'decode':self.decode}
            append(run_dir()/'rollouts'/f'{self.stage}_{self.checkpoint:04d}_{self.decode}_steps.jsonl',row)
            self.history[i].append((before['observation'],text));self.states[i]=result['state']
            rewards[i]=reward;dones[i]=status['done'];infos[i]={'is_action_valid':result['legal'],'won':status.get('termination_reason')=='gold_purchase'}
            if status['done']:
                append(run_dir()/'rollouts'/f'{self.stage}_{self.checkpoint:04d}_{self.decode}_episodes.jsonl',{
                    **{k:row[k] for k in ['run_id','task_id','split','checkpoint','interface_version','trajectory_id','seed','reward','reward_valid','termination_reason','decode']},
                    'length':len(self.history[i]),'strict_success':infos[i]['won'],'purchase':bool(status.get('purchase')),'status':status})
        return self.obs(),rewards,dones,infos
    def success_evaluator(self,total_infos,total_batch_list,episode_rewards,episode_lengths):
        return {'success_rate':[any(r.get('won') for r in rows) for rows in total_infos]}


def install():
    import torch
    import agent_system.environments as env_module
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector as C
    from verl.trainer.ppo import ray_trainer
    from verl.utils.model import compute_position_id_with_mask
    if getattr(C,'_shop_native',False):return
    C._shop_native=True
    env_module.make_envs=lambda config:(Manager(config,'train'),Manager(config,'V'))
    def preprocess(self,item,gen_batch,obs):
        manager=self._shop_manager; state=manager.states[item]
        actual=make_input(self.tokenizer,state['task'],manager.history[item],state['observation'],8192)
        manager.actual[item]=actual; ids=actual['token_ids'];size=self.config.data.max_prompt_length
        x=torch.full((size,),self.tokenizer.pad_token_id,dtype=torch.long);mask=torch.zeros(size,dtype=torch.long)
        x[-len(ids):]=torch.tensor(ids);mask[-len(ids):]=1
        return {'input_ids':x,'attention_mask':mask,'position_ids':compute_position_id_with_mask(mask[None])[0],
            'raw_prompt_ids':ids,'raw_prompt':actual['messages'],'anchor_obs':state['observation'],'index':item,
            'data_source':'shop_reward_v3','shop_seed':stable_seed(SEED,manager.keys[item],len(manager.history[item])),
            'shop_decision':json.dumps({'trajectory_id':manager.keys[item],'step':len(manager.history[item]),'task_id':manager.tasks[item]['task_id']})}
    C.preprocess_single_sample=preprocess
    old_collect=C.multi_turn_loop
    def collect(self,gen_batch,actor_rollout_wg,envs,is_train=True):
        self._shop_manager=envs
        if is_train:
            envs.checkpoint=int(os.environ.get('SHOP_COMPLETED_UPDATES','0'));envs.stage='train';envs.decode='sample'
        started=time.monotonic()
        from .native_cache import restore_batch, save_batch
        data=restore_batch(self,gen_batch,envs,is_train)
        reused=data is not None
        if data is None:
            data=old_collect(self,gen_batch,actor_rollout_wg,envs,is_train)
            save_batch(data,gen_batch,envs,is_train)
        # Verify that only current assistant outputs are trainable. Past assistant,
        # system, user and environment tokens are all in the masked prompt prefix.
        assert 'shop_decision' in data.non_tensor_batch
        for i in ([] if reused else range(len(data))):
            meta=json.loads(data.non_tensor_batch['shop_decision'][i]);responses=data.batch['responses'][i]
            rmask=data.batch['attention_mask'][i][-len(responses):].bool()
            ids=responses[rmask].tolist()
            assert ids
            append(run_dir()/'rollouts'/f'{envs.stage}_{envs.checkpoint:04d}_{envs.decode}_tokens.jsonl',{
                **meta,'response_token_ids':ids,'generation_truncated':len(ids)==512 and ids[-1]!=self.tokenizer.eos_token_id})
        if is_train:
            rewards={}
            for meta,r in zip(data.non_tensor_batch['shop_decision'],data.non_tensor_batch['episode_rewards']):
                m=json.loads(meta);rewards[m['trajectory_id']]=(m['task_id'],float(r))
            grouped=defaultdict(list)
            for tid,r in rewards.values():grouped[tid].append(r)
            assert all(len(v)==8 for v in grouped.values())
            append(run_dir()/'training_signal.jsonl',{'update':envs.checkpoint+1,'tasks':len(grouped),'episodes':len(rewards),
                'same_score_groups':sum(bool(np.ptp(v)<=1e-8) for v in grouped.values()),'group_std':[float(np.std(v,ddof=1)) for v in grouped.values()],
                'seconds':time.monotonic()-started,'reused_saved_batch':reused})
        return data
    C.multi_turn_loop=collect
    # Preserve native GRPO's default cross-step normalization; record it explicitly.
    original_adv=ray_trainer.compute_advantage
    def advantage(data,*args,**kwargs):
        out=original_adv(data,*args,**kwargs)
        assert torch.isfinite(out.batch['advantages']).all()
        mask=out.batch['response_mask']; assert mask.shape==out.batch['responses'].shape
        assert torch.all(out.batch['advantages'][~mask.bool()]==0)
        append(run_dir()/'advantage_checks.jsonl',{'update':int(os.environ.get('SHOP_COMPLETED_UPDATES','0'))+1,
            'finite':True,'response_only_mask':True,'nonzero_fraction':float((out.batch['advantages'][mask.bool()]!=0).float().mean()),
            'rows':len(out),'native_cross_step_normalization':True})
        return out
    ray_trainer.compute_advantage=advantage
    old_validate=ray_trainer.RayPPOTrainer._validate
    def validate(self):
        complete=int(os.environ.get('SHOP_COMPLETED_UPDATES','0'));self.val_envs.checkpoint=complete;self.val_envs.stage='evaluation'
        all_metrics={}
        from omegaconf import open_dict
        for decode in ['greedy','sample']:
            self.val_envs.decode=decode
            with open_dict(self.config):self.config.actor_rollout_ref.rollout.val_kwargs.do_sample=decode=='sample'
            metrics=old_validate(self);all_metrics.update({f'{decode}/{k}':v for k,v in metrics.items()})
            from .analysis import evaluate_saved
            epfile=run_dir()/'rollouts'/f'evaluation_{complete:04d}_{decode}_episodes.jsonl'
            stfile=run_dir()/'rollouts'/f'evaluation_{complete:04d}_{decode}_steps.jsonl'
            ep=read_rows(epfile);steps=read_rows(stfile)
            tokens={(r['trajectory_id'],r['step']):r for r in read_rows(run_dir()/'rollouts'/f'evaluation_{complete:04d}_{decode}_tokens.jsonl')}
            for row in steps:row['generation_truncated']=tokens[(row['trajectory_id'],row['step'])]['generation_truncated']
            assert len(ep)==len({r['trajectory_id'] for r in ep})==len(read(run_dir()/'splits.json')['V'])
            if self.val_envs.rpc is None:
                self.val_envs.rpc=environment(run_dir(),'native_V');atexit.register(self.val_envs.rpc.close)
            summary=evaluate_saved(run_dir(),ep,steps,lambda i,a:self.val_envs.rpc.call(cmd='maximum',goal_index=i,asin=a),f'checkpoint{complete}:{decode}')
            save(run_dir()/'evaluation'/f'checkpoint_{complete:04d}_{decode}.json',summary)
        return all_metrics
    ray_trainer.RayPPOTrainer._validate=validate
    # Instrument the native fit after actor update, without changing PPO or rewards.
    import inspect,textwrap
    source=textwrap.dedent(inspect.getsource(ray_trainer.RayPPOTrainer.fit))
    anchor='                    metrics.update(actor_output_metrics)\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'''                    import exploratory_shop_audit.grpo_forced_audit.native as shop_native
                    shop_native.after_update(self.global_steps, actor_output_metrics)
''')
    scope=dict(ray_trainer.__dict__);exec(compile(source,'<shop_native_health_instrumentation>','exec'),scope)
    ray_trainer.RayPPOTrainer.fit=scope['fit']


def after_update(step,metrics):
    import math
    numeric={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float,np.number))}
    assert all(math.isfinite(v) for v in numeric.values()),'Non-finite training metric'
    os.environ['SHOP_COMPLETED_UPDATES']=str(step)
    append(run_dir()/'training_metrics.jsonl',{'time':utc(),'update':step,'metrics':numeric})
