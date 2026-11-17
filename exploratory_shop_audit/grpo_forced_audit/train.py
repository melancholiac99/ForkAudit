"""Resolved WebShop-derived native GRPO launcher; no auxiliary credit or SFT."""
import json
import os
from pathlib import Path
import subprocess
from .common import *
from .budget import execute

SOURCE_VERL=ROOT/'.runtime/external/verl-agent-20bd331b'
MIGRATION=ROOT/'exploratory_ctha/verl_agent_migration'


def settings(run,updates=100,resume=None):
    output=run/'training';output.mkdir(exist_ok=True);(run/'rollouts').mkdir(exist_ok=True)
    data=run/'training_data'
    return {
        'algorithm.adv_estimator':'grpo','algorithm.filter_groups.enable':False,
        'algorithm.use_kl_in_reward':False,'algorithm.gamma':1.,'algorithm.norm_adv_by_std_in_grpo':True,
        'data.train_files':str(data/'train.parquet'),'data.val_files':str(data/'val.parquet'),
        '+data.seed':SEED,'data.train_batch_size':16,'data.val_batch_size':20,'data.max_prompt_length':8192,'data.max_response_length':512,
        'data.filter_overlong_prompts':False,'data.truncation':'error','data.return_raw_chat':True,
        'actor_rollout_ref.model.path':str(MODEL),'actor_rollout_ref.model.lora_rank':64,'actor_rollout_ref.model.lora_alpha':64,
        'actor_rollout_ref.model.target_modules':'all-linear','actor_rollout_ref.model.use_remove_padding':True,
        'actor_rollout_ref.model.enable_gradient_checkpointing':True,
        'actor_rollout_ref.model.external_lib':'exploratory_shop_audit.grpo_forced_audit.worker_hooks',
        'actor_rollout_ref.actor.optim.lr':1e-6,'actor_rollout_ref.actor.optim.lr_warmup_steps':0,
        'actor_rollout_ref.actor.optim.warmup_style':'constant','actor_rollout_ref.actor.ppo_mini_batch_size':64,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,'actor_rollout_ref.actor.ppo_epochs':1,
        'actor_rollout_ref.actor.clip_ratio_low':.2,'actor_rollout_ref.actor.clip_ratio_high':.2,'actor_rollout_ref.actor.clip_ratio_c':3.,
        'actor_rollout_ref.actor.entropy_coeff':.001,'actor_rollout_ref.actor.grad_clip':1.,
        'actor_rollout_ref.actor.use_kl_loss':True,'actor_rollout_ref.actor.kl_loss_coef':.01,'actor_rollout_ref.actor.kl_loss_type':'low_var_kl',
        'actor_rollout_ref.actor.use_invalid_action_penalty':False,
        'actor_rollout_ref.actor.fsdp_config.param_offload':False,'actor_rollout_ref.actor.fsdp_config.optimizer_offload':False,
        '+actor_rollout_ref.actor.fsdp_config.model_dtype':'bf16',
        'actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.load_format':'safetensors',
        'actor_rollout_ref.rollout.temperature':.7,'actor_rollout_ref.rollout.top_p':1.,'actor_rollout_ref.rollout.top_k':-1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size':1,'actor_rollout_ref.rollout.gpu_memory_utilization':.5,
        'actor_rollout_ref.rollout.max_num_seqs':16,'actor_rollout_ref.rollout.max_model_len':8704,
        'actor_rollout_ref.rollout.max_num_batched_tokens':16384,'actor_rollout_ref.rollout.enable_chunked_prefill':False,
        'actor_rollout_ref.rollout.enforce_eager':False,'actor_rollout_ref.rollout.free_cache_engine':False,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':4,
        '+actor_rollout_ref.rollout.cudagraph_capture_sizes':[1,2,4,8,16],
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode':'processed_logprobs',
        'actor_rollout_ref.rollout.val_kwargs.temperature':.7,'actor_rollout_ref.rollout.val_kwargs.do_sample':True,
        'actor_rollout_ref.rollout.val_kwargs.n':1,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu':4,'actor_rollout_ref.ref.fsdp_config.param_offload':True,
        'env.env_name':'Webshop','env.seed':SEED,'env.max_steps':35,'env.rollout.n':8,
        'trainer.critic_warmup':0,'trainer.logger':['console'],'trainer.project_name':'shop_grpo_forced_audit',
        'trainer.experiment_name':run.name,'trainer.n_gpus_per_node':4,'trainer.nnodes':1,
        'trainer.save_freq':10,'trainer.test_freq':10,'trainer.total_epochs':1000,'trainer.total_training_steps':updates,
        'trainer.val_before_train':resume is None,'trainer.max_actor_ckpt_to_keep':2,
        'trainer.resume_mode':'resume_path' if resume else 'disable','trainer.resume_from_path':str(resume) if resume else None,
        'trainer.default_local_dir':str(output/'checkpoints'),'trainer.rollout_data_dir':str(output/'rollouts'),
        'trainer.validation_data_dir':str(output/'validation'),
        'ray_init.num_cpus':16,'+ray_init.object_store_memory':2147483648,'+ray_init.dashboard_port':0,
        '+ray_init.runtime_env.worker_process_setup_hook':'exploratory_shop_audit.grpo_forced_audit.launch.setup_worker',
        'hydra.run.dir':str(output/'hydra')}


def prepare_data(run):
    # Only explicit TRAIN/V IDs enter native datasets. Text fields are modality
    # placeholders; every actual policy input comes from the checked environment.
    import pandas as pd
    split=read(run/'splits.json');folder=run/'training_data';folder.mkdir(exist_ok=True)
    for name,key in [('train','T'),('val','V')]:
        rows=[{'data_source':'shop_reward_v3','prompt':[{'role':'user','content':'Shopping task'}],
               'reward_model':{'style':'rule','ground_truth':''},'extra_info':{'index':i},
               'env_kwargs':{'goal_index':t['goal_index'],'task_id':t['task_id']}} for i,t in enumerate(split[key])]
        pd.DataFrame(rows).to_parquet(folder/f'{name}.parquet',index=False)


def launch(run,updates=100,resume=None,limit_hours=13):
    assert read(run/'preflight_report.json')['passed']
    assert read(run/'pilot_summary.json')['passed']
    from .parallel import prepare_restore
    if resume:resume=prepare_restore(resume,4)
    cfg=settings(run,updates,resume)
    env=os.environ.copy();env.update(VERL_AGENT_SOURCE=str(SOURCE_VERL),SHOP_GRPO_RUN=str(run),
        FORKAUDIT_VERL_OUTPUT=str(run/'training'),FORKAUDIT_MODEL_SEED=str(SEED),FORKAUDIT_ROLLOUT_SESSION='1',
        PYTHONPATH=f'{ROOT}:{MIGRATION}:{SOURCE_VERL}',CUDA_VISIBLE_DEVICES='0,1,2,3',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        VLLM_ATTENTION_BACKEND='FLASH_ATTN',HYDRA_FULL_ERROR='1',RAY_TMPDIR='/tmp/shop_grpo_20261117',
        SHOP_COMPLETED_UPDATES=str(int(resume.name.split('_')[-1])) if resume else '0')
    for name in ['FORKAUDIT_P1_CAMPAIGN','FORKAUDIT_P1_ARM','FORKAUDIT_REFERENCE_PACKET','FORKAUDIT_GRADIENT_BENCH_PACKET']:
        env.pop(name,None)
    authorization=run/'recovery_0042/continuation_authorization.json'
    if resume and resume.name=='global_step_40' and authorization.exists():
        assert read(authorization)['researcher_approved'] and read(authorization)['canonical_logs_prepared']
        env.update(SHOP_RECONSTRUCT_UPDATES='1',SHOP_RECONSTRUCTION_REQUEST=str(run/'recovery_0042/continued_replay_request.json'))
    args=[f'{k}='+('null' if v is None else json.dumps(v,separators=(',',':')) if isinstance(v,(list,bool)) else str(v)) for k,v in cfg.items()]
    save(run/f'train_launch_{updates:04d}.json',{'settings':cfg,'command':[str(GPU_PYTHON),'-m',__package__+'.launch',*args],
        'source_sha256':{p.name:digest_file(p) for p in HERE.glob('*.py')},'framework_commit':'20bd331bdbc9026a5668e11362178e10ab7400c8'})
    execute(run,f'train_{updates:04d}',[GPU_PYTHON,'-m',__package__+'.launch',*args],env,limit_hours)
    from .parallel import prune_full_checkpoints
    prune_full_checkpoints(run,keep=2)
