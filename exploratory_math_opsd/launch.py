"""Launch the registered four-GPU R/T0 training and evaluations on a prepared host."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from preflight import run as preflight

SEEDS = (1001, 2002, 1500, 4004)
BENCHMARKS = ('aime24','aime25','hmmt25')
MODELS = ('B','R@100','T0@100','R@50','T0@50')


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):
            h.update(block)
    return h.hexdigest()


def call(root,cmd,log,env):
    start=time.monotonic()
    with (root/'logs'/log).open('a',buffering=1) as f:
        print(json.dumps(cmd),file=f)
        subprocess.run(cmd,cwd=root/'source',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-start


def validate_checkpoint(path,step):
    state=read(path/'trainer_state.json')
    if state['global_step']!=step or not (path/'adapter_model.safetensors').is_file():
        raise ValueError(f'Invalid checkpoint {path}')
    return state


def train(root,env,arm):
    out=root/'run'/arm
    receipt=out/'complete.json'
    if receipt.exists():
        assert sha(out/'checkpoint-100/adapter_model.safetensors')==read(receipt)['adapter_sha256']
        return
    runs=sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    cmd=['bash','scripts/run_opsd_1b.sh','--stop_after_updates','100','--seed','42','--data_seed','42',
         '--local_train_glob',str(root/'datasets/train/data/*.parquet')]
    if arm=='T0':cmd+=['--question_only_teacher']
    if runs:
        last=runs[-1]
        step=int(last.name.split('-')[-1])
        validate_checkpoint(last,step)
        if step<100:cmd+=['--resume_from_checkpoint',str(last)]
    if not runs or step<100:
        seconds=call(root,cmd,arm+'-train.log',{**env,'OPSD_MODEL':str(root/'models/Qwen3-1.7B'),
                     'OPSD_OUTPUT':str(out),'OPSD_ARM':arm})
    else:seconds=None
    for k in (25,50,75,100):validate_checkpoint(out/f'checkpoint-{k}',k)
    state=read(out/'checkpoint-100/trainer_state.json')
    save(receipt,dict(arm=arm,seed=42,completed_steps=state['global_step'],scheduled_steps=state['max_steps'],
                      adapter_sha256=sha(out/'checkpoint-100/adapter_model.safetensors'),last_process_seconds=seconds))


def validate_eval(row,bench,seed,adapter):
    assert row['dataset']==bench and row['seed']==seed and row['adapter']==adapter
    assert row['enable_thinking'] and row['val_n']==12 and len(row['results'])==30 and row['total_solutions']==360
    assert [row[k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','max_new_tokens')]==[1.,.95,-1,0.,0.,38912]
    assert len({x['problem'] for x in row['results']})==30
    for x in row['results']:
        assert len(x['generations'])==12 and x['num_correct']==sum(g['correct'] for g in x['generations'])
        assert all(0<=g['generated_tokens']<=38912 for g in x['generations'])


def evaluate(root,env,model,bench,seed):
    folder=root/'run/evaluation'/model/bench
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/f'{seed}.json'
    receipt=folder/f'{seed}.receipt.json'
    adapter=None
    if model!='B':
        arm,k=model.split('@')
        path=root/'run'/arm/f'checkpoint-{k}'
        validate_checkpoint(path,int(k))
        adapter=str(path)
    adapter_sha=sha(Path(adapter)/'adapter_model.safetensors') if adapter else None
    if receipt.exists():
        rec=read(receipt)
        assert sha(output)==rec['sha256'] and rec['adapter_sha256']==adapter_sha
        validate_eval(read(output),bench,seed,adapter)
        return rec
    pending=output.with_suffix('.pending')
    parquet,=sorted((root/'datasets'/bench/'data').glob('*.parquet'))
    cmd=[sys.executable,'eval/evaluate_math.py','--base_model',str(root/'models/Qwen3-1.7B'),
         '--dataset',bench,'--local_eval_file',str(parquet),'--seed',str(seed),'--val_n','12',
         '--enable_thinking','--temperature','1.0','--top_p','0.95','--top_k','-1','--min_p','0',
         '--presence_penalty','0','--max_new_tokens','38912','--tensor_parallel_size','4',
         '--output_file',str(pending)]
    if adapter:cmd+=['--checkpoint_dir',adapter]
    seconds=call(root,cmd,f'{model}-{bench}-{seed}.log',{**env,'NCCL_P2P_DISABLE':'1'})
    row=read(pending)
    validate_eval(row,bench,seed,adapter)
    pending.replace(output)
    rec=dict(sha256=sha(output),adapter_sha256=adapter_sha,wall_seconds=seconds,rate=row['average_at_n_pct'])
    save(receipt,rec)
    return rec


def evaluations(root,env,quick_only):
    folder=root/'run/evaluation'
    quick={m:evaluate(root,env,m,'aime24',1001) for m in MODELS[:3]}
    total=sum(x['wall_seconds'] for x in quick.values())
    reduced=total>5400
    rates={m:v['rate'] for m,v in quick.items()}
    report=dict(rates_percent=rates,R_minus_T0_pp=rates['R@100']-rates['T0@100'],
                R_minus_B_pp=rates['R@100']-rates['B'],four_gpu_seconds=total,reduced_at_50=reduced,
                planned_responses=15120 if reduced else 21600)
    save(folder/'quick.json',report)
    print(json.dumps(report),flush=True)
    if quick_only:return
    for model in MODELS:
        seeds=(1001,) if reduced and model.endswith('@50') else SEEDS
        for bench in BENCHMARKS:
            for seed in seeds:evaluate(root,env,model,bench,seed)
    from paired_report import summarize
    save(folder/'final.json',summarize(folder))


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['R','T0','train','quick','evaluate','all'])
    parser.add_argument('--root',type=Path,default=Path('/mnt/data/ForkAudit/math-opsd'))
    args=parser.parse_args()
    root=args.root.resolve()
    if Path(sys.prefix).resolve()!=(root/'conda-env').resolve():
        parser.error('Run with the dedicated OPSD conda environment')
    if torch.cuda.device_count()!=4:parser.error('Select the four experiment GPUs via CUDA_VISIBLE_DEVICES')
    if not (root/'assets.json').is_file():parser.error('Missing verified asset manifest')
    base_env={**os.environ,'PATH':os.pathsep.join([str(root/'conda-env/bin'),str(root/'cuda-12.8/bin'),os.environ['PATH']]),
              'CUDA_HOME':str(root/'cuda-12.8'),'TORCH_CUDA_ARCH_LIST':'8.0',
              'HF_HOME':str(root/'hf-cache'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
              'WANDB_MODE':'offline','WANDB_DIR':str(root/'logs'),'PYTHONNOUSERSITE':'1',
              'TORCH_EXTENSIONS_DIR':str(root/'torch-extensions'),'TRITON_CACHE_DIR':str(root/'triton-cache')}
    (root/'run').mkdir(exist_ok=True)
    with (root/'run/runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(root/'run/preflight.json',preflight(root))
        if args.action in ('R','T0','train','all'):
            for arm in (('R','T0') if args.action in ('train','all') else (args.action,)):
                train(root,base_env,arm)
        if args.action in ('quick','evaluate','all'):evaluations(root,base_env,args.action=='quick')


if __name__=='__main__':
    main()
