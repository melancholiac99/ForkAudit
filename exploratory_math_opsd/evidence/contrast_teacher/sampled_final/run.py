import json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from contexts import ROOT,WORK as ORIGINAL,MODEL,read,save,sha
from train import replace
from launch import evaluate,validate_checkpoint
WORK=ORIGINAL.parent/'contrast-sampled-final'
ARM='R-contrast-sampled-final'

def prepare():
    src=WORK/'source'
    shutil.copytree(ORIGINAL/'source',src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy(HERE/'sampled.py',src/'sampled.py')
    shutil.copy(HERE.parent/'repair/runtime.py',src/'repair_runtime.py')
    parts=src/'training_parts.py';s=parts.read_text()
    a=s.index('            review=');b=s.index('        pos=super()',a)
    parts.write_text(s[:a]+s[b:s.index('def loss(')])
    path=src/'opsd_trainer.py';s=path.read_text()
    s=replace(s,'from training_parts import teachers, loss as contrast_loss','from training_parts import teachers\n        from sampled import loss as contrast_loss')
    s=replace(s,'inputs["has_negative"], self.state.global_step+1','inputs["has_negative"], inputs["sampled_behavior_logp"], self.state.global_step+1')
    s=replace(s,'self.vllm_engine = LLM(','self.vllm_engine = LLM(\n                    logprobs_mode="processed_logprobs",')
    key='sampling_params = SamplingParams('
    s=s.replace(key,key+'\n                logprobs=0,',1)
    s=replace(s,'        padded_completion_ids = torch.stack(padded_completion_ids_list)','''        padded_completion_ids = torch.stack(padded_completion_ids_list)
        from sampled import rollout_metadata
        inputs["sampled_behavior_logp"]=rollout_metadata(all_outputs,completion_ids,padded_completion_ids,self)''')
    path.write_text(s)
    path=src/'opsd_train.py';path.write_text(replace(path.read_text(),'    trainer.train(resume_from_checkpoint=', '''    from repair_runtime import WindowGate
    trainer._repair_eos=set(json.loads((Path(model_args.model_name_or_path)/'generation_config.json').read_text())['eos_token_id'])
    trainer.repair_counts=[0,0]
    trainer.add_callback(WindowGate(trainer))
    trainer.train(resume_from_checkpoint='''))
    return src

def train():
    src=prepare();output=WORK/'checkpoints'/ARM;output.mkdir(parents=True,exist_ok=True)
    link=ROOT/'run'/ARM
    if not link.exists():link.symlink_to(output,target_is_directory=True)
    assert link.resolve()==output and not list(output.glob('ratio-rank*'))
    negative=ORIGINAL/'negative.json'
    assert sha(negative)==read(HERE.parent/'negative_contexts.json')['sha256']
    with (WORK/'train.log').open('a') as f:
        subprocess.run(['bash','scripts/run_opsd_1b.sh','--stop_after_updates','100','--seed','42','--data_seed','42','--save_steps','25','--save_only_model','True','--local_train_glob',str(ROOT/'datasets/train/data/*.parquet')],cwd=src,stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OPSD_ARM':ARM,'OPSD_MODEL':str(MODEL),'OPSD_OUTPUT':str(output),'CONTRAST_NEGATIVES':str(negative)})

def finish():
    output=WORK/'checkpoints'/ARM
    windows=[json.loads(s) for s in (output/'end_windows.jsonl').read_text().splitlines()]
    last=windows[-1]['step'];assert [w['step'] for w in windows]==list(range(25,last+1,25))
    ratios=[]
    for rank in range(4):
        rows=[json.loads(s) for s in (output/f'ratio-rank-{rank}.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==[s for s in range(1,last+1) for _ in range(2)]
        ratios.extend(rows)
    for w in windows:
        rows=[r for r in ratios if w['step']-25<r['step']<=w['step']]
        w['clipped']=sum(r['clipped'] for r in rows);w['supervised']=sum(r['supervised'] for r in rows)
        w['clip_fraction']=w['clipped']/w['supervised']
    ck=output/f'checkpoint-{last}';validate_checkpoint(ck,last)
    report=dict(windows=windows,adapter_sha256=sha(ck/'adapter_model.safetensors'),ratio_clip_fraction=sum(w['clipped'] for w in windows)/sum(w['supervised'] for w in windows))
    if windows[-1]['stop']:report.update(verdict='CLOSED_EARLY_STOP')
    else:
        assert last==100
        scores=[]
        for arm in (ARM+'@100','R@100','R-noclip@100'):
            receipt=evaluate(ROOT,os.environ.copy(),arm,'aime24',1001)
            if not scores:assert receipt['adapter_sha256']==report['adapter_sha256']
            qs=read(ROOT/'run/evaluation'/arm/'aime24/1001.json')['results'];keys=[q['problem'] for q in qs]
            if scores:assert keys==order
            order=keys;scores.append(np.array([[g['correct'] for g in q['generations']] for q in qs],dtype=float));assert scores[-1].shape==(30,12)
        delta=100*(scores[0]-scores[1]).mean(1)
        ci=np.quantile(delta[np.random.default_rng(20261124).integers(30,size=(10000,30))].mean(1),[.025,.975])
        report.update(accuracy_pct=scores[0].mean()*100,delta_R_pp=delta.mean(),ci95=ci.tolist(),verdict='SUCCESS' if ci[0]>-5 and scores[0].sum()>=scores[2].sum() else 'CLOSED')
    save(HERE/'final.json',report)
if __name__=='__main__':
    if 'finish' not in sys.argv:train()
    finish()
