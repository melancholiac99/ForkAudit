"""Registered non-thinking MATH-500 trajectory and selected-checkpoint evaluation."""
import fcntl
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from four_engines import run
from launch import call,evaluate,read,save,sha,validate_checkpoint

ROOT=Path('/mnt/data/ForkAudit/math-opsd')
DATA=Path('/mnt/data2/ForkAudit/math-opsd/math500')
OUT=ROOT/'run/math500-trajectory-k20'
EVIDENCE=Path(__file__).resolve().parent/'evidence/math500_trajectory'
REV='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'
DIGEST='35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'
ARMS=('R','T0','P','Rres')
STEPS=(25,50,75,100)
BENCHES=('aime24','aime25','hmmt25')

def prepare():
    raw=DATA/'test.jsonl'
    assert sha(raw)==DIGEST
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows)==len({r['unique_id'] for r in rows})==len({r['problem'] for r in rows})==500
    assert all(r['solution'] and r['problem'] for r in rows)
    table=pa.Table.from_pylist(rows)
    path=DATA/'test.parquet'
    if not path.exists():
        pq.write_table(table,path.with_suffix('.pending'))
        path.with_suffix('.pending').replace(path)
    assert pq.read_table(path).equals(table)
    hashes={}
    for a in ARMS:
        for k in STEPS:
            folder=ROOT/'run'/a/f'checkpoint-{k}'
            validate_checkpoint(folder,k)
            hashes[f'{a}@{k}']=sha(folder/'adapter_model.safetensors')
    manifest=dict(dataset='HuggingFaceH4/MATH-500',revision=REV,split='test',rows=500,
                  jsonl_sha256=DIGEST,parquet_sha256=sha(path),checkpoint_sha256=hashes,
                  evaluator_sha256=sha(ROOT/'source/eval/evaluate_math.py'))
    dest=EVIDENCE/'manifest.json'
    if dest.exists(): assert read(dest)==manifest
    else:save(dest,manifest)
    return rows,hashes

def summarize(result,rows,adapter):
    assert result['dataset']=='math500' and result['adapter']==adapter and result['seed']==1001
    assert result['base_model']==str(ROOT/'models/Qwen3-1.7B')
    assert not result['enable_thinking'] and result['val_n']==4
    assert [result[k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','max_new_tokens')]==[.7,.8,20,0.,0.,4096]
    assert [r['problem'] for r in result['results']]==[x['problem'] for x in rows]
    scores=[]; tokens=0
    for r in result['results']:
        gs=r['generations']
        assert len(gs)==4 and all(0<=g['generated_tokens']<=4096 for g in gs)
        assert r['num_correct']==sum(g['correct'] for g in gs)
        scores.append(r['num_correct']);tokens+=sum(g['generated_tokens'] for g in gs)
    correct=sum(scores)
    assert result['total_solutions']==2000 and result['average_at_n']==correct
    return dict(correct=correct,answers=2000,accuracy_percent=correct/20,
                mean_generated_tokens=tokens/2000)

def cell(model,rows,hashes):
    folder=OUT/model
    adapter=None if model=='B' else str(ROOT/'run'/model.split('@')[0]/('checkpoint-'+model.split('@')[1]))
    fingerprint=dict(top_k=20,data_sha256=DIGEST,adapter_sha256=hashes.get(model),seed=1001,
                     evaluator_sha256=sha(ROOT/'source/eval/evaluate_math.py'))
    folder.mkdir(exist_ok=True,parents=True)
    config=folder/'identity.json'
    if config.exists(): assert read(config)==fingerprint
    else:save(config,fingerprint)
    merged=folder/'answers.json'; receipt=folder/'receipt.json'
    if receipt.exists():
        rec=read(receipt)
        assert rec['output_sha256']==sha(merged) and rec['metrics']==summarize(read(merged),rows,adapter)
        return rec
    cmd=[sys.executable,str(Path(__file__).with_name('verified_eval.py')),'--base_model',str(ROOT/'models/Qwen3-1.7B'),
         '--dataset','math500','--local_eval_file',str(DATA/'test.parquet'),'--seed','1001','--val_n','4',
         '--no_thinking','--temperature','.7','--top_p','.8','--top_k','20','--min_p','0','--presence_penalty','0',
         '--max_new_tokens','4096','--tensor_parallel_size','1','--output_file',str(merged)]
    if adapter:cmd+=['--checkpoint_dir',adapter]
    result,seconds=run(ROOT,os.environ.copy(),cmd,folder/'shards','math500-'+model,call,read,questions=500,samples=4)
    for lane in range(4):
        proof=read(folder/'shards'/f'lane{lane}.pending.adapter.json')
        assert proof['adapter']==adapter and proof['sha256']==hashes.get(model)
        assert bool(proof['resident_ids'])==bool(adapter)
    metrics=summarize(result,rows,adapter)
    save(merged,result)
    rec=dict(**fingerprint,output_sha256=sha(merged),metrics=metrics,wall_seconds=seconds)
    save(receipt,rec);return rec

def compare(by_question,draw):
    return dict(difference_pp=float(by_question.mean()),ci95_pp=np.percentile(by_question[draw].mean(axis=1),[2.5,97.5]).tolist())

def question_scores(model):
    rows=read(OUT/model/'answers.json')['results']
    return np.array([r['num_correct']*25. for r in rows])

def trajectory(cells):
    draw=np.random.default_rng(20261124).integers(500,size=(10000,500))
    ref={k:question_scores(f'R@{k}') for k in STEPS}
    out={};deltas={}
    for model,rec in cells.items():
        metrics=rec['metrics']
        out[model]={k:metrics[k] for k in ('accuracy_percent','mean_generated_tokens')}
        if model=='B':continue
        arm,k=model.split('@')
        delta=question_scores(model)-ref[int(k)]
        out[model]['minus_same_step_R']=compare(delta,draw)
        deltas.setdefault(arm,[]).append(delta)
    screens={}
    for arm in ('Rw','Rnt'):
        if arm not in deltas:continue
        diffs=np.stack([question_scores(f'{arm}@{k}')-ref[k] for k in STEPS])
        pooled=compare(diffs.mean(axis=0),draw)
        positive=int((diffs.mean(axis=1)>0).sum())
        passed=positive>=3 and pooled['ci95_pp'][0]>0
        step=max(STEPS,key=lambda k:(cells[f'{arm}@{k}']['metrics']['correct'],-k)) if passed else None
        screens[arm]=dict(positive_steps=positive,average=pooled,passed=passed,selected_step=step)
    return out,screens

def confirmation(arm,step):
    pair=[f'R@{step}',f'{arm}@{step}']
    stats={}; diffs=[]
    for bench in BENCHES:
        scores=[]
        for model in pair:
            rec=evaluate(ROOT,os.environ.copy(),model,bench,1001)
            rows=read(ROOT/'run/evaluation'/model/bench/'1001.json')['results']
            scores.append({r['problem']:r['num_correct']/12 for r in rows})
            gens=[g for r in rows for g in r['generations']]
            stats.setdefault(bench,{})[model]=dict(accuracy_percent=rec['rate'],
                mean_generated_tokens=np.mean([g['generated_tokens'] for g in gens]),
                truncation_fraction=np.mean([g['finish_reason']=='length' for g in gens]),receipt=rec)
        assert scores[0].keys()==scores[1].keys()
        diffs.append([100*(scores[1][q]-scores[0][q]) for q in sorted(scores[0])])
    diff=np.array(diffs)
    draws=np.random.default_rng(20261124).integers(30,size=(10000,3,30))
    boot=diff[np.arange(3)[None,:,None],draws].mean(axis=2)
    for i,b in enumerate(BENCHES):
        stats[b]['difference_pp']=diff[i].mean()
        stats[b]['ci95_pp']=np.percentile(boot[:,i],[2.5,97.5]).tolist()
    return dict(benchmarks=stats,pooled_difference_pp=diff.mean(),pooled_ci95_pp=np.percentile(boot.mean(1),[2.5,97.5]).tolist())

def main():
    rows,hashes=prepare()
    if '--prepare' in sys.argv:return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cells={}
        def run_cell(model):
            cells[model]=cell(model,rows,hashes)
            save(OUT/'progress.json',cells)
        for model in ['B']+[f'{a}@{k}' for a in ARMS for k in STEPS]:run_cell(model)
        existing,_=trajectory(cells)
        save(EVIDENCE/'existing_trajectories.json',existing)
        sys.path.insert(0,str(Path(__file__).resolve().parent/'pmath'))
        from check_candidates import check
        save(EVIDENCE/'correctness.json',check())
        from launch import train
        for a in ('Rw','Rnt'):
            train(ROOT,{**os.environ,'OPSD_SCREEN_ARM':a,'OPSD_SCREEN_PROJECTOR':str(ROOT/'run/P-projector/projector.pt')},a)
        for a in ('Rw','Rnt'):
            for k in STEPS:
                model=f'{a}@{k}'
                path=ROOT/'run'/a/f'checkpoint-{k}'
                validate_checkpoint(path,k)
                hashes[model]=sha(path/'adapter_model.safetensors')
                run_cell(model)
        values,screen=trajectory(cells)
        save(EVIDENCE/'trajectory.json',values)
        save(EVIDENCE/'screen.json',screen)
        confirmed={}
        for arm,v in screen.items():
            confirmed[arm]=confirmation(arm,v['selected_step']) if v['passed'] else {'status':'not run','reason':'screen not passed'}
        save(EVIDENCE/'confirmation.json',confirmed)
if __name__=='__main__':main()
