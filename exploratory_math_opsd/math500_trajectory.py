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
OUT=ROOT/'run/math500-trajectory'
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
    assert [result[k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','max_new_tokens')]==[.7,.8,-1,0.,0.,4096]
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
                se_pp=float(np.std(np.array(scores)*25,ddof=1)/np.sqrt(500)),
                mean_generated_tokens=tokens/2000)

def cell(model,rows,hashes):
    folder=OUT/model
    adapter=None if model=='B' else str(ROOT/'run'/model.split('@')[0]/('checkpoint-'+model.split('@')[1]))
    fingerprint=dict(data_sha256=DIGEST,adapter_sha256=hashes.get(model),seed=1001,
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
    cmd=[sys.executable,'eval/evaluate_math.py','--base_model',str(ROOT/'models/Qwen3-1.7B'),
         '--dataset','math500','--local_eval_file',str(DATA/'test.parquet'),'--seed','1001','--val_n','4',
         '--no_thinking','--temperature','.7','--top_p','.8','--top_k','-1','--min_p','0','--presence_penalty','0',
         '--max_new_tokens','4096','--tensor_parallel_size','1','--output_file',str(merged)]
    if adapter:cmd+=['--checkpoint_dir',adapter]
    result,seconds=run(ROOT,os.environ.copy(),cmd,folder/'shards','math500-'+model,call,read,questions=500,samples=4)
    metrics=summarize(result,rows,adapter)
    save(merged,result)
    rec=dict(**fingerprint,output_sha256=sha(merged),metrics=metrics,wall_seconds=seconds)
    save(receipt,rec);return rec

def select(cells):
    return {a:max(STEPS,key=lambda k:(cells[f'{a}@{k}']['metrics']['correct'],-k)) for a in ARMS}

def main():
    rows,hashes=prepare()
    if '--prepare' in sys.argv:return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cells={}
        for model in ['B']+[f'{a}@{k}' for a in ARMS for k in STEPS]:
            cells[model]=cell(model,rows,hashes)
            save(OUT/'progress.json',cells)
        selected=select(cells)
        save(EVIDENCE/'trajectory.json',cells)
        path=EVIDENCE/'selection.json'
        if path.exists():assert read(path)==selected
        else:save(path,selected)
        public=['# MATH-500 非思考模式轨迹','','每题四次采样，500 题，种子 1001。标准误按题计算。','',
                '| 模型 | 准确率 | 标准误（百分点） | 平均生成 token 数 |','|---|---:|---:|---:|']
        for model,c in cells.items():
            m=c['metrics'];public.append(f"| {model} | {m['accuracy_percent']:.2f}% | {m['se_pp']:.2f} | {m['mean_generated_tokens']:.1f} |")
        (EVIDENCE/'TRAJECTORY.md').write_text('\n'.join(public)+'\n')
        results={}
        for a,k in selected.items():
            model=f'{a}@{k}'
            results[model]={}
            for bench in BENCHES:
                receipt=ROOT/'run/evaluation'/model/bench/'1001.receipt.json'
                reused=receipt.exists()
                rec=evaluate(ROOT,os.environ.copy(),model,bench,1001)
                results[model][bench]=dict(receipt=rec,accuracy_percent=rec['rate'],selection_step=k)
                print(model,bench,'verified existing' if reused else 'completed',flush=True)
        save(EVIDENCE/'selected_benchmarks.json',results)
if __name__=='__main__':main()
