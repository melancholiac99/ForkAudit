"""Reconstruct the R stream and cache four independently graded B completions."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from launch import read, save, sha
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
WORK=Path('/mnt/data2/ForkAudit/contrast')
MODEL=ROOT/'models/Qwen3-1.7B'
SEED=20261127


def manifest():
    import torch
    import pyarrow.parquet as pq
    files=sorted((ROOT/'datasets/train/data').glob('*.parquet'))
    rows=[x for f in files for x in pq.ParquetFile(f).read(columns=['problem','solution','Answer']).to_pylist()]
    indices=torch.randperm(len(rows),generator=torch.Generator().manual_seed(42)).tolist()[:6400]
    jobs=[dict(idx=i,**rows[i]) for i in indices]
    assert len({j['problem'] for j in jobs})==6400
    for arm in ('R','R-extension'):
        for f in (ROOT/'run'/arm/'generations').glob('*.json'):
            for x in read(f)['generations']:
                step=x['step']
                q=x['prompt'].split('Problem: ',1)[1].rsplit('\n\nPlease reason',1)[0]
                assert q in [j['problem'] for j in jobs[step*32:(step+1)*32]]
    save(WORK/'manifest.json',jobs)
    save(HERE/'manifest.json',dict(rows=6400,draws=25600,seed=SEED,sha256=sha(WORK/'manifest.json'),
         inputs={f.name:sha(f) for f in files}))


def sample(lane):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/eval')]
    from data_collator import SelfDistillationDataCollator
    from evaluate_math import extract_boxed_answer,grade_answer
    tok=AutoTokenizer.from_pretrained(MODEL)
    collate=SelfDistillationDataCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
    jobs=read(WORK/'manifest.json')[lane::4]
    if all((WORK/f'batch-{lane}-{start}.json').exists() for start in range(0,len(jobs),64)):
        return
    engine=LLM(model=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,max_model_len=20000,seed=SEED,gpu_memory_utilization=.88)
    params=SamplingParams(n=4,temperature=1.,top_p=.95,top_k=20,max_tokens=1024,seed=SEED)
    for start in range(0,len(jobs),64):
        dest=WORK/f'batch-{lane}-{start}.json'
        batch=jobs[start:start+64]
        if dest.exists():
            assert [v['idx'] for v in read(dest)]==[j['idx'] for j in batch]
            continue
        prompts=[dict(prompt_token_ids=collate([j])['student_prompts'][0].tolist()) for j in batch]
        responses=engine.generate(prompts,params)
        records=[]
        for j,p,r in zip(batch,prompts,responses,strict=True):
            gold=j['Answer'] or extract_boxed_answer(j['solution'])
            assert p['prompt_token_ids']==list(r.prompt_token_ids),j['idx']
            if not gold:
                assert j['idx'] in (4807,6651), j['idx']
            answers=sorted(r.outputs,key=lambda g:g.index)
            assert [g.index for g in answers]==list(range(4))
            records.append(dict(idx=j['idx'],answers=[dict(text=g.text,correct=bool(grade_answer(extract_boxed_answer(g.text),str(gold))) if gold else None) for g in answers]))
        save(dest,records)


def combine():
    jobs=read(WORK/'manifest.json')
    rows={v['idx']:v['answers'] for f in WORK.glob('batch-*.json') for v in read(f)}
    assert len(rows)==len(jobs)==6400
    mapping={}
    review=read(HERE/'proof_review.json')
    reviewed=[]
    for j in jobs:
        batch=rows[j['idx']]
        assert len(batch)==4
        if any(x['correct'] is None for x in batch):
            assert all(x['correct'] is None for x in batch)
            import hashlib
            record=review[str(j['idx'])]
            assert record['problem']==j['problem'] and record['sample_sha256']==[hashlib.sha256(x['text'].encode()).hexdigest() for x in batch]
            reviewed.append(j['idx'])
            mapping[j['problem']]=next((x['text'] for x,g in zip(batch,record['grades']) if g=='incorrect'),None)
            assert mapping[j['problem']] is not None or record['grades']==['correct']*4
            continue
        mapping[j['problem']]=next((x['text'] for x in batch if not x['correct']),None)
    assert sorted(reviewed)==[4807,6651]
    save(WORK/'negative.json',mapping)
    missing=sum(v is None for v in mapping.values())
    save(HERE/'negative_contexts.json',dict(tasks=len(mapping),proof_review_ids=sorted(reviewed),officially_graded_tasks=6398,all_four_correct=missing,all_four_correct_pct=100*missing/len(mapping),
         no_negative_pct=100*missing/len(mapping),sha256=sha(WORK/'negative.json'),manifest_sha256=sha(WORK/'manifest.json')))


if __name__=='__main__':
    if sys.argv[1]=='manifest':manifest()
    elif sys.argv[1]=='sample':sample(int(sys.argv[2]))
    elif sys.argv[1]=='combine':combine()
