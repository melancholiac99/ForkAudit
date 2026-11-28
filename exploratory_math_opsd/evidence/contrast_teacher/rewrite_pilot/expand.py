"""Draft small exact edits to the archived reference. No edits are applied to the positive branch."""
import sys
from pathlib import Path
import json
from check import ROOT,HERE,DATA,read,save,boxes
from score import WORK
def pool():
    neg=read(WORK/'negative.json')
    return [dict(j,negative=neg[j['problem']]) for j in read(WORK/'manifest.json')[:3200] if neg[j['problem']]]
def done():
    d={}
    for fn in ('pilot','diagnostic_edits'):
        for r in read(HERE/(fn+'.json')):d[r['idx']]=r
    return d
def sample(lane):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    path=ROOT/'models/Qwen3-1.7B'
    tok=AutoTokenizer.from_pretrained(path)
    engine=LLM(model=str(path),dtype='bfloat16',max_model_len=16384,tensor_parallel_size=1,gpu_memory_utilization=.85,seed=20261128)
    params=SamplingParams(temperature=0,max_tokens=2400,seed=20261128)
    rows=[r for r in pool() if r['idx'] not in done()][lane::4]
    for offset in range(0,len(rows),40):
        dest=DATA/f'drafts-{lane}-{offset}.json'
        if dest.exists():continue
        batch=rows[offset:offset+40];prompts=[]
        for r in batch:
            target=boxes(r['negative'])[-1]
            system="""You edit math training examples. Identify the specific mathematical mistake in a wrong solution. Write an intentionally wrong but INTERNALLY COHERENT variant of the reference using exactly that mistake. Preserve unchanged words, lines and headings, replacing only the minimal fragments necessary, including all later computations and the last boxed answer. Do not correct the wrong method. If the answers are equivalent, choice letter vs value, or this needs major changes, say SKIP. Return JSON only: {"error":"specific B mistake","evidence":"short exact quote from B showing the mistake","edits":[["exact old substring from reference","replacement"], ...]}. Each edit replaces all occurrences. No newlines added or removed, no Markdown fence. JSON-escape LaTeX backslashes. The correct answer must no longer occur. An isolated boxed-answer swap without supporting reasoning is invalid."""
            task=f"Problem:\n{r['problem']}\nOfficial reference:\n{r['solution']}\nB wrong solution:\n{r['negative']}\nRequired final boxed contents: {target}"
            prompts.append(tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=task)],tokenize=False,add_generation_prompt=True,enable_thinking=False))
        generations=engine.generate(prompts,params)
        save(dest,[dict(idx=r['idx'],draft=g.outputs[0].text) for r,g in zip(batch,generations,strict=True)])
if __name__=='__main__':sample(int(sys.argv[1]))
