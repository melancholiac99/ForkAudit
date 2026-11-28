"""Second format: emit the edited reference verbatim to avoid losing valid edits to JSON escaping."""
import sys
from expand import pool,done
from check import ROOT,DATA,read,save,boxes
def main(lane):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    path=ROOT/'models/Qwen3-1.7B'
    tok=AutoTokenizer.from_pretrained(path)
    model=LLM(model=str(path),dtype='bfloat16',max_model_len=16384,tensor_parallel_size=1,gpu_memory_utilization=.85,seed=20261128)
    known=set(done()); jobs=[j for j in pool() if j['idx'] not in known][lane::4]
    for start in range(0,len(jobs),32):
        dest=DATA/f'full-{lane}-{start}.json'
        if dest.exists():continue
        batch=jobs[start:start+32]
        prompts=[]
        for j in batch:
            target=boxes(j['negative'])[-1]
            msg=f"""Edit the REFERENCE in place. Preserve its original lines, headings, numbering, language, and all unchanged mathematical steps. Introduce the same specific mistake as the WRONG SOLUTION and propagate it to the final box. The final boxed contents must be exactly {target}. If the wrong solution is actually equivalent, return SKIP. Otherwise output JUST the entire modified reference, with its original formatting. No explanation and no code fence.

PROBLEM
{j['problem']}
WRONG SOLUTION
{j['negative']}
REFERENCE TO COPY AND MINIMALLY EDIT
{j['solution']}"""
            prompts.append(tok.apply_chat_template([dict(role='user',content=msg)],tokenize=False,add_generation_prompt=True,enable_thinking=False))
        outputs=model.generate(prompts,SamplingParams(temperature=0,max_tokens=2500,seed=20261128))
        save(dest,[dict(idx=j['idx'],text=g.outputs[0].text) for j,g in zip(batch,outputs,strict=True)])
if __name__=='__main__':main(int(sys.argv[1]))
