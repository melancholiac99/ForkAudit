"""Continue both forced first-token conditions using a common R policy and remaining budget."""
import numpy as np
import os
from settings import *

def selected(labels):
    sites=sorted((int(i),int(pos)) for i,r in labels.items() for pos in r['sites'])
    rng=np.random.default_rng(SEED+1)
    pil=set(rng.choice(len(sites),min(32,len(sites)),replace=False).tolist())
    return [(i,pos,j in pil) for j,(i,pos) in enumerate(sites)]

def run(lane,stage):
    import torch
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ROOT/'source/eval'))
    from evaluate_math import extract_boxed_answer,grade_answer
    sys.path.insert(0,str(HERE.parents[3]/'pmath'))
    from common import MODEL,WORK
    labels=read(RAW/'labels.json'); jobs=[]
    sites=selected(labels)
    # Balance the pilot and remainder separately so the pilot uses all four engines.
    lanes=int(os.environ.get('REFLECTION_LANES','4'))
    assert lanes in (2,3,4) and 0<=lane<lanes
    for subset in (True,False):
        if stage=='pilot' and not subset:continue
        jobs.extend([(i,pos) for i,pos,pilot in sites if pilot==subset][lane::lanes])
    jobs=[(i,pos) for i,pos in jobs if not (RAW/f'fork-{i}-{pos}.json').exists()]
    if not jobs:return
    records={i:(sp,y) for j in range(4) for i,sp,_,y in torch.load(WORK/f'rollouts-{j}.pt',weights_only=True)}
    ck=ROOT/'run/R/checkpoint-100'
    import hashlib
    digest=hashlib.sha256((ck/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert digest==read(HERE.parents[2]/'internalization/inputs.json')['adapters']['R@100']
    tok=AutoTokenizer.from_pretrained(MODEL);eos=read(MODEL/'generation_config.json')['eos_token_id']
    llm=LLM(model=str(MODEL),dtype='bfloat16',enable_lora=True,max_lora_rank=64,
            max_model_len=20000,max_num_seqs=64,tensor_parallel_size=1,gpu_memory_utilization=.84,seed=lane)
    adapter=LoRARequest('R100',1,str(ck))
    handoff=os.environ.get('REFLECTION_HANDOFF')
    batch=16 if handoff else 64
    for start in range(0,len(jobs),batch):
        if handoff and Path(handoff).exists():break
        prompts=[];params=[];keys=[];parts={}
        for i,pos in jobs[start:start+batch]:
            with np.load(RAW/f'score-{i}.npz') as z:first=z['first'][list(z['sites']).index(pos)]
            sp,y=records[i]
            for arm in (0,1):
                for rep,t in enumerate(first[arm]):
                    t=int(t);key=(i,pos,arm,rep);parts[key]=[t]
                    if t in eos or pos+1==1024:continue
                    prompts.append(dict(prompt_token_ids=sp+y[:pos]+[t]))
                    seed=int(np.random.SeedSequence([SEED,i,pos,rep]).generate_state(1)[0])
                    params.append(SamplingParams(n=1,seed=seed,temperature=1.1,top_k=20,top_p=.95,
                                                 max_tokens=1024-pos-1,repetition_penalty=1.,presence_penalty=0.,min_p=0.))
                    keys.append(key)
        if keys:
            out=llm.generate(prompts,params,lora_request=adapter,use_tqdm=False)
            assert len(out)==len(keys) and 1 in llm.llm_engine.list_loras()
            for key,req,o in zip(keys,prompts,out):
                assert list(o.prompt_token_ids)==req['prompt_token_ids']
                parts[key].extend(o.outputs[0].token_ids)
        for i,pos in jobs[start:start+batch]:
            rows=[];_,y=records[i]
            for arm in (0,1):
                for rep in range(4):
                    tokens=parts[i,pos,arm,rep]
                    pred=extract_boxed_answer(tok.decode(y[:pos]+tokens,skip_special_tokens=True))
                    assert 0<len(tokens)<=1024-pos
                    rows.append(dict(arm=arm,rep=rep,first=tokens[0],remaining=len(tokens),correct=grade_answer(pred,labels[str(i)]['gold']),
                                     token_ids=tokens,cap=len(tokens)==1024-pos,eos=tokens[-1] in eos,predicted=pred))
            save(RAW/f'fork-{i}-{pos}.json',rows)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
