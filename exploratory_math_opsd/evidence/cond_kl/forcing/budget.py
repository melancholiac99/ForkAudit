"""Four independent vLLM lanes; native generated token IDs survive each continuation."""
import sys
import time
from common_io import *

MODELS=('B','R-cond@100')
LIMIT=16384

def measure(lane,mode,model):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    sys.path.insert(0,str(ROOT/'source/eval'))
    from evaluate_math import load_vllm_model,extract_boxed_answer,grade_answer
    ref=archived('R@100')
    path=ROOT/'run/R-cond/checkpoint-100'
    adapter=None if model=='B' else path
    if adapter:
        assert hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()==read(ROOT/'run/R-cond/complete.json')['adapters']['100']
    dest=RAW/model
    dest.mkdir(exist_ok=True)
    pilot=set((np.arange(20)*25+12).tolist())
    # Distribute pilot across all lanes. Retain the original full-evaluation row order.
    indices=[i for i in range(lane,500,4) if (mode=='full' or i in pilot) and not (dest/f'{i}.json').exists()]
    if not indices:return
    llm,tok=load_vllm_model(str(ROOT/'models/Qwen3-1.7B'),str(adapter) if adapter else None,enable_thinking=False)
    request=None if not adapter else LoRARequest('R-cond',1,str(adapter))
    marker=tok.encode('\n\nWait,',add_special_tokens=False)
    assert tok.decode(marker)=='\n\nWait,'
    eos=set(read(ROOT/'models/Qwen3-1.7B/generation_config.json')['eos_token_id'])
    prompts=[tok.apply_chat_template([dict(role='user',content=ref[i]['problem']+'\n\nPlease reason step by step, and put your final answer within \\boxed{}.')],tokenize=False,add_generation_prompt=True,enable_thinking=False) for i in indices]
    def params(n,seed,limit):
        return SamplingParams(n=n,seed=seed,temperature=.7,top_p=.8,top_k=20,min_p=0.,presence_penalty=0.,max_tokens=limit)
    outputs=llm.generate(prompts,params(2,1001,LIMIT),lora_request=request)
    if request:assert 1 in llm.llm_engine.list_loras()
    history={}; current={}; prompt_ids={}; active=[]; result={}
    for i,response in zip(indices,outputs):
        prompt_ids[i]=list(response.prompt_token_ids)
        assert len(response.outputs)==2
        for j,ans in enumerate(response.outputs):
            assert ans.index==j
            key=i,j
            history[key]=[];current[key]=ans;active.append(key)
    for round_id in range(9):
        pending=[];requests=[];sampling=[]
        for key in active:
            i,j=key; ans=current[key]; seq=history[key]
            seq.extend(ans.token_ids)
            L=ref[i]['generations'][j]['generated_tokens']
            assert len(seq)<=LIMIT and ans.finish_reason in ('stop','length')
            if ans.finish_reason=='stop':
                assert seq[-1] in eos, (ans.stop_reason,seq[-1])
            if ans.finish_reason=='stop' and len(seq)<L and round_id<8 and len(seq)-1+len(marker)<=LIMIT:
                seq.pop();seq.extend(marker)
                if len(seq)==LIMIT:
                    pred=extract_boxed_answer(tok.decode(seq,skip_special_tokens=True))
                    result[key]=dict(target=L,tokens=LIMIT,insertions=round_id+1,predicted=pred,correct=grade_answer(pred,ref[i]['ground_truth']),finish_reason='length',token_ids=seq)
                    continue
                pending.append(key);requests.append(dict(prompt_token_ids=prompt_ids[i]+seq))
                sampling.append(params(1,1001+j+(round_id+1)*1009,LIMIT-len(seq)))
                continue
            text=tok.decode(seq,skip_special_tokens=True)
            pred=extract_boxed_answer(text)
            result[key]=dict(target=L,tokens=len(seq),insertions=round_id,predicted=pred,correct=grade_answer(pred,ref[i]['ground_truth']),finish_reason=ans.finish_reason,token_ids=seq)
        if not pending:break
        assert round_id<8
        outputs=llm.generate(requests,sampling,lora_request=request)
        for k,res in zip(pending,outputs):current[k]=res.outputs[0]
        active=pending
    assert len(result)==2*len(indices)
    for i in indices:save(dest/f'{i}.json',[result[i,j] for j in range(2)])

if __name__=='__main__':measure(int(sys.argv[1]),sys.argv[2],sys.argv[3])
