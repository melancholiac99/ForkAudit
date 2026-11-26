"""Four independent vLLM lanes; native generated token IDs survive each continuation."""
import sys
import time
from common_io import *

MODELS=('B','R-cond@100')
LIMIT=16384

def install_forcing(worker,eos,marker):
    """Intercept the synchronized sampled IDs before both the KV runner and scheduler consume them."""
    runner=worker.model_runner
    assert not runner.use_async_scheduling and not getattr(runner,'speculative_config',None)
    original=runner._to_list
    pending={}
    events={}
    runner.forcing_events=events
    def apply(sampled):
        rows=original(sampled)
        batch=runner.input_batch
        discarded=set(runner.discard_request_indices.np[:runner.num_discarded_requests])
        for ix,(rid,row) in enumerate(zip(batch.req_ids,rows)):
            if ix in discarded:continue
            assert len(row)==1
            if rid in pending:
                row[0]=pending[rid].pop(0)
                if not pending[rid]:del pending[rid]
                # vLLM's own discarded-draw bookkeeping uses four Philox offsets.
                generator=batch.generators[ix]
                generator.set_offset(generator.get_offset()-4)
                continue
            if row[0] not in eos:continue
            req=runner.requests[rid]
            sp=req.sampling_params
            j=sp.seed-1001
            assert j in (0,1)
            data=sp.extra_args
            L=data['target_lengths'][j]
            key=(data['forcing_row'],j)
            pos=len(req.output_token_ids)
            if pos+1<L and pos+len(marker)<=sp.max_tokens and len(events.get(key,[]))<8:
                events.setdefault(key,[]).append((pos,row[0]))
                row[0]=marker[0]
                if len(marker)>1:pending[rid]=list(marker[1:])
        return rows
    runner._to_list=apply
    return True

def read_events(worker):
    return worker.model_runner.forcing_events

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
    llm.collective_rpc(install_forcing, args=(list(eos),marker))
    params=[SamplingParams(n=2,seed=1001,temperature=.7,top_p=.8,top_k=20,min_p=0.,
                    presence_penalty=0.,max_tokens=LIMIT,
                    extra_args={'forcing_row':i,'target_lengths':[g['generated_tokens'] for g in ref[i]['generations']]})
            for i in indices]
    outputs=llm.generate(prompts,params,lora_request=request)
    if request:assert 1 in llm.llm_engine.list_loras()
    events,=llm.collective_rpc(read_events)
    result={}
    for i,response in zip(indices,outputs):
        assert len(response.outputs)==2
        for j,ans in enumerate(response.outputs):
            seq=list(ans.token_ids)
            assert ans.index==j and 0<len(seq)<=LIMIT and ans.finish_reason in ('stop','length')
            L=ref[i]['generations'][j]['generated_tokens']
            edits=events.get((i,j),[])
            assert len(edits)<=8
            assert all(pos+1<L and eos_id in eos and seq[pos:pos+len(marker)]==marker for pos,eos_id in edits)
            if ans.finish_reason=='stop' and len(seq)<L and len(seq)-1+len(marker)<=LIMIT:
                assert len(edits)==8
            pred=extract_boxed_answer(ans.text)
            result[i,j]=dict(target=L,tokens=len(seq),insertions=len(edits),forced_at=edits,predicted=pred,
                       correct=grade_answer(pred,ref[i]['ground_truth']),finish_reason=ans.finish_reason,token_ids=seq)
    assert len(result)==2*len(indices)
    for i in indices:save(dest/f'{i}.json',[result[i,j] for j in range(2)])

if __name__=='__main__':measure(int(sys.argv[1]),sys.argv[2],sys.argv[3])
