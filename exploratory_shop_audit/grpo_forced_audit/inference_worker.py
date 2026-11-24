"""Frozen checkpoint inference for pilot, evaluation, and independent audit banks."""
import argparse
import contextlib
import json
import os
import sys
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter');a=p.parse_args()
    rpc=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
    with contextlib.redirect_stdout(sys.stderr):
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        options={}
        tape=None
        logprob_audit=os.environ.get('OPD_RAW_LOGPROBS')=='1'
        if logprob_audit:
            options['logprobs_mode']='raw_logprobs'
        if 'OPD_ENTROPY_TAPE' in os.environ:
            import numpy as np
            from exploratory_shop_audit.visible_opd_v0.d_entropy import NativeEntropy
            tape=np.memmap(os.environ['OPD_ENTROPY_TAPE'],dtype='float32',mode='w+',shape=(64,512))
            options['logits_processors']=[NativeEntropy]
        model=LLM(model=a.model,tokenizer=a.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8704,
            gpu_memory_utilization=.7,max_num_seqs=16,enable_prefix_caching=True,seed=20261117,
            enable_lora=bool(a.adapter),max_lora_rank=64,disable_log_stats=True,**options)
    adapter=LoRARequest('frozen_checkpoint',1,a.adapter) if a.adapter else None
    rpc.write(json.dumps({'ready':True})+'\n')
    for line in sys.stdin:
        req=json.loads(line)
        if req.get('cmd')=='quit':return
        try:
            if req.get('cmd') == 'reset_replay_cache':
                # Only between completed logical §26 shards, with no outstanding requests.
                if model.llm_engine.has_unfinished_requests():
                    raise RuntimeError('Cannot reset active inference')
                model.reset_prefix_cache()
                model.request_counter.reset()
                rpc.write(json.dumps({'ok':True,'result':'reset'})+'\n')
                continue
            rows=req['requests']
            if tape is not None:
                assert len(rows)<=len(tape)
                tape[:len(rows)]=np.nan
            params=[SamplingParams(temperature=r.get('temperature',.7),top_p=1,max_tokens=512,seed=r['seed'],
                    logprobs=0 if logprob_audit else None,extra_args={'entropy_slot':i} if tape is not None else None) for i,r in enumerate(rows)]
            with contextlib.redirect_stdout(sys.stderr):
                outputs=model.generate([{'prompt_token_ids':r['token_ids']} for r in rows],params,lora_request=adapter,use_tqdm=False)
            results=[]
            for row,out in zip(rows,outputs):
                assert list(out.prompt_token_ids)==row['token_ids'];reply=out.outputs[0]
                results.append({'text':reply.text,'token_ids':list(reply.token_ids),'finish_reason':reply.finish_reason})
                if logprob_audit:
                    assert len(reply.logprobs)==len(reply.token_ids)
                    results[-1]['raw_logprobs']=[d[tok].logprob for d,tok in zip(reply.logprobs,reply.token_ids,strict=True)]
                if tape is not None:
                    entropy=tape[len(results)-1,:len(reply.token_ids)]
                    assert np.isfinite(entropy).all() and (entropy>=0).all(), 'missing native token entropy'
                    results[-1]['entropy']=entropy.tolist()
            response={'ok':True,'result':results}
        except Exception:response={'ok':False,'error':traceback.format_exc()}
        rpc.write(json.dumps(response,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
