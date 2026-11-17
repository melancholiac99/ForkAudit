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
        model=LLM(model=a.model,tokenizer=a.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8704,
            gpu_memory_utilization=.7,max_num_seqs=16,enable_prefix_caching=True,seed=20261117,
            enable_lora=bool(a.adapter),max_lora_rank=64,disable_log_stats=True)
    adapter=LoRARequest('frozen_checkpoint',1,a.adapter) if a.adapter else None
    rpc.write(json.dumps({'ready':True})+'\n')
    for line in sys.stdin:
        req=json.loads(line)
        if req.get('cmd')=='quit':return
        try:
            rows=req['requests'];params=[SamplingParams(temperature=r.get('temperature',.7),top_p=1,max_tokens=512,seed=r['seed']) for r in rows]
            with contextlib.redirect_stdout(sys.stderr):
                outputs=model.generate([{'prompt_token_ids':r['token_ids']} for r in rows],params,lora_request=adapter,use_tqdm=False)
            results=[]
            for row,out in zip(rows,outputs):
                assert list(out.prompt_token_ids)==row['token_ids'];reply=out.outputs[0]
                results.append({'text':reply.text,'token_ids':list(reply.token_ids),'finish_reason':reply.finish_reason})
            response={'ok':True,'result':results}
        except Exception:response={'ok':False,'error':traceback.format_exc()}
        rpc.write(json.dumps(response,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
