"""Check archived behavior scores with a fresh, frozen vLLM adapter."""
import json
import os
import time


def main():
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    import torch
    from verl import DataProto
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from .common import DEFAULT_RUN, MODEL, save, utc
    run = DEFAULT_RUN.resolve(); folder = run/'alignment_diagnostic/checkpoint_0002'
    if (folder/'fresh_vllm.json').exists():
        raise FileExistsError('Preserve the completed frozen sampler check')
    packet = DataProto.load_from_disk(str(next((run/'native_batch_cache').glob('train_0002_sample_*.pkl'))))
    packet.to('cpu')
    examples = json.loads((folder/'largest_differences_with_advantage.json').read_text())[:12]
    requests = []
    for row in examples:
        i,j = row['row'],row['position']
        ids = packet.batch['prompts'][i]
        mask = packet.batch['attention_mask'][i,:len(ids)].bool()
        requests.append({'prompt_token_ids':ids[mask].tolist()+packet.batch['responses'][i,:j].tolist()})
    started = time.monotonic()
    model = LLM(model=str(MODEL), dtype='bfloat16', tensor_parallel_size=1,
                max_model_len=8704, gpu_memory_utilization=.25, kv_cache_memory_bytes=536870912,
                max_num_seqs=1, max_num_batched_tokens=1024, enforce_eager=True,
                enable_lora=True,max_lora_rank=64,max_loras=1,enable_prefix_caching=False,
                logprobs_mode='processed_logprobs',disable_log_stats=True)
    params = SamplingParams(temperature=.7, top_p=1, top_k=-1, max_tokens=1, logprobs=20, seed=20261117)
    adapter=LoRARequest('checkpoint2',2,str(run/'training/checkpoints/global_step_2/actor/lora_adapter'))
    result=[]
    outputs=model.generate(requests,params,lora_request=adapter,use_tqdm=False)
    for row, output in zip(examples,outputs):
        scores=output.outputs[0].logprobs[0]
        score=scores.get(row['token_id'])
        result.append({**row,'fresh_vllm_logp':None if score is None else score.logprob,
                       'diagnostic_generated_token':output.outputs[0].token_ids,
                       'top_tokens':{str(k):{'text':v.decoded_token,'logprob':v.logprob} for k,v in scores.items()}})
    save(folder/'fresh_vllm.json',{'time':utc(),'seconds':time.monotonic()-started,
                                 'generation_requests':len(requests),'max_new_tokens_per_request':1,
                                 'formal_trajectories':0,'eager':True,'records':result})
    print('COMPLETED',len(result),flush=True)


if __name__ == '__main__':main()
