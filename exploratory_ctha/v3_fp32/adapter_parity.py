"""Check updated unmerged PEFT vs vLLM merged policy before pilot expansion."""
import sys
import numpy as np
from util import *
from precision import check_environment,selected_logprobs

def main():
 check_environment();adapter=DATA/'direction_probe';out=DATA/'adapter_parity';out.mkdir(exist_ok=True)
 if sys.argv[1]=='sample':
  from merged_policy import export_policy
  from vllm import LLM,SamplingParams
  import shutil
  path=export_policy(MODEL,adapter)
  rows=[__import__('json').loads(l) for l in (DATA/'initial_prepared/samples.jsonl').read_text().splitlines()]
  rows=[rows[i] for i in load(adapter/'probe_indices.json')]
  llm=LLM(model=str(path),dtype='float32',seed=0,max_model_len=6144,gpu_memory_utilization=.7,enforce_eager=True,enable_lora=False,worker_cls='fp32_worker.FP32Worker',generation_config='vllm',logprobs_mode='processed_logprobs',max_num_seqs=8,enable_prefix_caching=True)
  result=llm.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in rows],[SamplingParams(temperature=.7,top_p=1.,top_k=-1,max_tokens=512,logprobs=0,seed=1000+i) for i in range(len(rows))],use_tqdm=False)
  saved=[]
  for r in result:
   c=r.outputs[0];saved.append({'prompt_ids':list(r.prompt_token_ids),'completion_ids':list(c.token_ids),'behavior_logprobs':[float(v[t].logprob) for t,v in zip(c.token_ids,c.logprobs)]})
  save(out/'replies.json',saved);llm.llm_engine.engine_core.shutdown();del llm;shutil.rmtree(path)
 else:
  import torch
  from transformers import AutoModelForCausalLM
  from peft import PeftModel
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('cuda')
  model=PeftModel.from_pretrained(model,str(adapter)).eval();errors=[]
  with torch.no_grad():
   for r in load(out/'replies.json'):errors.extend((selected_logprobs(model,r,.7).cpu()-torch.tensor(r['behavior_logprobs'])).abs().tolist())
  assert errors
  pg=load(HERE/'PLAN.json')['parity_gate'];report={'mean_abs':float(np.mean(errors)),'p99_abs':float(np.quantile(errors,.99)),'max_abs':float(np.max(errors)),'tokens':len(errors),'adapter_sha256':sha(adapter/'adapter_model.safetensors')}
  report['passed']=report['mean_abs']<=pg['mean_absolute_sampled_logprob_error_max'] and report['p99_abs']<=pg['p99_absolute_sampled_logprob_error_max'] and report['max_abs']<=pg['max_absolute_sampled_logprob_error_max']
  save(out/'verification.json',report);print(report,flush=True)
  if not report['passed']:raise RuntimeError('Updated adapter merge parity failed; no expansion')
if __name__=='__main__':main()
