"""Explicit FP32 forwards; no implicit BF16 autocast inherited from old trainer."""
import os

def check_environment():
 import importlib.metadata
 assert importlib.metadata.version('vllm')=='0.32.0'
 for key,value in {'VLLM_BATCH_INVARIANT':'1','TRITON_F32_DEFAULT':'tf32x3','NVIDIA_TF32_OVERRIDE':'0'}.items():
  if os.getenv(key)!=value:raise RuntimeError(f'Registered precision requires {key}={value}')

def selected_logprobs(model, example, temperature):
 import torch
 pi,ri=example['prompt_ids'],example['completion_ids'];device=next(model.parameters()).device
 assert next(model.parameters()).dtype==torch.float32
 with torch.autocast('cuda',enabled=False):
  logits=model(input_ids=torch.tensor([pi+ri],dtype=torch.long,device=device),use_cache=False).logits
 assert logits.dtype==torch.float32
 window=logits[0,len(pi)-1:len(pi)+len(ri)-1]
 assert window.shape[0]==len(ri)
 return (window/temperature).log_softmax(-1).gather(-1,torch.tensor(ri,device=device)[:,None]).squeeze(-1)
