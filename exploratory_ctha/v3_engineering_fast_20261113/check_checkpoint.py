"""Restore the actual probe checkpoint, including Adam moments; CPU only."""
import sys,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v3_fp32_fast'))
from util import *
def main():
 import torch
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 torch.set_num_threads(1)
 d=DATA/'direction_probe';saved=torch.load(d/'optimizer.pt',map_location='cpu',weights_only=True)
 log=load(d/'train_log.json');assert log['complete'] and saved['total_updates']==4
 assert saved['adapter_sha']==sha(d/'adapter_model.safetensors')==log['weights_sha']
 actor=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
 model=PeftModel.from_pretrained(actor,str(d),is_trainable=True)
 params=[p for p in model.parameters() if p.requires_grad]
 named=[n for n,p in model.named_parameters() if p.requires_grad]
 assert len(params)==len(saved['optimizer']['state'])
 spec=load(HERE.parent/'v3_fp32_fast/PLAN.json')['optimizer']
 optimizer=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=0.)
 optimizer.load_state_dict(saved['optimizer'])
 # Verify trainer-order restore of the actual serialized moments, not empty state.
 for p,key in zip(params,saved['optimizer']['param_groups'][0]['params']):
  before=saved['optimizer']['state'][key];after=optimizer.state[p]
  assert before['exp_avg'].shape==p.shape and before['exp_avg_sq'].shape==p.shape
  assert after['step'].item()==saved['total_updates']
  for k in ['step','exp_avg','exp_avg_sq']:assert torch.equal(before[k],after[k])
 # Independent live branch and disk round-trip branch with identical fixed gradients.
 clones=[torch.nn.Parameter(p.detach().clone()) for p in params]
 replay=torch.optim.AdamW(clones,lr=spec['lr'],weight_decay=0.)
 second=torch.load(d/'optimizer.pt',map_location='cpu',weights_only=True)
 replay.load_state_dict(second['optimizer'])
 for i,(p,q) in enumerate(zip(params,clones)):
  value=((i%7)-3)*1e-4
  p.grad=torch.full_like(p,value);q.grad=torch.full_like(q,value)
 optimizer.step();replay.step()
 for p,q in zip(params,clones):
  assert torch.equal(p,q)
  for k in ['step','exp_avg','exp_avg_sq']:assert torch.equal(optimizer.state[p][k],replay.state[q][k])
  assert optimizer.state[p]['step'].item()==5
 assert sha(d/'adapter_model.safetensors')==saved['adapter_sha']
 result={'passed':True,'actual_saved_checkpoint':str(d),'weights_sha256':saved['adapter_sha'],'optimizer_sha256':sha(d/'optimizer.pt'),'restored_updates':4,'next_test_updates':5,'trainable_tensors':len(params),'trainable_parameters':sum(p.numel() for p in params),'parameter_names':named,'moments_equal':True,'next_step_equal':True,'device':'cpu','gradients':'fixed deterministic engineering inputs, not rollout rewards','checkpoint_overwritten':False,'online_second_round_tested':False}
 save(DATA/'checkpoint_verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
