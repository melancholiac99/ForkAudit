"""BF16 transformer / FP32 vocabulary projection, one candidate at a time to bound KV memory."""
import copy
MODEL=None
class Scorer:
 def __init__(self):
  import torch
  from transformers import AutoModelForCausalLM
  self.t=torch
  self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
  self.model.lm_head.to(dtype=torch.float32)
 def score(self,job):
  t,m=self.t,self.model;pre=job['prefix_ids'];cmds=[c['suffix_ids'] for c in job['candidates']]
  with t.inference_mode():
   out=m.model(input_ids=t.tensor([pre],device='cuda'),use_cache=True)
   lp0=t.log_softmax(m.lm_head(out.last_hidden_state[0,-1].float()),-1)
   cache=out.past_key_values
   del out
   rows=[]
   for cmd in cmds:
    kv=copy.deepcopy(cache)
    h=m.model(input_ids=t.tensor([cmd[:-1]],device='cuda'),past_key_values=kv,use_cache=False).last_hidden_state
    lp=t.log_softmax(m.lm_head(h[0].float()),-1)
    rows.append([float(lp0[cmd[0]])]+lp[t.arange(len(cmd)-1,device='cuda'),t.tensor(cmd[1:],device='cuda')].tolist())
    del kv,h,lp
   return rows
 def direct(self,job,arm):
  t,m=self.t,self.model;pre=job['prefix_ids'];cmd=job['candidates'][arm]['suffix_ids'];ids=pre+cmd
  with t.inference_mode():
   h=m.model(input_ids=t.tensor([ids],device='cuda'),use_cache=False).last_hidden_state
   lp=t.log_softmax(m.lm_head(h[0,len(pre)-1:len(ids)-1].float()),-1)
   return lp[t.arange(len(cmd),device='cuda'),t.tensor(cmd,device='cuda')].tolist()
