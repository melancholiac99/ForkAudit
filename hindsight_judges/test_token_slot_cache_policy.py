"""Policy guard and exact memo reuse tests; CPU fixtures only."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
from exp1c_backend import ChunkedScorer
import token_slot_backend as K
class CachePolicyTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-7B-Instruct',local_files_only=True)
 def scorer(self,policy,memo):
  torch.manual_seed(11)
  cfg=Qwen2Config(vocab_size=len(self.tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512)
  model=Qwen2ForCausalLM._from_config(cfg,attn_implementation='sdpa').float().eval()
  s=ChunkedScorer(SimpleNamespace(t=torch,model=model),chunk=7);s.slot_cache_policy=policy;s.slot_memo_limit=memo
  return s
 def test_semantic_fp32_cache_preserves_inputs_and_passes_direct(self):
  s=self.scorer('semantic_prefix_fp32',8)
  with torch.inference_mode():
   e=K.Engine(s,self.tok,{'M':'Task','F':'Evidence'})
   for slot in ('A','E'):
    h=self.tok.encode('<think>First consider.',add_special_tokens=False)
    ids,stem,a=e.input(slot,'M',h);original=K.contexts(self.tok,'Task',slot,h)
    self.assertEqual(ids,original[0]);self.assertEqual(stem,original[1]);self.assertEqual(a['recomputed_prefix_tokens'],0)
    lp=e.lp(slot,'M',h);kv,last=s._feed(ids,len(ids)-1)
    direct=torch.log_softmax(s.model.lm_head(last[-1]),0)
    self.assertLess(float((lp.exp()-direct.exp()).abs().sum()/2),.005)
 def test_semantic_cache_cannot_accidentally_apply_to_fp16(self):
  s=self.scorer('semantic_prefix_fp32',8);s.model.half()
  with self.assertRaises(AssertionError):K.Engine(s,self.tok,{'M':'Task','F':'Evidence'})
 def test_larger_memo_skips_only_duplicate_forward_calls(self):
  s=self.scorer('canonical_complete_chunks',2)
  texts=['<think>First','<think>Next','<think>Third','<think>Last']*2
  outputs=[];counts=[]
  with torch.inference_mode():
   for size in (2,8):
    s.slot_memo_limit=size;e=K.Engine(s,self.tok,{'M':'Task','F':'Evidence'});out=[]
    for text in texts:out.append(e.lp('A','M',self.tok.encode(text,add_special_tokens=False)).clone())
    self.assertLessEqual(len(e.memo),size);outputs.append(out);counts.append(e.calls)
   self.assertEqual(counts,[8,4])
   for a,b in zip(*outputs):torch.testing.assert_close(a,b,atol=0,rtol=0)
 def test_parallel_requests_keep_exact_inputs_and_independent_caches(self):
  s=self.scorer('canonical_complete_chunks',32)
  prompts={'M':'Task','F':'Task with additional evidence'}
  h=self.tok.encode('<think>First consider.',add_special_tokens=False)
  with torch.inference_mode():
   serial=K.Engine(s,self.tok,prompts);expected=serial.quartet(h)
   s.slot_workers=4;parallel=K.make_engine(s,self.tok,prompts)
   try:
    for _ in range(2):
     actual=parallel.quartet(h)
     self.assertEqual(list(actual),list(expected))
     for d in expected:
      torch.testing.assert_close(actual[d],expected[d],atol=0,rtol=0)
      self.assertEqual(parallel.input(d[0],d[1],h),serial.input(d[0],d[1],h))
    self.assertEqual(parallel.calls,4)
    self.assertEqual(parallel.audit,serial.audit)
    torch.testing.assert_close(parallel.lp('A','M',h),expected['AM'],atol=0,rtol=0)
    self.assertEqual(sum(e.memo_limit for e in parallel.engines.values()),32)
   finally:parallel.close()
if __name__=='__main__':unittest.main()
