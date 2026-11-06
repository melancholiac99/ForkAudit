import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/math-slot-deps')
import math_slot_core as M
class Core(unittest.TestCase):
 def test_seed_namespaces(self):
  values=[M.seed(k,'algebra:3',i) for k in ('student','screen','eval_first32','eval_full') for i in range(4)]
  self.assertEqual(len(values),len(set(values)));self.assertEqual(M.seed('screen','x',3),M.seed('screen','x',3))
 def test_grading(self):
  self.assertTrue(M.grade(r'\boxed{\frac{1}{2}}',r'Thus \boxed{0.5}')['correct'])
  self.assertFalse(M.grade(r'\boxed{2}',r'\boxed{3}')['correct'])
  self.assertTrue(M.grade(r'\boxed{2}',r'\boxed{3}',r'\boxed{3}')['retains_student_answer'])
  self.assertEqual(M.boxed(r'\boxed{2} then \boxed{\frac{3}{4}}'),r'\frac{3}{4}')
  self.assertFalse(M.grade(r'\boxed{2}','No answer can be obtained.')['parseable'])
 def test_cut_errors_need_review(self):
  text='We first work through the arithmetic: $3 + 4 = 8$.\n\nThen continue the derivation carefully.\n\nThe answer is \\boxed{8}.'
  p=M.propose(text);self.assertEqual(p['status'],'review_required');self.assertEqual(len(p['candidate_errors']),1)
  a=dict(prefix_end=p['prefix_end'],cohort='error_observed',reviewer='test',reason='3+4=7',error_quote='$3 + 4 = 8$')
  self.assertEqual(M.validate_annotation(text,a),p['prefix'])
  with self.assertRaises(AssertionError):M.validate_annotation(text,dict(a,error_quote='not present'))
  with self.assertRaises(ValueError):M._number('__import__("os")')
  self.assertIsNone(M.propose('Short final answer \\boxed{8}'))
 def test_prompts_no_answer_or_future(self):
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-7B-Instruct',local_files_only=True)
  problem='Find 1 + 2.';prefix='First consider the calculation.\n\n'
  for arm in M.ARMS:
   p=M.prompt(tok,problem,arm,prefix);decoded=tok.decode(p['input_ids'],clean_up_tokenization_spaces=False)
   self.assertIn(problem,decoded)
   self.assertEqual(prefix in decoded,arm!='INDEPENDENT')
   if arm=='ACTIVE':self.assertTrue(decoded.endswith(prefix))
   else:self.assertTrue(decoded.endswith('<|im_start|>assistant\n'))
   self.assertNotIn('GOLD_SECRET',decoded)
 def test_question_paired_analysis(self):
  grades={arm:dict(correct=arm=='QUOTE-CONTINUE',retains_student_answer=arm=='ACTIVE') for arm in M.ARMS}
  out=M.analysis([dict(id='q',cohort='error_observed',grades=grades)])
  self.assertEqual(out['cohorts']['error_observed']['quote_continue_minus_active_pp']['correct']['ci95'],[100.,100.])
  self.assertEqual(out['cohorts']['error_unknown']['questions'],0)
 def test_decoder_incremental_alignment(self):
  import torch
  from transformers import Qwen2Config,Qwen2ForCausalLM
  from exp1c_backend import ChunkedScorer
  import exp6_math_slot as E
  class Frozen:pass
  f=Frozen();f.t=torch;f.model=Qwen2ForCausalLM(Qwen2Config(vocab_size=24,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).eval()
  s=ChunkedScorer(f,chunk=3)
  class Tok:
   def decode(self,ids,**kwargs):return str(ids)
  req=dict(input_ids=[1,3,4,5,6]);old=E.CONFIG
  E.CONFIG=dict(old,max_new_tokens=5)
  try:
   a=E.generate(s,Tok(),[],req,123);b=E.generate(s,Tok(),[],req,123)
   self.assertEqual(a['ids'],b['ids']);self.assertEqual(a['logp'],b['logp'])
   with torch.inference_mode():
    ids=req['input_ids']+a['ids'][:-1]
    z=s.model(torch.tensor([ids])).logits[0,len(req['input_ids'])-1:].log_softmax(-1)
    actual=[float(z[i,y]) for i,y in enumerate(a['ids'])]
   self.assertTrue(torch.allclose(torch.tensor(actual),torch.tensor(a['logp']),atol=1e-5))
  finally:E.CONFIG=old
if __name__=='__main__':unittest.main()
