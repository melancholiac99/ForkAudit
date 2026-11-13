"""Portable CPU CI: no server data, GPU, model weights, or environment simulator."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_cpu import checks
from action_mask import action_mask
from validate_pipeline import acceptance_states
class Tests(unittest.TestCase):
 def test_optimizer_batch_and_objective(self):self.assertTrue(all(checks().values()))
 def test_unicode_offsets_and_noncanonical_tokens(self):
  class Tok:
   def decode(self,ids,**kw):return ''.join(map(chr,ids))
   def __call__(self,text,**kw):return {'input_ids':list(map(ord,text)),'offset_mapping':[(i,i+1) for i in range(len(text))]}
  text='<think>观察苹果</think>[action] take apple 1 [/action]';ids=list(map(ord,text));m=action_mask(Tok(),ids)
  self.assertEqual(''.join(chr(t) for t,v in zip(ids,m) if v),'take apple 1')
  class Bad(Tok):
   def __call__(self,text,**kw):return {'input_ids':[0],'offset_mapping':[(0,len(text))]}
  self.assertIsNone(action_mask(Bad(),ids))
 def test_noncanonical_bytelevel_tokens(self):
  class ByteLevel:pass
  class Tok:
   all_special_ids=[]
   backend_tokenizer=type('Backend',(),{'decoder':ByteLevel()})()
   tokens=['<','think','>reason</think>[action]','Ġgo','Ġto','Ġtable','Ġ1','Ġ[/action]']
   def decode(self,ids,**kw):return ''.join(self.tokens[i] for i in ids).replace('Ġ',' ')
   def convert_ids_to_tokens(self,ids):return [self.tokens[i] for i in ids]
   def __call__(self,text,**kw):return {'input_ids':[100],'offset_mapping':[(0,len(text))]}
  t=Tok();ids=list(range(len(t.tokens)));m=action_mask(t,ids)
  self.assertEqual(t.decode([i for i,v in zip(ids,m) if v]),' go to table 1')
 def test_completed_is_not_effectiveness(self):
  r={'grpo':{'train':{'paired_change':.05,'ci95':[-.1,.2]},'tune':{'paired_change':.1,'ci95':[-.03,.23]}}}
  s=acceptance_states(r);self.assertEqual(s['learning']['grpo'],'NOT_DEMONSTRATED');self.assertEqual(s['generalization']['grpo'],'NOT_DEMONSTRATED')
 def test_positive_ci_is_qualified_pilot_evidence(self):
  r={'grpo':{'train':{'paired_change':.2,'ci95':[.1,.3]},'tune':{'paired_change':.1,'ci95':[.01,.2]}}}
  s=acceptance_states(r);self.assertEqual(s['learning']['grpo'],'PASS_IN_PILOT');self.assertEqual(s['replication_across_training_seeds'],'NOT_TESTED');self.assertEqual(s['ctha_training'],'NOT_RUN')
if __name__=='__main__':unittest.main()
