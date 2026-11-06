from pathlib import Path
import sys,unittest
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import token_slot_backend as K
import torch
from transformers import AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
from exp1c_backend import ChunkedScorer
from types import SimpleNamespace
class TestAlignment(unittest.TestCase):
 def test_exact_chunked_reference_and_unchanged_messages(self):
  tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-7B-Instruct',local_files_only=True)
  torch.manual_seed(11)
  model=Qwen2ForCausalLM._from_config(Qwen2Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512),attn_implementation='sdpa').float().eval()
  with torch.inference_mode():
   for chunk in (7,1024):
    scorer=ChunkedScorer(SimpleNamespace(t=torch,model=model),chunk=chunk)
    engine=K.Engine(scorer,tok,{'M':'Task','F':'Task with evidence'})
    for text in ['<think>First','<think>First consider the room. '+('Reason carefully. '*8)]:
     h=tok.encode(text,add_special_tokens=False)
     for slot in ('A','E'):
      for evidence in ('M','F'):
       ids,stem,audit=engine.input(slot,evidence,h);oldids,oldstem,oldaudit=K.contexts(tok,engine.prompts[evidence],slot,h)
       self.assertEqual(ids,oldids);self.assertEqual({k:audit[k] for k in oldaudit},oldaudit);self.assertEqual(len(stem)%chunk,0)
       lp=engine.lp(slot,evidence,h);kv,last=scorer._feed(ids,len(ids)-1)
       direct=torch.log_softmax(model.lm_head(last[-1]),0)
       torch.testing.assert_close(lp,direct,atol=0,rtol=0)
       base=engine.bases[slot,evidence][0]
       self.assertEqual(base.get_seq_length() if base is not None else 0,len(stem))
       torch.testing.assert_close(engine.lp(slot,evidence,h),lp,atol=0,rtol=0)
if __name__=='__main__':unittest.main()
