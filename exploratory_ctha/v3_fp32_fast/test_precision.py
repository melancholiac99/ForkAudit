"""CPU regression of explicit precision and shifted sampled-token probabilities."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from precision import selected_logprobs,check_environment
from unittest.mock import patch
class Tests(unittest.TestCase):
 def test_shift_temperature_gradient_no_bf16(self):
  class Model(torch.nn.Module):
   def __init__(self):
    super().__init__();self.weight=torch.nn.Parameter(torch.arange(24,dtype=torch.float32).reshape(1,4,6)/10)
   def forward(self,input_ids,use_cache):
    assert input_ids.tolist()==[[0,1,2,3]] and not use_cache
    return type('Output',(),{'logits':self.weight})()
  m=Model();r={'prompt_ids':[0,1],'completion_ids':[2,3]}
  actual=selected_logprobs(m,r,.7);expected=(m.weight[0,1:3]/.7).log_softmax(-1)[torch.arange(2),torch.tensor([2,3])]
  self.assertTrue(torch.equal(actual,expected));(-actual.mean()).backward();self.assertTrue(torch.isfinite(m.weight.grad).all())
  with self.assertRaises(AssertionError):selected_logprobs(m.to(torch.bfloat16),r,.7)
 def test_precision_mismatch_stops(self):
  with patch('importlib.metadata.version',return_value='0.32.0'),patch.dict('os.environ',{'VLLM_BATCH_INVARIANT':'1','TRITON_F32_DEFAULT':'tf32x3','NVIDIA_TF32_OVERRIDE':'0'},clear=True):
   check_environment()
   with patch.dict('os.environ',{'TRITON_F32_DEFAULT':'tf32'}):
    with self.assertRaises(RuntimeError):check_environment()
if __name__=='__main__':unittest.main()
