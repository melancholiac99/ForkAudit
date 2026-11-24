import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import projected_logits

class ProjectionTest(unittest.TestCase):
    def test_sparse_teacher(self):
        torch.manual_seed(42)
        base=Qwen3ForCausalLM(Qwen3Config(vocab_size=83,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)).eval()
        ids=torch.randint(83,(2,9))
        mask=torch.ones_like(ids)
        with torch.no_grad():
            logits=base(ids,attention_mask=mask,use_cache=False).logits[:,3:-1]
            h=base.model(ids,attention_mask=mask,use_cache=False).last_hidden_state[:,3:-1]
            A,B=torch.randn(8,32),torch.randn(8,32)
            d=(h@A.T)@(base.lm_head.weight@B.T).T
            idx=logits.topk(64).indices
            expected=logits.clone().scatter_add_(-1,idx,.5*d.gather(-1,idx))
            torch.testing.assert_close(projected_logits(base,ids,mask,3,dict(A=A,B=B,scale=.5)),expected)
            self.assertTrue(torch.equal(projected_logits(base,ids,mask,3,{'scale':0}),logits))
