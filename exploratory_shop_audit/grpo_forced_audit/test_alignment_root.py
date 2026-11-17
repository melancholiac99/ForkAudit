"""Check the diagnostic attention reference independently of any model."""
import unittest
import torch
from .alignment_root_cause import chunked_fp32_attention


class AttentionReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def check(self, query_length, key_length, mask_kind=None):
        q=torch.randn(1,4,query_length,8)
        k=torch.randn(1,2,key_length,8);v=torch.randn_like(k)
        allowed=torch.arange(key_length)[None,:]<=torch.arange(query_length)[:,None]+key_length-query_length
        mask=None
        if mask_kind=='bool':mask=allowed[None,None]
        if mask_kind=='additive':mask=torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed,float('-inf'))[None,None]
        actual,_=chunked_fp32_attention(None,q,k,v,mask,scaling=8**-.5)
        expected=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,dim=1),
            v.repeat_interleave(2,dim=1),attn_mask=allowed[None,None],scale=8**-.5)
        torch.testing.assert_close(actual,expected.transpose(1,2),rtol=2e-5,atol=2e-6)

    def test_causal_across_chunks(self):self.check(153,153)
    def test_cached_query_offset(self):self.check(1,153)
    def test_boolean_mask(self):self.check(153,153,'bool')
    def test_additive_mask(self):self.check(153,153,'additive')


if __name__=='__main__':unittest.main()
