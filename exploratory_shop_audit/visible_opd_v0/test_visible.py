import unittest
import torch
from transformers import AutoTokenizer
from exploratory_shop_audit.runtime import MODEL, make_input
from .packets import packet, render
from .parser import categories
from .projection import candidate_loss, candidate_target, VisibleProjection
from .score import centered, measure, support

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)

    def test_prompt_and_evidence(self):
        native=make_input(self.tok,'test',[('past','reply')],'now')
        self.assertEqual(render(self.tok,native,{})[0],native['token_ids'])
        page={'page_type':'search_results','products':[{'asin':'shown'}]}
        cat={'shown':{'customization_options':{'颜色':[]}},'other':{'customization_options':{'secret':[]}}}
        fields,why=packet('evidence',page,cat)
        self.assertIsNone(why)
        self.assertEqual(list(fields.values()),[['颜色']])
        self.assertEqual(packet('evidence',{'page_type':'search_home'},cat)[1],'PACKET_UNAVAILABLE')
        self.assertEqual(render(self.tok,native,{'超长':'word '*400})[1],'PACKET_OVERFLOW')
        self.assertEqual(set(packet('placeholder_evidence',page,cat)[0]),set(fields))

    def test_spans(self):
        public=dict(page_type='product_detail',available_options={'颜色':['蓝色']})
        for text,label in [('想法\nAction: click[1234567890]','click'),('Action: click[蓝色]','option'),('Action: search[冰箱]','search'),('click[蓝色]','option'),('search[中文]','search'),('invalid','unknown')]:
            ids=self.tok.encode(text)+[self.tok.eos_token_id]
            cats=categories(self.tok,ids,public)
            self.assertEqual(len(ids),len(cats))
            self.assertIn(label,cats)
            self.assertEqual(cats[-1],'format')

    def test_support_and_detach(self):
        torch.manual_seed(9)
        s=torch.randn(5,16).log_softmax(-1)
        ids,mask=support(s,torch.cat([s.argmax(-1)[:2],s.argmin(-1)[2:]]))
        self.assertEqual(mask.sum(1).tolist(),[8,8,9,9,9])
        raw,qs,_=measure(s,ids,mask)
        p,qp,_=measure(torch.randn_like(s).log_softmax(-1),ids,mask)
        d=centered(p-raw,qs).requires_grad_()
        target=candidate_target(raw,d,1.,mask)
        self.assertTrue(torch.allclose(target,qp,atol=1e-6))
        actor=raw.clone().requires_grad_()
        loss=candidate_loss(actor,target,mask,torch.tensor([0,0,1,2,2]),torch.tensor([0,0,0,1,1]))
        loss.backward()
        self.assertIsNone(d.grad)
        self.assertGreater(actor.grad.abs().sum(),0)
        self.assertEqual(actor.grad[~mask].abs().sum(),0)
        proj=VisibleProjection(12)(torch.randn(5,12),torch.randn(5,9,12),qs)
        self.assertLess((proj*qs).sum(-1).abs().max(),1e-6)
