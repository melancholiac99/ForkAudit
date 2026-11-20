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

class AmendmentTests(unittest.TestCase):
    def test_ten_point_boundary(self):
        from .section16 import behavior_result
        for q,ci,status in [(0.1,(.001,.16),'improved'),(.1,(0,.21),'PREFLIGHT_INCONCLUSIVE'),
                            (.08,(-.01,.10),'PREFLIGHT_INCONCLUSIVE'),(.08,(.01,.099),'no_improvement'),
                            (.08,(.03,.12),'PREFLIGHT_INCONCLUSIVE')]:
            self.assertEqual(behavior_result(q,ci),status)

    def test_packets_and_budget(self):
        from .packets import FIELDS
        goal=dict(asin='123',goal_options={'色':'蓝'},price_upper=50)
        data={'123':dict(title='商品',attribute=['蓝'],customization_options={'色':[]})}
        pub={'page_type':'product_detail'}
        full=packet('full',pub,data,goal)[0]; prod=packet('product',pub,data,goal)[0]
        self.assertEqual({k:prod[k] for k in full},full)
        self.assertEqual(set(prod),set(packet('placeholder_product',pub,data)[0]))
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        native=make_input(tok,'task',[],'page')
        evidence_key=FIELDS['evidence'][0][0]
        self.assertIsNone(render(tok,native,{evidence_key:' test'*400})[1])
        self.assertEqual(render(tok,native,{evidence_key:' test'*1100})[1],'PACKET_OVERFLOW')

    def test_invalid_utf8_and_calibration_domain(self):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        from .section16 import scoreable,freeze_p2_minima,p2_coverage
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        ids=tok.encode('search[')+[tok.convert_tokens_to_ids(bytes_to_unicode()[255])]+tok.encode(']')
        self.assertIn('search',categories(tok,ids,{'page_type':'search_home'}))
        self.assertTrue(scoreable('full','product_detail'))
        self.assertFalse(scoreable('evidence','product_detail'))
        limits=freeze_p2_minima(dict(click=768,option=128,search=512),512)
        self.assertEqual(limits,dict(click=480,option=80,search=320))
        self.assertFalse(p2_coverage(limits,limits,dict(click=51,option=49,search=50))['option'])
