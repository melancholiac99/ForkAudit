import unittest
from types import SimpleNamespace
import numpy as np
import exp3_token_diagnostic as T
import token_diagnostic_backend as B
import token_diagnostic_analysis as A

class CharacterTokenizer:
    def __call__(self,text,**kw):return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)

class DataTests(unittest.TestCase):
    def test_installed_qwen_prefix_and_space_merged_action(self):
        from pathlib import Path
        model=Path('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
        if not model.exists():self.skipTest('local Qwen tokenizer unavailable')
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
        ids=B.prompt_ids(tok,'A real decision prompt')
        self.assertGreater(len(ids),10);self.assertTrue(all(isinstance(i,int) for i in ids))
        for action in ['look','inventory']:
            text='<think>check it</think>\n[action> '+action+'</action>'
            r=T.tokenize_response(tok,text,action)
            self.assertIn('action',r['kinds']);self.assertEqual(tok.decode(r['ids']),text)

    def test_response_partition_and_no_synthetic_eos(self):
        text='<think>find it</think>\n[action] look [/action]'
        r=T.tokenize_response(CharacterTokenizer(),text,'look')
        self.assertEqual(len(r['ids']),len(text))
        self.assertEqual(''.join(c for c,k in zip(text,r['kinds']) if k=='think'),'find it')
        self.assertEqual(''.join(c for c,k in zip(text,r['kinds']) if k=='action'),'look')
        self.assertEqual(sum(r['kinds'].count(k) for k in T.KINDS),len(text))

    def test_evidence_redaction_and_student_context(self):
        ep={'steps':[dict(response='<think>SECRET</think><action>look</action>',executed_action='look',observation_after='unchanged'),
                     dict(response='<think>other</think><action>open</action>',executed_action='open')],'final_reward':0}
        p=T.evidence_prompts('original decision prompt',ep,1,[{'action':'look','response':'unchanged'}])
        self.assertEqual(p['M'],'original decision prompt')
        self.assertNotIn('SECRET',p['L']);self.assertNotIn('SECRET',p['LN'])
        self.assertIn('<think>other',p['L']);self.assertNotIn('<think>other',p['LN'])
        self.assertIn('SECRET',ep['steps'][0]['response'])

    def test_malformed_historical_action_is_preserved(self):
        text='<think>old sample</think>\n&action>go to drawer 2</action>'
        r=T.tokenize_response(CharacterTokenizer(),text,'action>go to drawer 2</action>')
        self.assertEqual(r['text'],text)
        self.assertEqual(r['action_span_method'],'exact_recorded_action_in_malformed_reply')
        self.assertEqual(''.join(c for c,k in zip(text,r['kinds']) if k=='action'),'action>go to drawer 2</action>')

    def test_ranking_uses_replay_quality_and_handles_ties(self):
        q=[0,0,1/3,2/3,1]
        a=A.ranking_metrics(q,q);b=A.ranking_metrics([-x for x in q],q)
        self.assertAlmostEqual(a['spearman'],1);self.assertAlmostEqual(b['spearman'],-1)
        self.assertEqual(a['pairwise_concordance'],1);self.assertEqual(b['pairwise_concordance'],0)
        c=A.ranking_metrics([0]*5,q)
        self.assertIsNone(c['spearman']);self.assertEqual(c['pairwise_concordance'],.5)
        self.assertAlmostEqual(c['top_proxy_value_gain'],0)

class BackendTests(unittest.TestCase):
    def test_full_vocabulary_kl_directions_and_sampled_proxy(self):
        import torch
        head=torch.nn.Identity()
        sl=torch.tensor([[.25,.75],[.4,.6]]).log();tl=torch.tensor([[.5,.5],[.8,.2]]).log()
        ids=[0,1]
        with torch.inference_mode():
            ref=B.sampled_logps(head,sl,ids);r=B.signals(head,tl,head,sl,ids,ref)
        sp=sl.exp().numpy();tp=tl.exp().numpy()
        np.testing.assert_allclose(r['reverse_kl'],(sp*(np.log(sp)-np.log(tp))).sum(1),atol=1e-6)
        np.testing.assert_allclose(r['forward_kl'],(tp*(np.log(tp)-np.log(sp))).sum(1),atol=1e-6)
        np.testing.assert_allclose(r['delta'],[np.log(2),np.log(1/3)],atol=1e-6)
        summary=T.summarize(r,['think','action'])
        self.assertAlmostEqual(summary['all']['delta']['sum'],sum(r['delta']))
        self.assertEqual(summary['other']['n'],0)
        self.assertIsNone(summary['other']['delta']['mean'])

    def test_cached_prefix_predicts_correct_tokens_without_mutation(self):
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from exp1c_backend import ChunkedScorer
        torch.manual_seed(72)
        cfg=Qwen2Config(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)
        model=Qwen2ForCausalLM._from_config(cfg,attn_implementation='sdpa').float().eval()
        scorer=ChunkedScorer(SimpleNamespace(t=torch,model=model),chunk=7)
        pre=list(range(10,31));replies=[[3,8,6,9,2],[8,4,2]]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            cache,last=B.prefix(scorer,pre)
            original_length=cache.get_seq_length()
            for ids in replies:
                a=B.response_hidden(scorer,cache,last,ids);b=B.direct_hidden(scorer,pre,ids)
                full=model.model(input_ids=torch.tensor([pre+ids[:-1]]),use_cache=False).last_hidden_state[0,len(pre)-1:]
                self.assertLess(float((a-full).abs().max()),1e-5)
                self.assertLess(B.distribution_tv(model.lm_head,a,b),1e-5)
                self.assertEqual(len(a),len(ids));self.assertEqual(cache.get_seq_length(),original_length)
                lp=B.sampled_logps(model.lm_head,a,ids)
                signals=B.signals(model.lm_head,a,model.lm_head,a.cpu(),ids,lp)
                self.assertLess(max(abs(x) for x in signals['delta']),1e-6)
                self.assertLess(max(abs(x) for x in signals['reverse_kl']),1e-6)
            # The cache must still predict the first arm identically after scoring another arm.
            np.testing.assert_allclose(B.sampled_logps(model.lm_head,B.response_hidden(scorer,cache,last,replies[0]),replies[0]),
                                       B.sampled_logps(model.lm_head,B.direct_hidden(scorer,pre,replies[0]),replies[0]),atol=1e-5)

if __name__=='__main__':unittest.main()
