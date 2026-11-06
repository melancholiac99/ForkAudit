import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import exp5_token_slot as S
import token_slot_backend as K
import token_slot_analysis as A


class RuleTests(unittest.TestCase):
    def test_stable_commitment_is_not_first_crossing(self):
        curve=[dict(t=i,prob=[p]) for i,p in enumerate([.95,.8,.91,.94])]
        self.assertEqual(S.stable_k(curve),2)
        self.assertIsNone(S.stable_k([dict(t=i,prob=[p]) for i,p in enumerate([.95,.8])]))
        self.assertEqual(S.phase(2,2),'at_or_after_k');self.assertEqual(S.phase(1,2),'pre_k')
        self.assertEqual(S.phase(1,None),'no_k')

    def test_cluster_point_weighting_and_vectorized_slope(self):
        c=A.ci([0,0,100],['a','a','b']);self.assertAlmostEqual(c['estimate'],100/3)
        self.assertEqual(c['ci95'],[0,100])
        z=A.slope_ci([0,1,2,3],[1,3,5,7],['a','a','b','b']);self.assertAlmostEqual(z['estimate'],2)
        np.testing.assert_allclose(z['ci95'],[2,2])

    def test_targets_and_no_zero_probability_outside_topk(self):
        import torch
        torch.manual_seed(62)
        q={k:torch.log_softmax(torch.randn(512),0) for k in ('AF','AM','EF','EM')}
        targets={name:lp for name,lp,_ in K.targets(q)}
        self.assertEqual(len(targets),17)
        for lp in targets.values():self.assertTrue(torch.isfinite(lp).all());self.assertAlmostEqual(float(lp.exp().sum()),1,places=5)
        expected=torch.log_softmax(2*q['AF']-q['AM'],0)
        torch.testing.assert_close(targets[K.BASELINE],expected)
        d=((q['EF']-q['EM'])-(q['AF']-q['AM'])).clamp(-5,5)
        torch.testing.assert_close(targets[K.MAIN],torch.log_softmax(q['AF']+d,0))
        union=torch.unique(torch.cat([torch.topk(q[k],128).indices for k in ('AF','EF')]))
        mask=torch.ones(512,dtype=torch.bool);mask[union]=False
        lp=targets['T2_b1_clip5_top128'];self.assertTrue((lp[mask].exp()>0).all())
        differences=lp[mask]-q['AF'][mask]
        self.assertLess(float(differences.max()-differences.min()),2e-6)
        # All contexts equal -> every target reduces to AF, no invented evidence signal.
        for _,lp,_ in K.targets({k:q['AF'] for k in q}):torch.testing.assert_close(lp,q['AF'])

    def test_kl_directions_and_sampled_student_proxy(self):
        import torch
        q={k:torch.tensor([.2,.3,.5]).log() for k in ('AF','AM','EF','EM')}
        q['EF']=torch.tensor([.6,.3,.1]).log()
        r=K.summaries(q,0,-2,{'restart':[0]})
        t3=r['targets']['T3'];self.assertAlmostEqual(t3['proxy'],np.log(.6)+2,places=5)
        self.assertAlmostEqual(t3['kl_target_AF'],float(np.sum(np.array([.6,.3,.1])*np.log(np.array([.6,.3,.1])/np.array([.2,.3,.5])))),places=5)
        self.assertNotAlmostEqual(t3['kl_target_AF'],t3['kl_AF_target'],places=3)


class TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        p=Path('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
        if not p.exists():raise unittest.SkipTest('Qwen tokenizer missing')
        cls.tok=AutoTokenizer.from_pretrained(p,local_files_only=True,use_fast=True)

    def group(self):
        actions=['look','inventory','go to desk 1','open fridge 1','take apple 1']
        text='<think>First inspect the room.\nThen look carefully.\n\nThis is my final plan.</think>\n[action] look [/action]'
        r=S.OLD.tokenize_response(self.tok,text,'look')
        return dict(id='fixture',cluster='test',q=[0,1,.5,0,.2],response=r,actions=actions,prompts={'M':'Task original','F':'Task with symmetric evidence'},
                    boundary_rules=S.response_boundaries(self.tok,r,actions))

    def test_native_positions_are_real_and_quotes_see_only_prefix(self):
        g=self.group();self.assertGreater(len(g['boundary_rules']['boundaries']),2)
        positions=S.native_positions(g,None)
        self.assertTrue(all(p['phase']=='no_k' for p in positions))
        for p in positions:
            t=p['t'];h=g['response']['ids'][:t]
            for slot in ('A','E'):
                ids,stem,audit=K.contexts(self.tok,g['prompts']['M'],slot,h)
                self.assertEqual(audit['causal_prefix_ids'],h)
                self.assertEqual(ids[:len(stem)],stem)
                if slot=='E':
                    self.assertEqual(audit['messages'][0]['content'],g['prompts']['M']+K.OPEN+self.tok.decode(h)+K.CLOSE)
                    self.assertEqual(audit['active_prefix_ids'],[])
                else:self.assertEqual(ids[-len(h):],h)
        h=self.tok.encode('Known prefix',add_special_tokens=False)
        ids,_,audit=K.contexts(self.tok,'Task','E',h)
        self.assertNotIn('SECRET FUTURE',str(audit))

    def test_probe_preserves_all_five_full_actions_and_original_spacing(self):
        g=self.group();h,candidates=S.probe_job(self.tok,g,g['boundary_rules']['boundaries'][0])
        self.assertEqual(len(candidates),5)
        for action,c in zip(g['actions'],candidates):
            text=self.tok.decode(h+c,clean_up_tokenization_spaces=False)
            self.assertIn('</think>',text)
            self.assertTrue(text.endswith('[action] '+action+' [/action]'+self.tok.eos_token))
        self.assertIn('action_first',[reason for p in S.native_positions(g,None) for reason in p['reasons']])

    def test_engine_caching_matches_direct_and_never_mutates_base(self):
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from exp1c_backend import ChunkedScorer
        # Real tokenizer IDs require full vocab, but a tiny hidden-state model is CPU-safe.
        torch.manual_seed(11)
        cfg=Qwen2Config(vocab_size=len(self.tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512)
        model=Qwen2ForCausalLM._from_config(cfg,attn_implementation='sdpa').float().eval()
        scorer=ChunkedScorer(SimpleNamespace(t=torch,model=model),chunk=7)
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            engine=K.Engine(scorer,self.tok,{'M':'Task','F':'Task with evidence'})
            for htext in ['<think>First','<think>First consider the room.']:
                h=self.tok.encode(htext,add_special_tokens=False)
                for slot in ('A','E'):
                    lp=engine.lp(slot,'M',h);ids,stem,_=engine.input(slot,'M',h)
                    z=model.model(input_ids=torch.tensor([ids]),use_cache=False)
                    direct=torch.log_softmax(model.lm_head(z.last_hidden_state[0,-1]),0)
                    torch.testing.assert_close(lp,direct,atol=1e-5,rtol=0)
                    self.assertEqual(engine.bases[slot,'M'][0].get_seq_length(),len(stem))
                    torch.testing.assert_close(engine.lp(slot,'M',h),lp,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
