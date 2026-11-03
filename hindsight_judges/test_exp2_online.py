"""CPU-only scientifically relevant tests for synchronous online Hindsight OPD."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import exp2_online as E
import exp2_online_report as R


class OnlineProtocolTests(unittest.TestCase):
    def setUp(self):
        self.games={'train':[{'game_file':f'a_{i}','seed':i} for i in range(13)],
                    'tune':[{'game_file':'t'}],
                    'valid_seen':[{'game_file':'v'}],
                    'valid_unseen':[{'game_file':'u'}]}
        self.ep=dict(seed=4,game_file='a_4',task='find a cup',initial_obs='start',won=False,steps=[
            dict(prompt='decision one',response='<think>maybe cup here</think><action>open drawer</action>',
                 response_ids=[1,2,3],action='open drawer',obs_after='empty',won_after=False),
            dict(prompt='decision two',response='<think>perhaps in cabinet</think><action>open cabinet</action>',
                 response_ids=[4,5,6],action='open cabinet',obs_after='cup',won_after=False),
            dict(prompt='decision three',response='<think>take cup</think><action>take cup</action>',
                 response_ids=[7,8,9],action='take cup',obs_after='success',won_after=True)])

    def test_task_allocation_is_fixed_and_outcome_blind(self):
        a=[g['seed'] for i in range(5) for g in E.eligible_games(self.games,i,7,3)]
        b=[g['seed'] for i in range(5) for g in E.eligible_games(self.games,i,7,3)]
        self.assertEqual(a,b)
        self.assertEqual(len(set(a[:13])),13)
        self.assertNotEqual(a,[g['seed'] for i in range(5) for g in E.eligible_games(self.games,i,8,3)])
        self.assertNotEqual(E.rollout_seed(5,1,0,2),E.rollout_seed(5,1,1,2))

    def test_hindsight_evidence_pairwise_matched_except_reasoning(self):
        rows=E.pack_decisions([self.ep],5,12,10)
        self.assertEqual(len(rows),3)
        for row in rows:
            la=json.loads(row['future']['L_guarded'])
            ln=json.loads(row['future']['LN_guarded'])
            self.assertEqual(la['final_reward'],ln['final_reward'])
            for st_a,st_b in zip(la['steps'],ln['steps']):
                self.assertEqual(st_a['action'],st_b['action'])
                self.assertEqual(st_a['obs_after'],st_b['obs_after'])
                self.assertEqual(st_a['won_after'],st_b['won_after'])
            self.assertNotIn('<think>',la['steps'][row['turn']-1]['response'])
            self.assertNotIn('<think>',row['future']['LN_guarded'])
        # KEEP other turns' think in L_guarded so this is a real intervention.
        self.assertIn('maybe cup',rows[2]['future']['L_guarded'])
        self.assertNotIn('take cup</think>',rows[2]['future']['L_guarded'])

    def test_m_native_is_exact_student_prompt(self):
        row=E.pack_decisions([self.ep],0,1,10)[0]
        self.assertEqual(E.teacher_prompt(row,'M_native'),row['prompt'])
        self.assertIn('Unavailable',E.teacher_prompt(row,'M'))
        self.assertNotIn('Unavailable',E.teacher_prompt(row,'LN_guarded'))

    def test_selection_does_not_use_terminal_outcome(self):
        a=E.pack_decisions([self.ep],3,7,2)
        ep2=json.loads(json.dumps(self.ep))
        ep2['won']=True
        b=E.pack_decisions([ep2],3,7,2)
        self.assertEqual([x['id'] for x in a],[x['id'] for x in b])

    def test_train_split_leakage_is_refused(self):
        dup=json.loads(json.dumps(self.games))
        dup['valid_unseen']=[{'game_file':'a_3'}]
        with self.assertRaises(AssertionError):
            E.eligible_games(dup,0,1,2)

    def test_incomplete_think_is_counted(self):
        kinds=E._think_types_incomplete('<think>unfinished',[(0,7),(7,17)])
        self.assertEqual(kinds,['think','think'])
        self.assertEqual(E._think_types_incomplete('X<think>A</think>Y',[(0,1),(1,8),(8,9),(9,17),(17,18)]),
                         ['other','think','think','think','other'])
        ep=json.loads(json.dumps(self.ep))
        ep['steps'][0]['response']='<think>truncated rationale'
        guarded=E.guarded_context_log(ep,1,'L_guarded')
        self.assertNotIn('truncated rationale',guarded)

    def test_exact_full_vocab_kl_and_grad_sign(self):
        import torch
        x=torch.tensor([[0.,0.]],requires_grad=True)
        p=torch.tensor([[0.2,0.8]])
        k=E.full_forward_kl(x,p)
        self.assertGreater(float(k.detach()),0.)
        self.assertAlmostEqual(float(k),float((p*(p.log()-torch.log_softmax(x,-1))).sum()),places=5)
        k.mean().backward()
        self.assertGreater(float(x.grad[0,0]),0.)
        self.assertLess(float(x.grad[0,1]),0.)
        self.assertAlmostEqual(float(x.grad.sum()),0.,places=5)
        self.assertEqual(E.split_statistics([1.,2.,3.],['think','think','other'])['think']['n'],2)
        xr=torch.tensor([[0.1,-0.1]],requires_grad=True)
        exact_log=torch.tensor([[0.2,0.8]]).log()
        rev=E.full_vocab_kl(xr,exact_log,'reverse')
        self.assertGreater(float(rev.detach()),0.)
        rev.sum().backward()
        self.assertIsNotNone(xr.grad)
        self.assertTrue(bool(torch.isfinite(xr.grad).all()))
        self.assertAlmostEqual(float(E.full_vocab_kl(x.detach(),exact_log,'forward')),float(k.detach()),places=5)

    def test_paired_bootstrap_preserves_game_and_seed(self):
        import numpy as np
        x=np.ones((3,7,2));y=np.zeros((3,7,2))
        r=R.paired_cluster_bootstrap(x,y,nboot=500,seed=2)
        self.assertEqual(r['diff_pp'],100.)
        self.assertEqual(r['ci95_pp'],[100.,100.])
        self.assertEqual(r['cluster_axes'],['training_seed','heldout_game'])


if __name__=='__main__':unittest.main()
