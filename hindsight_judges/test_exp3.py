"""Fast CPU checks for the three Hindsight follow-up components."""
import json
import unittest
import torch
import numpy as np

import exp3_action_diagnostic as diagnostic
import exp3_hln_dpo as hln
import rationale_blind_credit as credit


class TestActionDiagnostic(unittest.TestCase):
    def test_coherent_think_toggle(self):
        s='<think>Open the fridge.</think>\n<action>open fridge</action>'
        keep=diagnostic.student_prefix(s,'keep')
        drop=diagnostic.student_prefix(s,'drop')
        self.assertIn('Open the fridge.',keep)
        self.assertEqual(drop,'<action>')
        self.assertIn('fridge',s)

    def test_trajectory_redaction_keeps_all_environment_facts(self):
        ep={'steps':[{'turn':1,'response':'<think>A</think><action>a</action>',
                      'executed_action':'a','observation_after':'obs1','reward_after':0},
                     {'turn':2,'response':'<think>B</think><action>b</action>',
                      'executed_action':'b','observation_after':'obs2','reward_after':1}],
            'final_reward':1}
        L=diagnostic.redact_episode(ep,1,False)
        LN=diagnostic.redact_episode(ep,1,True)
        self.assertEqual(L['steps'][0],LN['steps'][0])
        self.assertEqual(L['final_reward'],LN['final_reward'])
        for new in (L,LN):
            self.assertEqual(new['steps'][1]['executed_action'],'b')
            self.assertEqual([x['observation_after'] for x in new['steps']],['obs1','obs2'])
        self.assertIn('B',L['steps'][1]['response'])
        self.assertNotIn('B',LN['steps'][1]['response'])
        self.assertEqual(ep['steps'][0]['executed_action'],'a')

    def test_action_contexts_have_identical_target_set(self):
        s={'text':'base\n<decision_time_prompt>\nTask here\n</decision_time_prompt>\nCandidates',
           'actions':['open fridge','look','take','go','examine'],
           'point_id':'s000_t01','traj_id':'s000','turn':1}
        ep={'steps':[dict(turn=1,response='<think>One</think><action>open fridge</action>',
                         executed_action='open fridge',observation_after='empty'),
                     dict(turn=2,response='<think>Two</think><action>look</action>',
                         executed_action='look',observation_after='empty')]}
        templ='Episode:\n';ftempl='Five peek results:\n'
        L={'text':s['text']+'\n'+templ+json.dumps(ep)}
        F={'text':s['text']+'\n'+ftempl+json.dumps([dict(action=a,response='empty') for a in s['actions']])}
        cases=diagnostic.make_cases(s,L,F,templ,ftempl)
        self.assertEqual(len(cases),8)
        self.assertEqual(len({x['id'] for x in cases}),8)
        self.assertEqual(len({tuple(x['actions']) for x in cases}),1)
        self.assertTrue(all('Open' not in x['prompt'] for x in cases))
        self.assertNotIn('executed_action": "open fridge"',next(x for x in cases if x['condition']=='L')['prompt'])


class TestHLN(unittest.TestCase):
    def test_ln_only_removes_think(self):
        template='Log:\n'
        ep={'steps':[{'response':'<think>A</think><action>open</action>',
                      'executed_action':'open','observation_after':'nothing'}], 'final_reward':0}
        l={'text':'base\n'+template+json.dumps(ep),'point_id':'p1','request_id':'p1__L','condition':'L'}
        ln,n=hln.make_ln(l,template)
        self.assertEqual(n,1)
        self.assertNotIn('<think>',ln['text'])
        self.assertIn('open',ln['text'])
        self.assertEqual(ln['point_id'],'p1')

    def test_equal_budget_and_common_points(self):
        points=[dict(point_id=f'p{i}',prompt='P',arm_responses=['f','a','b','c','d']) for i in range(8)]
        scores={(p['point_id'],c):dict(candidate_prob=[.2,.5,.1,.1,.1] if (i+(c=='LN'))%3 else [.6,.1,.1,.1,.1])
                for i,p in enumerate(points) for c in ('L','LN')}
        data,info=hln.make_pairs(points,scores,100)
        self.assertEqual(len(data['H_re']),len(data['H_LN']))
        self.assertEqual([x['point_id'] for x in data['H_re_common']],
                         [x['point_id'] for x in data['H_LN_common']])
        self.assertEqual(info['matched_n'],len(data['H_re']))


class TestBlindCredit(unittest.TestCase):
    def test_weight_preserves_reward_sign(self):
        for advantage in (-2,-1,0,1,2):
            for delta in (-5,-1,0,1,5):
                w=credit.blind_weight(advantage,delta,0)
                self.assertTrue(0.5<=w<=2)
                self.assertEqual(np.sign(advantage*w),np.sign(advantage))

    def test_all_response_tokens_get_grad(self):
        new=torch.zeros(7,requires_grad=True)
        loss,_=credit.weighted_ppo_policy_loss(new,torch.zeros(7),1,2,1,torch.ones(7))
        loss.backward()
        self.assertTrue(torch.all(new.grad.abs()>0))


if __name__=='__main__':unittest.main()
