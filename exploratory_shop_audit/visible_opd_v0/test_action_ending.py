import unittest

import numpy as np

from exploratory_shop_audit.grpo_forced_audit.common import read
from .action_ending import candidate_classes, locate
from .action_ending_train import DEST, metrics


class EndingBoundaryTest(unittest.TestCase):
    def test_first_command_nested_and_fused(self):
        pos,following,why = locate([b'Action: ',b'click[foo[a',b']',b')]\n',b'click',b'[Buy Now]',b'EOS'])
        self.assertEqual((following,why),('newline','located'))
        self.assertEqual([(p.index,p.role,p.prefix) for p in pos],
                         [(3,'bracket',b')]'),(4,'after_fused_newline',b'')])
        tokens = {1:b')]',2:b')]\n',3:b']',4:b'other',5:b'click',6:b'<EOS>'}
        self.assertEqual(candidate_classes(pos[0],[2,1,3,4],tokens,{6}),([1],[0]))
        self.assertEqual(candidate_classes(pos[1],[4,5,6],tokens,{6}),([2],[1]))

    def test_following_token_and_escaped_delimiter(self):
        pos,following,_ = locate([b'click[42',b']',b'\\nAction: click[17]'])
        self.assertEqual((following,pos[1].role),('newline','after_bare'))
        tokens = {1:b']',2:b'\n',3:b'\\n',4:b'<E1>',5:b'<E2>',6:b'click'}
        self.assertEqual(candidate_classes(pos[1],[1,2,3,4,5,6],tokens,{4,5}),([3,4],[1,2]))
        self.assertEqual(locate([b'click[42] click[17]'])[1],'same_line')
        self.assertEqual(len(locate([b'finish[done]'])[0]),1)
        self.assertEqual(locate([b'click[unclosed'])[2],'unclosed_first_command')
        self.assertEqual(locate([b'Buy Now'])[2],'no_command')

    def test_zero_support_probability_and_log_denominators(self):
        obs=[(1,True,None,'bracket',False,True,0.,0.,.6,.3),
             (2,True,None,'bracket',True,True,.1,.2,.6,.7)]
        r = metrics(obs)
        np.testing.assert_allclose(r['mean'],[.05,.1,.5,.6,.5])
        self.assertEqual(r['log_stop_ratio']['n'],1)
        self.assertAlmostEqual(r['log_stop_ratio']['mean'][0],np.log(.5))


class PublishedEndingTest(unittest.TestCase):
    def test_all_windows_accounted_against_registered_monitors(self):
        report = read(DEST/'section26_action_ending.json')
        base = read(DEST.parent.parent/'endpoint_report.json')
        for arm,steps in base['training'].items():
            windows = [w for w in report['windows'] if w['arm']==arm]
            self.assertEqual([w['start'] for w in windows],list(range(1,len(steps)+1,5)))
            for w in windows:
                c = w['coverage']
                monitors=steps[w['start']-1:w['end']]
                self.assertEqual(c['replies'],sum(r['actions'] for r in monitors))
                self.assertEqual(c['legal_replies'],sum(r['legal'] for r in monitors))
                self.assertEqual(c['next_command_replies'],c.get('next_command_newline',0)+c.get('next_command_same_line',0))
                allrows=w['cohorts']['all']
                self.assertEqual(allrows['all']['positions']+c['structural_na_positions'],c['positions_located'])
                self.assertEqual(allrows['all']['positions'],allrows['bracket']['positions']+allrows['following']['positions'])
            agg=report['overall'][arm]['cohorts']['all']['all']['positions']
            self.assertEqual(agg,sum(w['cohorts']['all']['all']['positions'] for w in windows))

    def test_tcal_candidate_intersection_and_original_format(self):
        d=read(DEST/'tcal_action_ending.json')
        self.assertEqual((d['all_positions'],d['tasks']),(34825,512))
        self.assertEqual(sum(d['reply_coverage'].values()),d['replies'])
        n=d['overall']['n']
        self.assertEqual(n,sum(x['n'] for x in d['by_token']))
        self.assertEqual(n,d['positions'][0]['n']+d['positions'][1]['n'])
        self.assertEqual(n+d['old_only_candidates'],d['native_B']['n'])
        self.assertEqual(n+d['new_only_candidates'],d['native_block1']['n'])
        self.assertLessEqual(d['identical_support_only']['n'],n)
        for x in d['positions']:
            if x['n']:
                self.assertEqual(len(x['mean']),len(d['columns']))
                self.assertEqual(len(x['mean']),len(x['positive_share']))
