import unittest
import numpy as np
import analyze_interim as A
import monitor as M
import token_slot_analysis as Legacy

class FutilityTests(unittest.TestCase):
    def test_matches_original_cluster_bootstrap(self):
        values=[None,-20,5,10,2,-3,18];clusters=['z','a','a','b','b','c','c']
        self.assertEqual(A.interval(values,clusters),Legacy.ci(values,clusters))

    def test_strict_upper_bound_and_never_early_success(self):
        s=dict(n=190,n_trajectories=30,estimate=0,ci95=[-4,2.999])
        self.assertEqual(A.decision(s),'stop_for_futility')
        for bound in (3.,10.,100.):
            self.assertEqual(A.decision(dict(s,ci95=[-4,bound])),'continue_to_full')
        self.assertEqual(A.decision(dict(s,estimate=99,ci95=[90,100])),'continue_to_full')

    def test_insufficient_or_undefined_does_not_stop(self):
        s=dict(n=190,n_trajectories=30,estimate=-50,ci95=[-60,-40])
        for change in [dict(n=29),dict(n_trajectories=9),dict(ci95=None),dict(ci95=[-60,float('nan')])]:
            self.assertEqual(A.decision(dict(s,**change)),'continue_to_full')

    def test_point_average_and_no_k_not_zero(self):
        arms=[]
        for i in range(5):
            vals={A.T2:float(i),A.T1:0.}
            arms.append(dict(sequence_logp=vals,tokens=[dict(target_logp=vals)]))
        r=dict(q=[0,0,0,0,1],k_char=10,probes=[dict(char=1,phase='pre_k',arms=arms),dict(char=11,phase='at_or_after_k',arms=arms)])
        expected=100*(np.exp(4)/np.exp(np.arange(5)).sum()-.2)
        self.assertAlmostEqual(A.point_contrast(r),expected)
        r.update(k_char=None,probes=[dict(char=1,phase='no_k',arms=arms)])
        self.assertIsNone(A.point_contrast(r))

    def test_reject_corrupt_action_sum(self):
        vals={A.T2:0.,A.T1:0.};arms=[dict(sequence_logp=vals.copy(),tokens=[dict(target_logp=vals.copy())]) for _ in range(5)]
        arms[0]['sequence_logp'][A.T2]=1.
        with self.assertRaises(AssertionError):A.point_contrast(dict(q=[0,1,0,0,0],k_char=2,probes=[dict(char=1,phase='pre_k',arms=arms)]))

    def test_stop_scope_excludes_pilot_other_outputs_and_other_models(self):
        words=['python',str(M.H/'exp5_token_slot.py'),'score','--out','/tmp/h800','--name','32b','--limit','0']
        self.assertTrue(M.matches_score(words,'/tmp/h800'))
        self.assertFalse(M.matches_score(words,'/tmp/other'))
        for old,new in [('0','32'),('32b','7b'),('score','verify')]:
            changed=[new if w==old else w for w in words]
            self.assertFalse(M.matches_score(changed,'/tmp/h800'))

if __name__=='__main__':unittest.main()
