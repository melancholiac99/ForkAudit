import itertools
import unittest
import numpy as np
from metrics import cross_action_cov, completion_effect, outcome_components
from protocol import ARMS, seed_for


class Tests(unittest.TestCase):
    def test_exact_expectation_under_paired_outcomes(self):
        # Native/completed are correlated within a repeat. Enumerate two iid
        # repeats exactly to verify that the contrast is not regression-to-mean bias.
        states = [(np.array([0.,0.]), np.array([0.,1.]), .2),
                  (np.array([1.,0.]), np.array([1.,1.]), .5),
                  (np.array([1.,1.]), np.array([1.,1.]), .3)]
        q0=sum(p*x for x,y,p in states); q1=sum(p*y for x,y,p in states)
        c0=c1=cov=0.
        for (x0,y0,p0),(x1,y1,p1) in itertools.product(states,repeat=2):
            z=completion_effect(np.array([x0,x1]),np.array([y0,y1]))
            c0+=p0*p1*z['native_C']; c1+=p0*p1*z['completed_C']
            cov+=p0*p1*z['baseline_lift_covariance']
        self.assertAlmostEqual(c0,np.var(q0)); self.assertAlmostEqual(c1,np.var(q1))
        self.assertAlmostEqual(cov,np.mean((q0-q0.mean())*((q1-q0)-(q1-q0).mean())))

    def test_equal_shift_changes_success_not_action_contrast(self):
        q=np.tile([.1,.2,.4,.5,.6],(4,1))
        z=completion_effect(q,q+.2)
        self.assertAlmostEqual(z['mean_lift'],.2)
        self.assertAlmostEqual(z['C_change'],0.)

    def test_shared_completion_removes_contrast_in_oracle_example(self):
        x=np.tile([.1,.1,.1,.1,.8],(4,1)); y=np.full_like(x,.8)
        z=completion_effect(x,y)
        self.assertGreater(z['mean_lift'],0)
        self.assertGreater(z['C_reduction'],0)
        self.assertAlmostEqual(z['completed_C'],0)
        m=outcome_components(y)
        self.assertAlmostEqual(m['outcome_variance'],.16)

    def test_new_seed_namespace_and_no_collisions(self):
        seeds=[seed_for(p,a,r,s) for p in range(24) for a in ARMS
               for r in range(16) for s in range(50)]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertGreater(min(seeds),90_100_000_000_000)


if __name__=='__main__':
    unittest.main()
