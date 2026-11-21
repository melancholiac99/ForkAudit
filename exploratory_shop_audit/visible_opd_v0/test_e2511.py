import unittest

from .e2511_schedule import CAL, MATCH, classify, interpolate, scheduled


class StageEMatchingAmendment(unittest.TestCase):
    def test_unrounded_interpolation(self):
        lr,power=interpolate(1.e-6,.0019670045461,2.496624166599355e-6,.04521593856939092,.01226059943154)
        self.assertAlmostEqual(lr*1e6,1.7058350962,places=8)
        self.assertAlmostEqual(power,3.42638813,places=7)

    def test_match_labels_include_both_boundaries(self):
        for r,label in ((.79,'MOVEMENT_MISMATCH'),(.8,'MATCHED'),(1.25,'MATCHED'),(1.26,'MOVEMENT_MISMATCH')):
            self.assertEqual(classify(r*r,1.)[1],label)

    def test_all_arm_indices_and_separate_calibration(self):
        rows=[j for n in range(1,111) for j in scheduled(n,{})]
        for a in ('R','P',MATCH):
            self.assertEqual([k for name,k,arm in rows if name=='formal' and arm==a],list(range(1,101)))
        self.assertEqual([k for name,k,arm in rows if name==CAL],list(range(1,11)))
        self.assertEqual(scheduled(11,{}),[('formal',11,'R'),('formal',11,'P'),('formal',1,MATCH)])
        self.assertNotIn(('formal',21,'P'),scheduled(21,{'P':20}))

    def test_balanced_shards_preserve_all_jobs_and_goals(self):
        from .e2511_targets import assignment
        work=[(name,i,100+i) for name in ('R','P','matched') for i in range(32)]
        bins=assignment(work)
        self.assertEqual(sorted(x for group in bins for x in group),sorted((j,g) for j,g,_ in work))
        self.assertEqual(len(bins),4)
        sizes=[sum(100+g for _,g in group) for group in bins]
        self.assertLess(max(sizes)-min(sizes),250)
