import os
import unittest
from unittest.mock import patch

from .e2512_native import native_rank
from .e25_layout import Layout
from .e2514_search import selection


class MigrationTests(unittest.TestCase):
    def test_physical_rank_in_group(self):
        for start in (0,4):
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':','.join(str(start+r) for r in range(4))}):
                for r in range(4):
                    native_rank(r)
                    self.assertEqual(os.environ['SHOP_GPU_IDS'],str(start+r))
                    self.assertEqual(os.environ['SHOP_GPU_ID'],str(start+r))

    def test_frozen_projectors(self):
        l=Layout('formal_2512')
        self.assertEqual(l.projector_file(30,'P-frozen',2),l.projector_file(1,'P',2))
        self.assertNotEqual(l.projector_file(30,'P',2),l.projector_file(30,'P-frozen',2))

    def test_split_selection_and_strict_subset(self):
        split='T_opd_eval'
        counts={20:[2,2,1,0],30:[0,1,1,1]}
        def records(step):
            return {(split,g,r):{'reason':'gold_purchase' if r<n else 'wrong_purchase'}
                    for g,n in enumerate(counts[step]) for r in range(4)}
        with patch('exploratory_shop_audit.visible_opd_v0.e2514_search.original',records),patch('exploratory_shop_audit.visible_opd_v0.e2514_search.save'):
            selected=selection()
        self.assertEqual([(r['goal'],r['strict_subset']) for r in selected],[(0,True),(1,False)])
        self.assertEqual(selected[1]['good_reps'],[0,1])
        self.assertEqual(selected[1]['failed_reps'],[1,2,3])
