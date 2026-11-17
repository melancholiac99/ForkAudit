import unittest
import numpy as np
from .alignment_metrics import summarize, ess_fraction


class AlignmentTests(unittest.TestCase):
    def test_identical(self):
        p=np.log(np.array([[.9,.2],[.5,.3]]));m=np.ones_like(p,dtype=bool)
        s=summarize(p,p,m,np.ones_like(p),['a','b'])
        self.assertAlmostEqual(s['token_ess_fraction'],1.)
        self.assertEqual(s['behavior_to_actor_k3_sample_estimate'],0.)
        self.assertAlmostEqual(s['surrogate_coefficient_cosine'],1.)

    def test_mask_and_trajectory_aggregation(self):
        actor=np.log([[.4,.1],[.1,.1]]);behavior=np.log([[.2,.9],[.2,.9]])
        s=summarize(actor,behavior,[[1,0],[1,0]],trajectory_ids=['same','same'])
        self.assertAlmostEqual(s['trajectory_ess_fraction'],1.)
        self.assertAlmostEqual(s['token_ess_fraction'],2.5**2/(2*(4+.25)))
        self.assertEqual(s['tokens'],2)

    def test_clipping_directions(self):
        a=np.log([[.4,.1,.4,.1]]);b=np.log([[.2,.2,.2,.2]])
        s=summarize(a,b,[[1,1,1,1]],[[1,-1,-1,1]])
        self.assertEqual(s['counterfactual_initial_clipping_fraction'],.5)
        self.assertAlmostEqual(s['behavior_to_actor_k3_sample_estimate'],.25)

    def test_empty_and_nonfinite(self):
        with self.assertRaises(ValueError):summarize([[0]],[[0]],[[0]])
        with self.assertRaises(ValueError):summarize([[float('nan')]],[[0]],[[1]])
        self.assertAlmostEqual(ess_fraction([1000,1000]),1.)

    def test_capture_preserves_inputs_and_excludes_padding_duplicates(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from .alignment_metrics import capture
        class Tensor:
            def __init__(self,value):self.value=np.asarray(value)
            def detach(self):return self
            def float(self):return self
            def cpu(self):return self
            def numpy(self):return self.value
        original=np.log([[.4],[.1],[.4]])
        data=SimpleNamespace(batch={
            'old_log_probs':Tensor(original), 'rollout_log_probs':Tensor(np.log([[.2],[.2],[.2]])),
            'response_mask':Tensor([[1],[1],[1]]), 'advantages':Tensor([[1],[-1],[1]]),
            'responses':Tensor([[1],[2],[1]])}, non_tensor_batch={'shop_decision':[
                json.dumps({'trajectory_id':k,'step':0}) for k in ['a','b','a']]})
        with tempfile.TemporaryDirectory() as temp:
            capture(data,Path(temp),3)
            row=json.loads((Path(temp)/'alignment_live/metrics.jsonl').read_text())
            self.assertEqual((row['saved_rows'],row['unique_rows'],row['tokens']),(3,2,2))
            np.testing.assert_array_equal(data.batch['old_log_probs'].value,original)
            self.assertEqual(len(list((Path(temp)/'alignment_live').glob('*.npz'))),1)


if __name__=='__main__':unittest.main()
