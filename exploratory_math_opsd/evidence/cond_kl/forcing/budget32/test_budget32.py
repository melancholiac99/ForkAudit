import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from report import differences
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from budget import install_forcing
from test_forcing import Generator

class Budget32Test(unittest.TestCase):
    def test_maximum_and_marker(self):
        req = NS(output_token_ids=[], sampling_params=NS(seed=1002,max_tokens=200,
                 extra_args={'target_lengths':[1,185], 'forcing_row':7}))
        runner = NS(use_async_scheduling=False, _to_list=lambda t:[[t]],
                    discard_request_indices=NS(np=[]),num_discarded_requests=0,
                    input_batch=NS(req_ids=['row'], generators={0:Generator()}), requests={'row':req})
        marker = [5,6,7,8]
        install_forcing(NS(model_runner=runner),[10],marker,32)
        for _ in range(32):
            for expected in marker:
                token = runner._to_list(10)[0][0]
                self.assertEqual(token,expected)
                req.output_token_ids.append(token)
        self.assertEqual(runner._to_list(10),[[10]])
        self.assertEqual(len(runner.forcing_events[7,1]),32)

    def test_subset_keeps_paired_sample_indices(self):
        forced=np.array([[1,0],[0,1],[0,0]])
        base=np.array([[0,0],[1,0],[1,1]])
        match=np.array([[1,1],[0,1],[0,0]],bool)
        result=differences(forced,base,match)
        self.assertAlmostEqual(result['all_questions']['mean'],-100/6)
        self.assertEqual(result['both_answers_matched']['mean'],50)
        self.assertEqual(result['matched_answers_equal_question_weight']['mean'],75)
if __name__=='__main__':unittest.main()
