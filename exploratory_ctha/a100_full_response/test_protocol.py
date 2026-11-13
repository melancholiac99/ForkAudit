import ast
import math
import unittest
from pathlib import Path
from direction_policy import evaluate


class ProtocolTests(unittest.TestCase):
    def test_action_signs_are_diagnostic_only_under_new_protocol(self):
        result = evaluate(dict(positive_delta_mean_response_logprob=.06,
            negative_delta_mean_response_logprob=-.04,
            positive_delta_mean_action_logprob=-1e-7,
            negative_delta_mean_action_logprob=5e-5))
        self.assertTrue(result['passed'])
        self.assertFalse(result['action_span_signs_ok'])
        self.assertFalse(result['action_span_blocking'])

    def test_full_response_failure_still_blocks(self):
        for positive, negative in [(0., -.1), (.1, 0.), (-.1, -.1), (.1, .1)]:
            self.assertFalse(evaluate(dict(positive_delta_mean_response_logprob=positive,
                negative_delta_mean_response_logprob=negative,
                positive_delta_mean_action_logprob=.1,
                negative_delta_mean_action_logprob=-.1))['passed'])

    def test_nonfinite_diagnostic_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(dict(positive_delta_mean_response_logprob=math.nan))

    def test_training_and_loss_loop_ast_matches_original(self):
        here = Path(__file__).resolve().parent
        old = ast.parse((here.parent / 'v3_fp32_fast/trainer.py').read_text())
        new = ast.parse((here / 'trainer.py').read_text())
        def learning_loop(module):
            function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            return next(n for n in function.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
        self.assertEqual(ast.dump(learning_loop(old)), ast.dump(learning_loop(new)))


if __name__ == '__main__':
    unittest.main()
