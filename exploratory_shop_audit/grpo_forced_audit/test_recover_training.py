"""Recovery must preserve recorded DP padding and reject ambiguous artifacts."""
from pathlib import Path
import tempfile
import unittest

from .common import save, append, read, read_rows, digest_file
from .recover_training import replay_indices, promote, replace_fit_adjust, prepare_authorized_continuation


class RecoveryTests(unittest.TestCase):
    def test_replaces_the_compiled_fit_scope_not_only_module_scope(self):
        original = lambda: 'old'
        replacement = lambda: 'new'
        scope = {'adjust_batch': original}
        exec('def fit(): return adjust_batch()', scope)
        replace_fit_adjust(scope['fit'], original, replacement)
        self.assertEqual(scope['fit'](), 'new')
        with self.assertRaises(AssertionError):
            replace_fit_adjust(scope['fit'], original, replacement)

    def rows(self, n):
        return [{'trajectory_id': f't:{i}', 'task_id': f'task:{i}', 'step': 0} for i in range(n)]

    def test_recorded_order_preserves_all_rows_and_duplicate_multiplicity(self):
        for n in [16, 542, 577, 560]:
            source = self.rows(n)
            original = list(range(n)) + list(range((-n) % 16))
            recorded = [source[i] for i in original[::-1]]
            self.assertEqual(replay_indices(source, recorded), original[::-1])

    def test_missing_foreign_or_repeated_padding_is_rejected(self):
        source = self.rows(17)
        for bad in [source, source + source[:14] + [self.rows(99)[-1]],
                    source + [source[0]] * 15, source[1:] + source[1:]]:
            with self.assertRaises(AssertionError):
                replay_indices(source, bad)
        with self.assertRaises(AssertionError):
            replay_indices(source + [source[0]], source + source[:14])

    def test_unverified_update_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / 'attempt'
            save(attempt / 'verified_update_41.json', {'passed': True})
            save(attempt / 'verified_update_42.json', {'passed': False})
            with self.assertRaises(AssertionError):
                promote(root, attempt)
            self.assertFalse((root / 'training/checkpoints/global_step_42').exists())

    def test_intermediate_recovery_keeps_original_validation_grid(self):
        self.assertEqual(list(range((42 // 10 + 1) * 10, 101, 10)), [50, 60, 70, 80, 90, 100])

    def test_authorized_resume_preserves_original_tail_and_refuses_second_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            save(run / 'recovery_0042/attempt_02/verified_update_41.json', {'passed': False})
            for step in range(1, 43):
                append(run / 'training_metrics.jsonl', {'update': step})
                append(run / 'weight_updates.jsonl', {'before': str(step - 1), 'after': str(step)})
            original = digest_file(run / 'training_metrics.jsonl')
            decisions = {}
            for step in [41, 42]:
                p = run / f'alignment_live/step_{step:04d}_test_decisions.json'
                save(p, self.rows(16))
                decisions[str(step)] = str(p)
            save(run / 'recovery_0042/request.json', {'decision_files': decisions})
            prepare_authorized_continuation(run)
            archive = run / 'recovery_0042/pre_continuation_archive'
            self.assertEqual(digest_file(archive / 'training_metrics.jsonl'), original)
            self.assertEqual(len(read_rows(run / 'training_metrics.jsonl')), 40)
            self.assertEqual(len(read_rows(run / 'weight_updates.jsonl')), 40)
            request = read(run / 'recovery_0042/continued_replay_request.json')
            self.assertTrue(request['researcher_approved_non_bitwise_resume'])
            self.assertTrue(all(Path(p).exists() for p in request['decision_files'].values()))
            with self.assertRaises(AssertionError):
                prepare_authorized_continuation(run)


if __name__ == '__main__':
    unittest.main()
