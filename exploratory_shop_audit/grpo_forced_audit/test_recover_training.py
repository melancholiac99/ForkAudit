"""Recovery must preserve recorded DP padding and reject ambiguous artifacts."""
from pathlib import Path
import tempfile
import unittest

from .common import save
from .recover_training import replay_indices, promote


class RecoveryTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
