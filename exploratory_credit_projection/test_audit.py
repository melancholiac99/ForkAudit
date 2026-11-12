"""Synthetic CPU-only checks; these are NOT results on ForkAudit replay data."""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit


class CreditProjectionTests(unittest.TestCase):
    def test_tie_preserving_bins(self):
        labels, count = audit.tied_bins(np.array([0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 1]), 10)
        self.assertEqual(count, 4)
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[3], labels[5])

    def test_only_visible_context_enters_features(self):
        t = audit.feature_text('Task: look for apple', 'go to shelf')
        self.assertIn('look for apple', t)
        self.assertIn('go to shelf', t)
        self.assertNotIn('hidden', t)

    def test_arm_order_and_repetitions_are_validated(self):
        record = dict(point_id='s1000_t01', seed=1000, game_file='kitchen0',
                      prompt='The task is to find apple', factual='go to shelf',
                      alternatives=['go to sink', 'go to fridge', 'go to counter', 'go to drawer'])
        replay = dict(factual_action='go to shelf', alternatives=record['alternatives'],
                      factual_outcomes=[0, 0, 1], alt_outcomes=[[0, 0, 0]] * 4)
        point = audit._make_point(record, replay, 'e4', 3)
        self.assertEqual(point.actions[0], 'go to shelf')
        self.assertEqual(len(point.outcomes), 5)
        replay['alternatives'] = list(reversed(record['alternatives']))
        with self.assertRaisesRegex(ValueError, 'arm order'):
            audit._make_point(record, replay, 'e4', 3)

    def test_heldout_replicate_does_not_select_itself(self):
        # On the first two repetitions A looks better than B; the third
        # independent repetition reverses that result.
        p = audit.Point('p', 'game', 'task', ('a', 'b', 'c', 'd', 'e'),
                        ((1, 1, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
        y = np.array([[2/3, 1/3, 0, 0, 0]])
        model = np.array([[.8, .2, .1, .1, .1]])
        _, heldout = audit._rows([p], y, model)
        entry = next(r for r in heldout if r['heldout_rep'] == 2)
        self.assertEqual(entry['gap'], 1.0)
        self.assertEqual(entry['heldout_diff'], -1.0)

    def test_group_heldout_predictions_are_finite(self):
        pts = []
        for i in range(12):
            q = ((1, 1, 0), (0, 0, 1), (0, 1, 0), (0, 0, 0), (0, 0, 0))
            pts.append(audit.Point(f'p{i}', f'g{i}', f'Task {i % 2}: find apple',
                                   ('go to shelf', 'go to sink', 'go to fridge', 'open drawer', 'look'), q))
        y, pred, base, folds = audit.crossfit(pts, folds=3)
        self.assertEqual(y.shape, (12, 5))
        self.assertEqual(pred.shape, (12, 5))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(sum(f['heldout_points'] for f in folds), 12)
        self.assertTrue(np.isfinite(base).all())
        self.assertGreaterEqual(pred.min(), 0)
        self.assertLessEqual(pred.max(), 1)

    def test_archive_parts_read_and_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'exploratory_E4' / 'results_archive'
            base.mkdir(parents=True)
            rec = dict(point_id='s1000_t01', seed=1000, game_file='g0',
                       prompt='find an apple', factual='go shelf',
                       alternatives=['go sink', 'go fridge', 'go table', 'go drawer'], cf_defined=True)
            replay = dict(seed=1000, turn=1, factual_action=rec['factual'],
                          alternatives=rec['alternatives'], cf_defined=True,
                          factual_outcomes=[0, 0, 1], alt_outcomes=[[0, 0, 0]] * 4)
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w:gz') as tar:
                for name, payload in [('data/points/s1000_t01.json', rec),
                                      ('data/replay/s1000/t01.json', replay)]:
                    data = json.dumps(payload).encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            data = stream.getvalue()
            cut = len(data) // 2
            parts = []
            for i, content in enumerate((data[:cut], data[cut:])):
                name = f'part{i}'
                (base / name).write_bytes(content)
                import hashlib
                parts.append(dict(path=name, bytes=len(content),
                                  sha256=hashlib.sha256(content).hexdigest()))
            (base / 'archive_manifest.json').write_text(json.dumps(dict(parts=parts)))
            pts, sources = audit.load_e4(root, archive='yes')
            self.assertEqual(len(pts), 1)
            self.assertEqual(pts[0].group, 'e4:g0')
            self.assertEqual(len(sources), 2)
            (base / 'part1').write_bytes(b'corrupt')
            with self.assertRaises((FileNotFoundError, ValueError)):
                audit.load_e4(root, archive='yes')


if __name__ == '__main__':
    unittest.main()
