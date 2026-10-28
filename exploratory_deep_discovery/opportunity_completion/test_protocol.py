"""Small tests for intervention placement, truth extraction, seeds and paired analysis."""
import copy
from pathlib import Path
import tempfile
import unittest

from protocol import (ARMS, CONDITIONS, common, location_clues, presented_observation,
                      relation_fact, seed_for, write)
from run import advance, result_key, validate_record
from analyze import collect


class ProtocolTests(unittest.TestCase):
    def test_only_extract_observed_locations(self):
        clues = location_clues('You arrive at shelf 1. On the shelf 1, you see a mug 1 and a pen 1.',
                                'go to shelf 1', 'mug')
        self.assertEqual([(c['object'], c['location']) for c in clues], [('mug 1', 'shelf 1')])
        opened = location_clues('You open the cabinet 1. The cabinet 1 is open. In it, you see a mug 2.',
                                 'open cabinet 1', 'mug')
        self.assertEqual(opened[0]['location'], 'cabinet 1')
        self.assertEqual(location_clues('In it, you see a mug 1.', 'go to shelf 1', 'mug'), [])
        self.assertEqual(location_clues("There's nothing special about mug 1.", 'examine mug 1', 'mug'), [])
        self.assertEqual(location_clues('On the shelf 1, you see a mug 1.', 'look', 'cup'), [])

    def test_fact_validation_requires_correct_location(self):
        facts = {'inreceptacle(mug 1: object, shelf 1: receptacle)'}
        self.assertIsNotNone(relation_fact('mug 1', 'shelf 1', facts))
        self.assertIsNone(relation_fact('mug 1', 'shelf 2', facts))

    def test_native_unchanged_and_cue_ages_out(self):
        clue = 'Additional location information: mug 1 is at shelf 1.'
        self.assertEqual(presented_observation('Real feedback.', 'native', clue), 'Real feedback.')
        c = dict(obs='Before', raw_obs='Before', memory=[], steps_taken=0, transitions=[],
                 point={'clue': {'text': clue}}, condition='completed')
        advance(c, 'look', True, 'forced', None, None, ('After', ['look'], False, False), forced=True)
        self.assertEqual(c['raw_obs'], 'After')
        self.assertEqual(c['steps_taken'], 1)
        for index in range(4):
            prompt = common.build_prompt('find a mug', c['memory'], c['obs'], c['admissible'])
            self.assertEqual(clue in prompt, index < 3)
            advance(c, 'look', True, 'xml', 'response', 1, ('Later', ['look'], False, False))

    def test_original_step_cap_includes_forced_action(self):
        c = dict(obs='Before', raw_obs='Before', memory=[], steps_taken=48, transitions=[],
                 point={'clue': {'text': 'clue'}}, condition='native')
        advance(c, 'look', True, 'forced', None, None, ('After', ['look'], False, False), forced=True)
        self.assertFalse(c['done'])
        advance(c, 'look', True, 'xml', 'response', 1, ('Later', ['look'], False, False))
        self.assertTrue(c['done'])
        self.assertEqual(c['stop_reason'], 'step_cap')
        self.assertEqual(c['steps_taken'], 50)

    def test_seeds_unique_across_independent_requests(self):
        seeds = [seed_for(p, a, r, s) for p in range(3) for a in ARMS for r in range(16) for s in range(50)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seed_for(2, 'other', 15, 49), seed_for(2, 'other', 15, 49))

    def test_analysis_detects_rescue_and_refuses_incomplete(self):
        point = dict(id='case', index=0, task_id='task', task_type='test', old_fork=True,
                     old_masked=False, clue={'text': 'clue', 'object': 'mug 1'},
                     arms={a: {'action': 'look'} for a in ARMS})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'run_manifest.json', dict(run_fingerprint='test',
                  config=dict(point_ids=['case'], reps=1, dry=True), design={'points': [point]}))
            for arm in ARMS:
                for condition in CONDITIONS:
                    key = result_key(point, arm, condition, 0)
                    record = dict(key=key, run_fingerprint='test', status='complete', dry=True,
                                  point_id='case', arm=arm, condition=condition, rep=0,
                                  won=(arm == 'revealing' or condition == 'completed'),
                                  generated_steps=1, generated_tokens=0, transitions=[])
                    write(root / 'results' / f'{key}.json', record)
            _, rows, blocks, cost = collect(root)
            self.assertEqual(rows[0]['native_gap'], 1)
            self.assertEqual(rows[0]['completed_gap'], 0)
            self.assertEqual(rows[0]['other_lift'], 1)
            self.assertEqual(rows[0]['revealing_lift'], 0)
            self.assertEqual(cost['complete'], 4)
            (root / 'results' / f'{key}.json').unlink()
            with self.assertRaises(ValueError):
                collect(root)
            self.assertEqual(collect(root, allow_partial=True)[1], [])

    def test_resume_rejects_mixed_configuration(self):
        with self.assertRaises(ValueError):
            validate_record(dict(run_fingerprint='old', key='a', status='complete', won=False), 'new', 'a')


if __name__ == '__main__':
    unittest.main()
