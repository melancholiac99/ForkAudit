"""Pure CPU tests for E5-L: python -m unittest -v test_e5l.py"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import e5l as L


def obs(*titles):
    return '<information>' + json.dumps({'result': ''.join('Doc %d: "%s"\nexample\n' % (i + 1, t)
                                                            for i, t in enumerate(titles))}) + '</information>'


class TestE5L(unittest.TestCase):
    def test_replay_classes(self):
        f = lambda fact, alt: dict(factual_outcomes=fact, alt_outcomes=alt)
        self.assertEqual(L.classify(f([0, 0, 0], [[0]*3]*4)), 'zero')
        self.assertEqual(L.classify(f([1, 1, 1], [[1]*3]*4)), 'equal_other')
        self.assertEqual(L.classify(f([0, 0, 0], [[0]*3]*3 + [[1, 0, 0]])), 'gap')
        with self.assertRaises(ValueError):
            L.classify(f([0], [[0]*3]*4))

    def test_support_and_novelty_distinct(self):
        self.assertEqual(L.titles_in(obs('First Page', 'Second Page')), {'first page', 'second page'})
        acts = ['<search>old</search>', '<search>new</search>', '<answer>guess</answer>']
        seen = {'first page'}
        o = [obs('First Page'), obs('Second Page', 'Third Page'), '']
        self.assertEqual(L.choose_support(acts, o, seen, ['First Page', 'Second Page']), 1)
        self.assertEqual(L.choose_novelty(acts, o, seen), 1)
        self.assertIsNone(L.choose_support(acts, o, seen, ['First Page', 'Unknown']))
        self.assertIsNone(L.choose_novelty(['<answer>a</answer>'] + acts[1:], o, seen))
        self.assertIsNone(L.choose_novelty(acts, [obs('X'), obs('X'), ''], set()))

    def test_matched_sampling_deterministic(self):
        pools = {}
        for arm in L.ARMS:
            pools[arm] = [dict(point_id=f'{arm}-{i:03d}', task_type='bridge', prompt='q',
                               chosen='a', rejected='b') for i in range(40)]
        first, n, q = L.select_rows(pools, dict(gap_types={'bridge': 300}), cap=32, min_n=16)
        second, nn, qq = L.select_rows(pools, dict(gap_types={'bridge': 300}), cap=32, min_n=16)
        self.assertEqual((n, q), (32, {'bridge': 32}))
        self.assertEqual(n, nn)
        self.assertEqual(q, qq)
        self.assertEqual(first, second)
        self.assertEqual(len(first['Szero']), 32)
        with self.assertRaises(RuntimeError):
            L.select_rows({a: p[:2] for a, p in pools.items()}, dict(gap_types={'bridge': 300}),
                          cap=32, min_n=16)

    def test_available_requires_alignment_and_keeps_class(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tid = 's10000'
            step = dict(turn=1, action='<search>old</search>', response='raw old', valid=True,
                        prompt='the decision', obs_after=obs('First Page'))
            L.save(root / 'rollout' / 's10000.json', dict(traj_id=tid, game_file='q1',
                   initial_obs='question', meta=dict(type='bridge', level='hard', supporting_titles=['Support A','Support B']),
                   steps=[step]))
            actions = ['<search>old</search>', '<search>new</search>', '<search>x</search>',
                       '<search>y</search>', '<search>z</search>']
            bank = dict(cf_defined=True, factual_action=actions[0], alternatives=actions[1:],
                        factual_outcomes=[0, 0, 0], alt_outcomes=[[1, 0, 0]] + [[0, 0, 0]] * 3,
                        drafts=[dict(action=a, text=f'<think>candidate</think>{a}', valid=True, admissible=True)
                                for a in actions[1:]], continuations=[dict(refeed_div=0)])
            peek = dict(peeks=[dict(action=a, obs=o, refeed_div=0) for a, o in zip(actions,
                        [obs('First Page'), obs('Support B','New Page'), obs('Other'), '', ''])],
                        factual_obs_match=True)
            L.save(root / 'replay_b1' / tid / 't01.json', bank)
            L.save(root / 'peek' / tid / 't01.json', peek)
            pool, counts = L.available(root)
            self.assertEqual(counts['gap'], 1)
            self.assertEqual(len(pool['Sgap']), 1)
            self.assertEqual(len(pool['Ngap']), 1)
            self.assertEqual(pool['Sgap'][0]['preferred'], 1)
            self.assertEqual(pool['Sgap'][0]['rejected'], 'raw old')
            bank['alt_outcomes'] = [[0, 0, 0]] * 4
            L.save(root / 'replay_b1' / tid / 't01.json', bank)
            pool, counts = L.available(root)
            self.assertEqual(counts['zero'], 1)
            self.assertEqual(len(pool['Szero']), 1)
            peek['factual_obs_match'] = False
            L.save(root / 'peek' / tid / 't01.json', peek)
            with self.assertRaises(ValueError):
                L.available(root)

    def test_freeze_train_dev_disjoint(self):
        with tempfile.TemporaryDirectory() as td:
            old_here, old_data, old_e5 = L.HERE, L.DATA, L.E5
            try:
                L.HERE, L.DATA, L.E5 = Path(td), Path(td) / 'data', Path(td) / 'missing_e5'
                def hotpot(i):
                    return dict(_id=str(i), question='Question '+str(i), answer='correct',
                                supporting_facts=[['Page A', 0], ['Page B', 0]],
                                type='bridge', level='hard', context=[])
                L.save(Path(td) / 'train.json', [hotpot(i) for i in range(10)])
                L.save(Path(td) / 'dev.json', [hotpot(i) for i in range(30, 40)])
                with open(Path(td) / 'wiki.jsonl', 'w') as f:
                    for t in ('Page A', 'Page B'):
                        f.write(json.dumps({'contents':t+'\ntext'})+'\n')
                args=SimpleNamespace(hotpot_train=str(Path(td)/'train.json'),
                    hotpot_dev=str(Path(td)/'dev.json'), wiki18=str(Path(td)/'wiki.jsonl'),
                    exclude_eval=[],n_train=3,n_pilot=2,n_eval=4)
                L.freeze(args)
                ids = [{q['qid'] for q in L.load(Path(td)/f'tasks_{p}.json')['tasks']}
                       for p in ('train','pilot','eval')]
                self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
                with self.assertRaises(FileExistsError):
                    L.freeze(args)
            finally:
                L.HERE, L.DATA, L.E5 = old_here, old_data, old_e5

    def test_ci_and_assessment(self):
        self.assertEqual(L.ci([0.2] * 5), [20., 20.])
        self.assertEqual(L.quotas({'b':.5,'a':.5}, 3), {'a':2,'b':1})


if __name__ == '__main__':
    unittest.main()
