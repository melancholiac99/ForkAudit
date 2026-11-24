import csv
import gzip
import unittest
from collections import Counter

from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read

import numpy as np

from exploratory_shop_audit.visible_opd_v0.gap_cpu_actions import classify, legality
from exploratory_shop_audit.visible_opd_v0.gap_cpu_packet import OUT, cluster_mean
from exploratory_shop_audit.visible_opd_v0.gap_cpu_targets import sampled_bounds, sampled_columns, summarize


class NativeActionTest(unittest.TestCase):
    def test_parse_and_click_taxonomy(self):
        public = {'products':[{'asin':'123456770012'}]}
        options = {'red'}
        def result(action, page='search_results'):
            record = dict(action=action,page=page,legal_clicks=['123456770012'])
            self.assertFalse(legality(record))
            return classify(record,'123456779999',public,options)
        self.assertEqual(result('click[123456770012')[0],'format_unparseable')
        self.assertEqual(result('buy_now[]')[0],'action_type_unavailable')
        self.assertEqual(result('click[red]')[:2],('action_type_unavailable','option'))
        self.assertEqual(result('click[red]','product_detail')[:2],('unavailable_option','option'))
        self.assertEqual(result('click[next >]')[0],'nonexistent_navigation')
        self.assertEqual(result('click[one]\nclick[two]')[0],'other')
        reason,_,detail = result('click[123456779999]')
        self.assertEqual(reason,'absent_product_id')
        self.assertTrue(detail['partial'] and detail['gold'])
        self.assertFalse(result('click[987009870098]')[2]['partial'])
        self.assertTrue(legality(dict(action='finish',legal_clicks=[])))
        self.assertTrue(legality(dict(action='click[RED]',legal_clicks=['red'])))
        self.assertFalse(legality(dict(action='click[search]',legal_clicks=['search'])))


class ArchivedScoreTest(unittest.TestCase):
    def test_sampled_column_handles_masked_duplicate(self):
        ids = np.array([[6,4,4],[6,4,7]])
        mask = np.array([[1,1,0],[1,1,1]],dtype=bool)
        np.testing.assert_array_equal(sampled_columns(ids,mask,[4,7]),[1,2])
        with self.assertRaises(ValueError):
            sampled_columns(ids,np.ones_like(mask),[4,7])

    def test_non_identifiability_and_bounds(self):
        q = np.array([[.25,.375,.375]]*2, dtype=np.float32)
        p = np.array([[.20,.20,.60],[.20,.60,.20]])
        strength = (q*(np.log(q)-np.log(p))).sum(1)
        self.assertAlmostEqual(strength[0],strength[1])
        exact = np.log(q[:,1]/p[:,1])
        self.assertGreater(exact[0],0)
        self.assertLess(exact[1],0)
        limited = sampled_bounds(q.astype(float),[1,1],strength)
        self.assertTrue(np.all(limited[:,0]<=exact) and np.all(limited[:,1]>=exact))

    def test_unequal_trajectory_lengths_and_unscored_records(self):
        result = cluster_mean([2,2,3],[[1.],[3.],[8.]])
        self.assertEqual(result['mean'],[4.])
        self.assertEqual(result['ci95'],[[2.,8.]])
        rows = [
            ['R-gap',1,2,0,0,True,'click',2,1.,1.,1.],
            ['R-gap',2,3,0,1,True,'click',3,None,-3.,5.],
            ['R-gap',3,2,0,1,True,'click',0,None,None,None],
        ]
        r = summarize(rows)
        self.assertEqual((r['records'],r['scored'],r['structural_na'],r['block_anchor_missing']),(3,2,1,1))
        self.assertEqual(r['exact_subset']['mean'],[1.])
        self.assertIsNone(r['exact_subset']['ci95'])
        self.assertEqual(r['all_scored_identification_bounds'],[-1.,3.])
        self.assertIsNone(summarize([])['all_scored_identification_bounds'])


class PublishedCoverageTest(unittest.TestCase):
    def test_windows_against_preexisting_independent_counts(self):
        actions = read(OUT/'illegal_actions.json')
        endpoint = read(OUT.parent/'endpoint_report.json')
        earlier = read(OUT.parent/'archived_illegal_gold_clicks.json')['training_by_update']
        self.assertEqual(len(actions['source_sha256']), 2*(45+50+30+33+30))
        for w in actions['windows']:
            arm = w['arm']
            steps = range(w['start'],w['end']+1)
            if arm.endswith('-gap'):
                m = [endpoint['training'][arm][k-1] for k in steps]
                counts = sum(r['actions'] for r in m),sum(r['actions']-r['legal'] for r in m)
            else:
                m = [earlier[arm][str(k)] for k in steps]
                counts = sum(r['actions'] for r in m),sum(r['illegal_actions'] for r in m)
            self.assertEqual((w['actions'],w['illegal']),counts)
            self.assertEqual(sum(r['n'] for r in w['types'].values()),w['illegal'])
            self.assertEqual(sum(v['n'] for v in w['absent_id_cross'].values()),w['types']['absent_product_id']['n'])

    def test_every_gap_action_is_accounted_for(self):
        actions = read(OUT/'illegal_actions.json')
        result = read(OUT/'target_score_windows.json')
        data = OUT/'sampled_action_scores.csv.gz'
        self.assertEqual(digest_file(data),result['row_file_sha256'])
        counts = Counter()
        exact_counts = Counter()
        keys = set()
        with gzip.open(data,'rt') as stream:
            for r in csv.DictReader(stream):
                update = int(r['update'])
                arm = r['arm']
                key = arm,update,r['task'],r['rep'],r['action_index']
                self.assertNotIn(key,keys)
                keys.add(key)
                valid = int(r['scored_tokens'])>0
                exact = bool(r['exact_log_ratio'])
                self.assertEqual(exact, valid and update%5==1)
                self.assertEqual(valid,bool(r['lower_log_ratio']) and bool(r['upper_log_ratio']))
                if valid:
                    self.assertLessEqual(float(r['lower_log_ratio']),float(r['upper_log_ratio']))
                index = arm,((update-1)//5)*5+1, r['legal']=='True'
                counts[index] += 1
                exact_counts[index] += exact
        for w in actions['windows']:
            if w['arm'].endswith('-gap'):
                self.assertEqual(counts[w['arm'],w['start'],False],w['illegal'])
                self.assertEqual(counts[w['arm'],w['start'],True],w['actions']-w['illegal'])
        for w in result['windows']:
            for status,table in w['by_class'].items():
                key = w['arm'],w['start'],status=='legal'
                self.assertEqual(table['all']['records'],counts[key])
                self.assertEqual(table['all']['exact_subset']['n'],exact_counts[key])
                self.assertEqual(sum(v['records'] for c,v in table.items() if c!='all'),counts[key])


if __name__=='__main__':
    unittest.main()
