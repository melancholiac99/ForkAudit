import unittest
from exploratory_shop_audit.analyze_saved import task_equal, reward_groups, stop_metrics, funnels
from exploratory_shop_audit.derive_environment import configurations, item_text, evidence_label


class AnalysisTests(unittest.TestCase):
    def test_task_weighting_not_row_weighting(self):
        rows = [{'task_id': 'a', 'x': 1}] * 20 + [{'task_id': 'b', 'x': 0}]
        self.assertEqual(task_equal(rows, lambda r: r['x'], 'test')['estimate'], .5)

    def test_all_failed_is_not_no_reward_difference(self):
        episodes = [{'task_id': 'a', 'reward': x, 'success': False} for x in [-.85, -.35, 0, .14]]
        metric = reward_groups(episodes, 'reward')
        self.assertEqual(metric['all_failed']['estimate'], 1)
        self.assertEqual(metric['equal']['estimate'], 0)
        self.assertEqual(metric['strict_pairs'], 6)

    def test_no_purchase_and_no_stop_not_zero(self):
        ep = {'task_id': 'a', 'trajectory_id': 'e', 'purchase': False, 'reward': -.35}
        step = {'trajectory_id': 'e', 'step': 0, 'before': {'stop_available': False, 'public': {}}, 'after': {'public': {}}}
        metric, rows = stop_metrics([ep], [step], {('e', 0): {'stop_score': None}}, 'reward')
        self.assertEqual(metric['coverage']['estimate'], 0)
        self.assertIsNone(metric['opportunity']['estimate'])
        result = funnels([ep], [step], {}, 'reward')
        self.assertEqual(result['eligible_episodes'], 0)
        self.assertEqual(result['excluded'][0]['reason'], 'no_purchase')

    def test_options_last_mapping_and_partial_choices(self):
        product = {'options': {'axis_a': ['red', 'blue'], 'axis_b': ['red', 'large']}}
        combinations = list(configurations(product))
        self.assertEqual(len(combinations), 6)
        self.assertNotIn({'axis_a': 'red'}, combinations)
        self.assertIn({'axis_a': 'blue', 'axis_b': 'red'}, combinations)

    def test_product_specific_visibility(self):
        observation = 'query: 防水\n1|111|20|品牌|类目|普通|普通商品\n2|222|30|品牌|类目|防水|防水商品'
        self.assertEqual(evidence_label(item_text(observation, '111'), '防水'), 'unknown')
        self.assertEqual(evidence_label(item_text(observation, '222'), '防水'), 'support_proxy')
        self.assertEqual(evidence_label('不支持防水', '防水'), 'conflict_proxy')


if __name__ == '__main__':
    unittest.main()
