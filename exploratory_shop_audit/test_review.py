import copy
import json
import unittest

from exploratory_shop_audit.judge_prompt import build_choice_messages, CHOICE_SYSTEM
from exploratory_shop_audit.review_recompute import summarize_configurations, option_completeness, action_class
from exploratory_shop_audit.review_early_probe import landmark_features


class ReviewTests(unittest.TestCase):
    def test_invalid_zero_does_not_win_valid_diagnostic(self):
        rows = [{'options': {}, 'score': {'reward': 0, 'no_budget_gate': .2, 'valid': False, 'diagnostic_detail': {'reward_valid': True}}},
                {'options': {'size': 'x'}, 'score': {'reward': -.85, 'no_budget_gate': -.85, 'valid': True, 'diagnostic_detail': {'reward_valid': True}}}]
        result = summarize_configurations(rows)
        self.assertEqual(result['reward']['numeric_max'], 0)
        self.assertEqual(result['reward']['valid_max'], -.85)
        self.assertTrue(result['reward']['invalid_numeric_max_strictly_exceeds_valid'])
        self.assertEqual(result['no_budget_gate']['valid_configurations'], 2)

    def test_missing_valid_max_is_na(self):
        row = {'options': {}, 'score': {'reward': 0, 'no_budget_gate': 0, 'valid': False, 'diagnostic_detail': {'reward_valid': False}}}
        self.assertIsNone(summarize_configurations([row])['reward']['valid_max'])

    def test_axis_completeness_is_not_goal_correctness(self):
        product = {'options': {'Color': ['red', 'blue'], 'size': ['m']}}
        self.assertEqual(option_completeness(product, {'color': 'blue'}), 'incomplete')
        self.assertEqual(option_completeness(product, {'color': 'blue', 'size': 'm'}), 'complete')
        self.assertEqual(option_completeness({'options': {}}, {}), 'no_options')

    def test_dedicated_prompt_drops_policy_instructions_but_keeps_history(self):
        history = [{'role': 'system', 'content': '最后输出Action: 动作'}, {'role': 'user', 'content': '购物任务'},
                   {'role': 'assistant', 'content': 'Action: search[保温杯]'}, {'role': 'user', 'content': '商品观察'}]
        candidates = [{'action': 'click[1]', 'clean_feedback': '正常观察'}, {'action': 'finish[x]', 'clean_feedback': '该动作终止回合。'}]
        s = build_choice_messages(history, candidates, 'S'); f = build_choice_messages(history, candidates, 'F')
        self.assertEqual(s[0]['content'], CHOICE_SYSTEM)
        self.assertNotIn('最后输出Action:', str(s))
        sp = json.loads(s[1]['content'].rsplit('\n', 1)[0]); fp = json.loads(f[1]['content'].rsplit('\n', 1)[0])
        self.assertEqual(sp['决策前资料'], history[1:]); self.assertEqual(sp['决策前资料'], fp['决策前资料'])
        self.assertNotIn('一步正常观察', sp['候选动作'][0]); self.assertIn('一步正常观察', fp['候选动作'][0])

    def test_landmark_never_reads_current_action_or_future(self):
        first = {'action': 'search[杯]', 'legal': True, 'done': False, 'before': {'public': {}}, 'step': 0}
        second = {'action': 'click[1]', 'generation': {'text': 'future'}, 'after': {'secret': 1}, 'step': 1,
                  'input': {'messages': [{'role': 'user', 'content': '要求'}, {'role': 'assistant', 'content': 'Action: search[杯]'},
                                          {'role': 'user', 'content': '搜索结果'}], 'token_count': 50}}
        before = landmark_features([first, second], 'after_first_search')
        second['action'] = 'finish[x]'; second['generation']['text'] = 'changed'; second['after'] = {'secret': 999}
        self.assertEqual(before, landmark_features([first, second], 'after_first_search'))
        self.assertEqual(before['largest_feature_action_step'], 0)
        self.assertIsNone(landmark_features([first], 'after_first_search'))


if __name__ == '__main__': unittest.main()
