"""CPU tests for scientific invariants of the measurement adapter."""
import unittest
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.runtime import make_input, parse_generation
from exploratory_shop_audit.data import stable_seed


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = '\n'.join(m['role'] + ':' + m['content'] for m in messages)
        return list(text.encode()) if tokenize else text


class RuntimeTests(unittest.TestCase):
    def test_brackets_and_unicode(self):
        self.assertEqual(parse_action_fixed('click[蓝色[大号]\n500ml]'), ('click', '蓝色[大号]\n500ml'))
        self.assertEqual(parse_action_fixed('click[]'), ('click', ''))
        self.assertEqual(parse_action_fixed('oops click[购买]'), ('oops click[购买]', None))

    def test_truncation_drops_complete_pairs(self):
        t = Tokenizer()
        minimal = make_input(t, '任务', [], '当前')['token_count']
        x = make_input(t, '任务', [('最早' * 100, '回答一'), ('晚一点', '回答二')], '当前', minimal + 100)
        self.assertEqual(len(x['removed_pairs']), 1)
        self.assertIn('晚一点', x['text'])
        self.assertNotIn('最早', x['text'])
        self.assertEqual(x['token_ids'], list(x['text'].encode()))

    def test_current_observation_cannot_be_silently_cut(self):
        with self.assertRaisesRegex(ValueError, 'CURRENT_OBSERVATION_OVERFLOW'):
            make_input(Tokenizer(), 'task', [], 'very large observation', limit=10)

    def test_banks_and_drafts_have_independent_stable_streams(self):
        seeds = {stable_seed('state', 'candidate', bank, repeat) for bank in ['A', 'B'] for repeat in range(6)}
        self.assertEqual(len(seeds), 12)
        self.assertEqual(stable_seed('A', 0), stable_seed('A', 0))

    def test_generation_parser_keeps_invalid_requests(self):
        self.assertEqual(parse_generation('想法\nAction: click[蓝色[大号]]'), 'click[蓝色[大号]]')
        self.assertEqual(parse_generation('不能决定'), '不能决定')


if __name__ == '__main__':
    unittest.main()
