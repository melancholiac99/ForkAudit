from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from exploratory_shop_audit.runtime import parse_generation
from .first_command_replay import first_command
from .first_command_report import action_counts,action_summary
from .first_command_entropy import quantities
from .gap_eval import run_batch


class FirstCommandTest(TestCase):
    def select(self,raw):
        return first_command(raw,parse_generation(raw))

    def test_complete_commands(self):
        cases=[('click[1]\nclick[Buy Now]','click[1]'),
               ('Action: click[外[内]壳] click[draw]','click[外[内]壳]'),
               ('click[B]\\nAction: click[C]','click[B]'),
               ('Thought: compare the options\nAction: click[wrong]\nsearch[another]','click[wrong]'),
               ('search_products[query]\nclick[1]','search_products[query]')]
        for raw,expected in cases:
            self.assertEqual(self.select(raw),(expected,True))

    def test_ordinary_replies_retain_native_parse(self):
        for raw in ('click[1]','Thought: see click[X]\nAction: click[Y]','click[1]\nclick[unterminated',
                    'Action: click[a[inner]b]','click[unterminated','click[1]\nthanks','Buy Now'):
            self.assertEqual(self.select(raw),(parse_generation(raw),False))

    def test_history_and_real_next_observation(self):
        raw='click[item]\nclick[Buy Now]'
        calls=[]
        class Engine:
            def call(self,cmd,**kw):
                if cmd=='open':return dict(task='goal',observation='page1')
                if cmd=='step':
                    calls.append(kw['action'])
                    return dict(state=dict(task='goal',observation='real page2'),legal=True,status=dict(
                        done=len(calls)==2,reward_valid=True,termination_reason='gold_purchase'))
        class Model:
            def call(self,req):
                self.seeds=req[0]['seed']
                return [dict(text=raw if len(calls)==0 else 'click[Buy Now]')]
        class Runner:
            engine,model,tokenizer=Engine(),Model(),None
        class Intervention:
            select=staticmethod(first_command)
            def event(self,g,state,text,act,cut,legal):return dict(text=text,action=act,truncated=cut,legal=legal)
        def prompt(tok,goal,h,obs):
            if calls:self.assertEqual((h,obs),([['page1',raw]],'real page2'))
            return {'token_ids':[1]}
        with TemporaryDirectory() as d,patch(__package__+'.gap_eval.make_input',prompt):
            result=run_batch(Runner(),'T_opd_eval',[3],2,Path(d),'free',intervention=Intervention())
        self.assertEqual(calls,['click[item]','click[Buy Now]'])
        self.assertEqual(Runner.model.seeds,2203)
        x=action_summary(action_counts(result[0]['events']))
        self.assertEqual((x['truncations'],x['actions'],x['truncated_legal_rate'],x['subsequent_actions']),(1,2,1.,1))

    def test_entropy_masks_duplicate_and_top1_can_switch(self):
        mask=np.array([[True,True,False]])
        h,p=quantities(np.array([[0.,0.,100.]]),mask)
        self.assertAlmostEqual(h[0],np.log(2))
        self.assertEqual(p[0],.5)
        hnew,pnew=quantities(np.array([[0.,2.,100.]]),mask)
        self.assertLess(hnew[0],h[0])
        self.assertGreater(pnew[0],p[0])

    def test_resident_model_requires_identical_cold_batch(self):
        from unittest.mock import Mock
        from .first_command_replay import ReplaySession
        runner = Mock()
        runner.model.models = [Mock()]
        runner.model.models[0].call.return_value = 'reset'
        runner.model.call.return_value = [{'token_ids':[1, 2]}]

        def generate(runner, *args, intervention):
            runner.model.call([])
            return ['same text']

        with patch(__package__+'.first_command_replay.Runner',return_value=runner), \
             patch(__package__+'.first_command_replay.run_batch',side_effect=generate):
            for diverge in (False, True):
                replay = ReplaySession()
                self.assertFalse(replay.acquire('folder','model')[1])
                replay.evaluate_batch([],None)
                runner.model.call.return_value = [{'token_ids':[1, 3] if diverge else [1, 2]}]
                self.assertEqual(replay.acquire('next','model')[1],not diverge)
                self.assertEqual(replay.disable_reuse,diverge)
                replay.close()
