"""CPU-only tests for the review version of Experiment 1c.

No Qwen checkpoint, GPU, TextWorld installation or external model service required.
Includes a small synthetic prepare -> score-fixture -> analyze test.
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import numpy as np

import exp1c as C


class FakeAlfGame:
    """A deterministic environment with two independent replay continuations."""
    def __init__(self, game_file):
        self.game_file=game_file
        self.reset()

    def load(self, game_file):
        self.game_file=game_file
        self.reset()

    def reset(self):
        self.state='start'
        self.count=0
        return 'start', [], False

    def refeed(self, actions, expected_obs=None):
        self.reset()
        assert not actions
        assert not expected_obs
        return 'start', [], False, 0

    def step(self, action):
        self.count+=1
        if action=='open right':
            self.state='found key'
            return self.state, [], True, True
        if action=='open left':
            self.state='empty'
            return self.state, [], False, False
        if action=='wait':
            self.state='still empty'
            return self.state, [], False, True
        raise AssertionError('Unexpected action: '+action)

    def close(self):pass


class Tests(unittest.TestCase):
    def test_written_cot_preserves_commands_and_does_not_touch_original(self):
        ep={'task':'a task','initial_observation':'hello','steps':[
            {'turn':1,'response':'<think>Visible reasoning</think>\n<action>go to desk 1</action>',
             'executed_action':'go to desk 1','observation_after':'desk'}],'final_reward':1}
        x,n=C.remove_visible_think(ep)
        self.assertEqual(n,1)
        self.assertNotIn('Visible reasoning',json.dumps(x))
        self.assertIn('<action>go to desk 1</action>',x['steps'][0]['response'])
        self.assertIn('<think>',ep['steps'][0]['response'])

    def test_truncated_has_step_t_but_no_later_observation_or_reward(self):
        ep={'task':'task','initial_observation':'start','steps':[{'turn':1,'observation_after':'left'},
             {'turn':2,'observation_after':'secret future'}],'final_reward':1}
        x=C.truncate_after_t(ep,1)
        self.assertEqual(len(x['steps']),1)
        self.assertNotIn('secret future',json.dumps(x))
        self.assertNotIn('final_reward',x)

    def test_bank1_rep0_not_best_reward(self):
        cont0=dict(role=2,rep=0,won=False)
        cont1=dict(role=2,rep=1,won=True)
        c=C.replay_continuation({'continuations':[cont1,cont0]},1)
        self.assertIs(c,cont0)

    def test_reconstruct_detects_false_observation(self):
        tr={'traj_id':'s000','steps':[{'obs_before':'start','action':'open left','obs_after':'empty'}]}
        c={'first_action':'open left','actions':['open left','wait'],'won':False,'len':2}
        with self.assertRaisesRegex(AssertionError,'first feedback'):
            C.reconstruct(FakeAlfGame('fake'),tr,1,c,'INCORRECT')
        with self.assertRaisesRegex(AssertionError,'outcome changed'):
            C.reconstruct(FakeAlfGame('fake'),tr,1,dict(c,won=True),'empty')

    def test_dual_anonymous_provenance_and_random_order(self):
        fac={'steps':[{'turn':1,'action':'a','observation_after':'a obs'}],'terminal_success':0}
        alt={'steps':[{'turn':1,'action':'b','observation_after':'b obs'}],'terminal_success':1}
        for i in range(20):
            t,order=C.dual_text('M',f's{i:03d}_t01',fac,alt)
            self.assertIn('branch_A',t['D0'])
            self.assertNotIn('Original-agent provenance',t['D0'])
            self.assertIn('Original-agent provenance',t['DE'])
            self.assertEqual(t['DE'].replace(C.LABEL.format(branch=order['factual_branch']),''),t['D0']+'\n')
            self.assertEqual(t['DX'].replace(C.LABEL.format(branch=order['contrast_branch']),''),t['D0']+'\n')
            self.assertNotEqual(order['factual_branch'],order['contrast_branch'])
        self.assertEqual(C.branch_order('s006_t01'),C.branch_order('s006_t01'))

    def test_request_hash_must_change_with_label(self):
        s={'point_id':'s001_t01','traj_id':'s001','turn':1,'actions':['a','b','c','d','e']}
        a=C.make_request(s,'DE','Executed branch A')
        b=C.make_request(s,'DX','Executed branch B')
        self.assertNotEqual(a['request_sha256'],b['request_sha256'])
        self.assertEqual(a['actions'],b['actions'])

    def test_cluster_bootstrap_weighted_point_estimate(self):
        v=C.mean_ci([1,0,0],['t1','t1','t2'],nboot=100)
        self.assertEqual(v['n_trajectories'],2)
        self.assertAlmostEqual(v['estimate'],1/3)

    def test_synthetic_cpu_prepare_and_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for d in ('exploratory_P4/prompts','exploratory_deep_discovery','data/rollout','data/replay',
                      'hindsight_judges','games'):(root/d).mkdir(parents=True,exist_ok=True)
            (root/'exploratory_P4/prompts/feedback.txt').write_text('Five aligned feedback:\n')
            arms_file=root/'exploratory_deep_discovery/exploratory_arms.parquet'
            arms_file.write_bytes(b'synthetic arms')
            tasks=[dict(seed=i,game_file=f's{i:03d}/game.tw-pddl',game_sha256=C.sha(b'fake game')) for i in range(100)]
            (root/'data/task_list.json').write_text(json.dumps({'tasks':tasks}))
            (root/'exploratory_P4/e2.json').write_text(json.dumps({'inputs':{'exploratory_deep_discovery/exploratory_arms.parquet':C.file_sha(arms_file)}}))
            meta=[];e2={};e2m={};template='Original full episode: '
            acts=['open left','open right','look','wait','inventory']
            for i in range(2):
                traj=f's{i:03d}';pid=f'{traj}_t01'
                (root/'data/replay'/traj).mkdir(parents=True)
                (root/'games'/traj).mkdir(parents=True)
                (root/'games'/traj/'game.tw-pddl').write_text('fake game')
                tr={'traj_id':traj,'game_file':f'{traj}/game.tw-pddl','task':'find key','initial_obs':'start',
                    'steps':[{'turn':1,'obs_before':'start',
                    'action':'open left','obs_after':'empty','response':'<think>Thought</think><action>open left</action>'}]}
                (root/'data/rollout'/f'{traj}.json').write_text(json.dumps(tr))
                replay={'traj_id':traj,'turn':1,'cf_defined':True,'factual_action':acts[0],
                    'alternatives':acts[1:],'factual_outcomes':[0],
                    'alt_outcomes':[[1],[0],[0],[0]],'continuations':[
                    {'role':1,'rep':0,'first_action':'open left','actions':['open left','wait'],
                     'len':2,'won':False,'refeed_div':0},
                    {'role':2,'rep':0,'first_action':'open right','actions':['open right'],
                     'len':1,'won':True,'refeed_div':0}]}
                (root/'data/replay'/traj/'t01.json').write_text(json.dumps(replay))
                ep={'task':'find key','initial_observation':'start','steps':[{'turn':1,
                    'response':'<think>Why</think><action>open left</action>',
                    'executed_action':'open left','observation_after':'empty'}],'final_reward':0}
                s={'point_id':pid,'traj_id':traj,'turn':1,'actions':acts,
                   'text':'S base\nCandidates:\n'+json.dumps(acts)}
                ll=dict(s,text=s['text']+'\n'+template+json.dumps(ep))
                feedback=[{'action':a,'response':('found key' if a=='open right' else 'empty')}
                          for a in acts]
                ff=dict(s,text=s['text']+'\nFive aligned feedback:\n'+json.dumps(feedback))
                mm=dict(s,text=s['text']+'\nHistory before marked decision')
                e2[pid]={'S':s,'F':ff,'L':ll};e2m[pid]={'M':mm}
                meta.append(dict(point_id=pid,traj=traj,layer='I' if i==0 else 'U',
                                 swap_arm=1,q12_gap=2/3,factual_arm=0))
            # Simulate the real repository: some originals are absent, but E2 L has every prefix.
            (root/'data/rollout/s001.json').unlink()
            out=root/'hindsight_judges/results/exp1c_p4'
            old_n=C.N_POINTS
            try:
                C.N_POINTS=2
                fake=ModuleType('alfenv')
                fake.ALFWORLD_DATA=str(root/'games')
                fake.AlfGame=FakeAlfGame
                with patch.object(C,'original_sources',return_value=(meta,e2,e2m,template,{'TEST':'synthetic'})),\
                     patch.dict(sys.modules,{'alfenv':fake}),\
                     patch.dict(os.environ,{'ALFWORLD_DATA':str(root/'games')}):
                    ns=SimpleNamespace(repo=root,out=out,replay_root=root/'data/replay',arms=arms_file)
                    C.preflight(ns)
                    C.prepare(ns)
                requests,aa,manifest=C.prepared(out)
                self.assertEqual(len(requests),2*len(C.CONDITIONS))
                self.assertEqual(aa[0]['think_blocks'],1)
                self.assertEqual(manifest['source_sha256']['P4-ARMS'],C.file_sha(arms_file))
                # Synthetic 32B log-prob scores: dual provenance points to whichever branch is named.
                scores=[]
                for r in requests:
                    k=r['condition']
                    odds={'M':0,'L':1.8,'LN':1.2,'LT':1.4,'D0':-1.0,'DE':0.9,'DX':-2.2}[k]
                    logits=np.array([odds,0,-8,-8,-8],float)
                    xx=np.exp(logits-logits.max());pr=(xx/xx.sum()).tolist()
                    scores.append(dict(request_id=r['request_id'],request_sha256=r['request_sha256'],
                        point_id=r['point_id'],condition=k,sequence_logp=logits.tolist(),
                        candidate_prob=pr,argmax=int(np.argmax(logits))))
                (out/'32b').mkdir()
                raw='\n'.join(json.dumps(s) for s in scores)+'\n'
                (out/'32b/scores.jsonl').write_text(raw)
                C.save(out/'32b/run.json',{'complete':True,'contexts':len(requests),
                    'requests_sha256':manifest['requests_sha256'],'audit_sha256':manifest['audit_sha256'],
                    'scores_sha256':C.file_sha(out/'32b/scores.jsonl')})
                qs={z['point_id']:{j:dict(q12=(0.0 if j==0 else (2/3 if j==1 else 0.0)),
                    q3456=(0.0 if j==0 else (0.8 if j==1 else 0.0)),
                    qgold=(0.0 if j==0 else (0.8 if j==1 else 0.0))) for j in range(5)} for z in meta}
                with patch.object(C,'read_arms',return_value=qs):
                    C.analyze(SimpleNamespace(repo=root,out=out,arms=arms_file))
                report=C.load(out/'32b/analysis.json')
                self.assertGreater(report['metrics']['all']['dual_pointer_swap_nat']['estimate'],0)
                self.assertGreater(report['metrics']['all']['dual_no_pointer_minus_true_value']['estimate'],0)
                self.assertIn('CAUTION: D0 vs L', (out/'32b/REPORT.txt').read_text())
            finally:
                C.N_POINTS=old_n


if __name__=='__main__':unittest.main()
