"""CPU tests for the non-frozen Think localization and OPD plumbing."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import exp1c
import think_localization as loc
import exp2_opd as opd
from exp2_common import decode_rows, read_json


class HindsightExp2CPU(unittest.TestCase):
    def test_think_intervention_preserves_environment(self):
        ep={'steps':[{'turn':1,'response':'<think>A</think><action>open</action>',
                      'executed_action':'open','obs_after':'empty'},
                     {'turn':2,'response':'<think>B</think><action>look</action>',
                      'executed_action':'look','obs_after':'still empty'}]}
        for c in loc.CONDITIONS:
            r, n=loc.edit_episode(ep,1,c,'<think>DONOR</think>')
            self.assertEqual([s['executed_action'] for s in r['steps']],['open','look'])
            self.assertEqual([s['obs_after'] for s in r['steps']],['empty','still empty'])
            self.assertEqual(n,1)
        self.assertEqual(ep['steps'][0]['response'],'<think>A</think><action>open</action>')
        self.assertNotIn('<think>',loc.edit_episode(ep,1,'DROP_T')[0]['steps'][0]['response'])
        self.assertIn('<think>A',loc.edit_episode(ep,1,'KEEP_T')[0]['steps'][0]['response'])
        self.assertIn('<think>B',loc.edit_episode(ep,1,'DROP_T')[0]['steps'][1]['response'])

    def test_localize_makes_four_frozen_format_requests(self):
        meta=[]; e2={}; template='LOG\n'
        for i in range(3):
            pid=f's{i:03d}_t01'
            s=dict(point_id=pid,traj_id=f's{i:03d}',turn=1,actions=['open','look','take','put','go'],text='M')
            ep=dict(steps=[dict(turn=1,response=f'<think>reason {i}</think><action>open</action>',executed_action='open'),
                           dict(turn=2,response='<think>later</think><action>look</action>',executed_action='look')])
            meta.append(dict(point_id=pid,traj=f's{i:03d}'))
            e2[pid]={'S':s,'L':dict(text='M\n'+template+json.dumps(ep))}
        rows,audit=loc.build(meta,e2,template)
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r['request_id'] for r in rows}),12)
        anchors=[r for r in rows if r['condition']==loc.ANCHOR]
        self.assertEqual(len(anchors),3)
        self.assertEqual([r['text'] for r in anchors], [e2[z['point_id']]['L']['text'] for z in meta])
        self.assertEqual(sum(a['condition']=='SWAP_T' and a['changed_blocks']==1 for a in audit),3)
        self.assertEqual(len({a['donor_id'] for a in audit if a['condition']=='SWAP_T'}),len(set(a['donor_id'] for a in audit if a['condition']=='SWAP_T')))

    def test_teacher_context_and_training_token_preservation(self):
        opd.selftest(None)
        text='<think>think!</think><action>open</action>'
        offsets=[(0,7),(7,13),(13,21),(21,29),(29,40)]
        types=opd.token_types(text,offsets)
        self.assertEqual(types[0],'think')
        self.assertIn('other',types)

    def test_data_preparation_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            repo=tmp/'repo'
            (repo/'exploratory_E4').mkdir(parents=True)
            (repo/'exploratory_E4/games.json').write_text(json.dumps(dict(
                train=[dict(game_file='train/game.tw-pddl')],tune=[],valid_seen=[],valid_unseen=[])))
            roll=tmp/'roll';roll.mkdir()
            ep=dict(seed=1000,rep=0,game_file='train/game.tw-pddl',task='find apple',initial_obs='start',
                    won=False,provenance=dict(adapter=None),steps=[
                dict(prompt='Find apple',response='<think>search</think><action>open</action>',
                     action='open',obs_after='empty',won_after=False),
                dict(prompt='Find apple again',response='<think>try again</think><action>look</action>',
                     action='look',obs_after='empty',won_after=False)])
            (roll/'s1000_r0.json').write_text(json.dumps(ep))
            out=tmp/'out'
            a=argparse.Namespace(repo=str(repo),rollouts=str(roll),out=str(out),seed=11,max_examples=2,
                                 adapter_provenance=None)
            opd.prepare(a);opd.prepare(a)
            data,man=opd.prepared(out)
            self.assertEqual(len(data),2)
            self.assertNotIn('<think>search',data[0]['future']['LN_guarded'])
            self.assertEqual(len(man['rollout_sha256']),1)

    def test_sparse_teacher_cross_entropy_gradients(self):
        import torch
        logits=torch.randn(3,7,requires_grad=True)
        ids=torch.arange(7).repeat(3,1)
        probs=torch.softmax(torch.randn(3,7),dim=-1)
        loss,mass=opd.sparse_ce(logits,ids,probs)
        torch.testing.assert_close(mass,torch.ones(3))
        loss.mean().backward()
        self.assertEqual(tuple(logits.grad.shape),(3,7))
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad.abs().sum(-1)>0).all().item())


if __name__=='__main__':unittest.main()
