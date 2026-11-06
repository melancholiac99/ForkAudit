"""CPU fixtures only; never create or modify formal experiment results."""
import hashlib,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import supplement_slot_analysis_20261106 as S

def probe(seqs=None):
 seqs=seqs or [[7,i+10,8] for i in range(5)];arms=[]
 for i,ids in enumerate(seqs):
  toks=[dict(target_logp={S.T1:-1.-i*.1,S.T2:-1.-i*.2,'T3':-1.-i*.3},raw_logp={'EF':-1.-i*.3,'EM':-1.-i*.1}) for _ in ids]
  arms.append(dict(ids=ids,tokens=toks))
 return dict(arms=arms,char=12,phase='pre_k')
class SupplementTests(unittest.TestCase):
 def test_true_ordinary_excludes_decision_overlap(self):
  p=dict(kind='think',reasons=['ordinary_pre','first_paragraph'])
  self.assertTrue(S.ordinary(p))
  self.assertFalse(S.ordinary(dict(p,reasons=p['reasons']+['literal_action_mention'])))
  self.assertFalse(S.ordinary(dict(p,kind='action')))
  self.assertFalse(S.ordinary(dict(p,reasons=['first_paragraph'])))
 def test_nonoverlapping_common_spans(self):
  self.assertEqual(S.bounds(probe()['arms']),(1,1))
  p=probe([[1],[1,2],[1,3],[1,4],[1,5]])
  self.assertEqual(S.bounds(p['arms']),(1,0));self.assertIsNone(S.probe_value(p,[0,.2,.4,.6,.8],S.T1,True))
  p=probe([[1,2]]*5);self.assertEqual(S.bounds(p['arms']),(2,0));self.assertIsNone(S.probe_slope(p,[0,1,2,3,4],True))
 def test_span_value_and_slope_independently(self):
  p=probe();q=np.arange(5)/4;original=q.copy()
  lp=np.array([-1.-i*.3 for i in range(5)]);a=np.exp(lp-lp.max());a/=a.sum()
  self.assertAlmostEqual(S.probe_value(p,q,'T3',True),float(100*a@q))
  self.assertAlmostEqual(S.probe_slope(p,q,True),-.8)
  self.assertTrue(np.array_equal(q,original))
  em=np.array([-1.-i*.1 for i in range(5)]);b=np.exp(em-em.max());b/=b.sum()
  self.assertAlmostEqual(S.external_value_delta(p,q,True),float(100*(a-b)@q))
 def test_lexical_features_and_limits(self):
  prefix='A long partial calculation with sufficiently detailed explanation. '*3+' The intermediate value is 144.'
  f=S.continuity_features(prefix,"First let's solve it again; 144 appears.",[1,2])
  self.assertTrue(f['restart_opener_heuristic']);self.assertTrue(f['carryover_hint']);self.assertEqual(f['continuation_token_count'],2)
  self.assertTrue(S.continuity_features(prefix,prefix,[1])['prefix_repetition_heuristic'])
  self.assertFalse(S.continuity_features('x=2','Thus this is done.',[])['carryover_hint'])
 def test_output_guard_including_custom_roots(self):
  args=SimpleNamespace(output='/tmp/scores/token/subdir',token_root='/tmp/scores/token',math_root='/tmp/scores/math')
  with self.assertRaises(AssertionError):S.output_directory(args)
  args.output='/tmp/supplement-output';self.assertEqual(S.output_directory(args),Path(args.output))
 def test_original_verification_hash_is_required(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);S.C.save_json(p/'analysis.json',{'v':1});S.C.save_json(p/'verification.json',{'passed':True,'analysis_sha':S.C.sha_file(p/'analysis.json')})
   self.assertTrue(S.require_verified(p))
   S.C.save_json(p/'analysis.json',{'v':2})
   with self.assertRaises(AssertionError):S.require_verified(p)
 def test_blind_review_linkage_and_coverage(self):
  sample=[{'id':'q1','student_prefix_tail':'prefix','continuation_X_opening':'x','continuation_Y_opening':'y'}]
  self.assertEqual(S.blind_review_summary(None,sample)['coverage'],0.)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'labels.json';doc=dict(blind_payload_sha256=hashlib.sha256(S.C.json_bytes(S.blind_payload(sample))).hexdigest(),samples=[dict(id='q1',X='genuine_continuation',Y='ambiguous',reviewer='fixture',reason='CPU test')]);S.C.save_json(p,doc)
   r=S.blind_review_summary(p,sample);self.assertEqual(r['coverage'],1.);self.assertEqual(r['status'],'complete')
   self.assertEqual(sum(sum(v.values()) for v in r['counts_by_arm'].values()),2)
   doc['samples']*=2;S.C.save_json(p,doc)
   with self.assertRaises(AssertionError):S.blind_review_summary(p,sample)
 def test_token_report_preserves_legacy_and_all_points(self):
  targets={S.T1:dict(kl_target_AF=1.,kl_AF_target=2.,logp=-1.),S.T2:dict(kl_target_AF=1.05,kl_AF_target=2.1,logp=-1.05)}
  point=dict(t=2,kind='think',phase='pre_k',reasons=['ordinary_pre'],targets=targets)
  record=dict(id='fixture',cluster='trajectory',native=[point,dict(point,t=3,reasons=['ordinary_pre','literal_action_mention'])],probes=[probe()],q=[0,.25,.5,.75,1])
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);legacy={'training_design_candidate':False};a=dict(points=1,decision=legacy,paired={'pre_k_T2_minus_T1_b1_value_pp':dict(n=1,n_trajectories=1,estimate=0.,ci95=[0.,0.])});S.C.save_json(d/'analysis.json',a);S.C.save_json(d/'manifest.json',{})
   args=SimpleNamespace(token_root=tmp,model='7b',cohort='first32')
   with patch.object(S,'require_verified',return_value='fixture'),patch.object(S.A,'load',return_value=([],[record],d,{}, {},[])):
    r=S.analyze_token(args)
   self.assertEqual(r['legacy_gate'],legacy);self.assertIsNone(r['supplement_gate']['eligible_to_consider_training'])
   self.assertTrue(r['ordinary_native']['all']['kl_target_AF']['passes'])
   self.assertGreater(r['ordinary_native']['all']['kl_target_AF']['T2_minus_T1']['estimate'],0)
   self.assertEqual(r['ordinary_native']['all']['position_count'],1)
   self.assertEqual(len(r['position_and_span_audit'][0]['native_positions']),2)
   self.assertIn('external_F_minus_M_value_pp',r['probe_sensitivity']['all']['discriminating'])
   self.assertEqual(S.C.read_json(d/'analysis.json'),a)
 def test_math_report_blinding_and_read_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'first32';qid='fixture:1'
   prefix='The intermediate number is 144.\n\n';source=prefix+'A final Student answer.'
   S.C.save_json(d/'cut_review_request.json',{'questions':[dict(id=qid,gold_for_offline_review_only='GOLD_SECRET',student={'generation':{'text':source}})]})
   S.C.save_json(d/'cut_annotations.json',{'questions':{qid:dict(prefix_end=len(prefix),cohort='error_observed')}})
   grades={};hashes={}
   for arm in ('ACTIVE','QUOTE-CONTINUE','QUOTE-REWRITE','INDEPENDENT'):
    p=d/'responses'/qid/(arm+'.json');S.C.save_json(p,{'generation':{'text':'Continuing the calculation with 144.','ids':[] if arm=='ACTIVE' else [2,3]}});hashes[str(p)]=S.C.sha_file(p)
    grades[arm]=dict(correct=False,retains_student_answer=True)
   agg=[dict(id=qid,grades=grades)];S.C.save_json(d/'aggregate_rows.json',agg)
   S.C.save_json(d/'analysis.json',dict(records_sha=hashlib.sha256(S.C.json_bytes(agg)).hexdigest(),raw_results_hashes=hashes))
   S.C.save_json(d/'verification.json',dict(passed=True,analysis_sha=S.C.sha_file(d/'analysis.json'),annotations_sha=S.C.sha_file(d/'cut_annotations.json')))
   before={str(p):S.C.sha_file(p) for p in d.rglob('*') if p.is_file()}
   r=S.analyze_math(SimpleNamespace(math_root=str(root),stage='first32',blind_labels=None))
   self.assertIsNone(r['per_question'][0]['features']['length_ratio_vs_ACTIVE'])
   self.assertEqual(r['strata']['all_descriptive']['zero_ACTIVE_token_denominators'],1)
   blind=S.blind_payload(r['blinded_review_sample']);self.assertNotIn('GOLD_SECRET',str(blind));self.assertNotIn('correctness',str(blind['samples']));self.assertNotIn('ACTIVE',str(blind))
   self.assertEqual(before,{str(p):S.C.sha_file(p) for p in d.rglob('*') if p.is_file()})
if __name__=='__main__':unittest.main()
