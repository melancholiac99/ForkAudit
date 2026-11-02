"""CPU-only tests of resume integrity, arithmetic and experimental contrasts."""
import json
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from pathlib import Path
import numpy as np
import execution_checks as X
import exp1 as E
import verify_exp1 as V


class Tests(unittest.TestCase):
    def rows(self):
        jobs=[];rows=[]
        for c in E.CONDS:
            j=dict(request_id='p__'+c,point_id='p',condition=c,request_sha256=c)
            jobs.append(j);rows.append(dict(j,candidate_prob=[.2]*5,sequence_logp=[0.]*5,argmax=0,hardware_sha256='fp',session='one'))
        return jobs,rows

    def test_invalid_scores(self):
        jobs,rows=self.rows();X.validate_row(rows[0],jobs[0],'fp')
        for change in (dict(hardware_sha256='other'),dict(sequence_logp=[1.,0,0,0,0]),dict(sequence_logp=[float('nan')]*5),dict(argmax=3)):
            with self.assertRaises(AssertionError):X.validate_row(dict(rows[0],**change),jobs[0],'fp')

    def test_whole_point_resume(self):
        jobs,rows=self.rows()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scores.jsonl'
            def put(items,tail='\n'):path.write_text('\n'.join(json.dumps(r) for r in items)+tail)
            put(rows);self.assertEqual(len(X.load_scores(path,jobs,'fp',True)),10)
            for bad in (rows[:-1],rows+[rows[0]],[dict(r,session='two') if i==0 else r for i,r in enumerate(rows)]):
                put(bad)
                with self.assertRaises(AssertionError):X.load_scores(path,jobs,'fp',True)
            put(rows,tail='')
            with self.assertRaises(AssertionError):X.load_scores(path,jobs,'fp')

    def test_common_bootstrap(self):
        v=[.2,-.7,.1,1.2];t=['a','a','b','c']
        V.compare(E.mean_ci(v,t),V.statistic(v,t))
        V.compare(E.mean_ci([],[]),V.statistic([],[]))

    def test_identity_formula_and_saturation(self):
        e=np.array([30.,1,0,0,0]);x=np.array([28.,3,0,0,0]);factual=0;alternative=1
        effect=.5*((e[factual]-e[alternative])-(x[factual]-x[alternative]))
        pe=np.exp(e-e.max());pe/=pe.sum();px=np.exp(x-x.max());px/=px.sum()
        probability=.5*(pe[0]-px[0]+px[1]-pe[1])
        self.assertEqual(effect,2.);self.assertGreater(probability,0);self.assertLess(probability,1e-8)
        self.assertEqual(effect+(.5*((x[0]-x[1])-(e[0]-e[1]))),0.)

    def test_protocol_seed_used_in_analysis(self):
        import inspect
        source=inspect.getsource(E.analyze)
        self.assertIn('seed=SEED)',source);self.assertNotIn('SEED+',source)
        self.assertIn('gold_q12_factual_worse_ge_1_3',source)

    def test_full_analysis_synthetic(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';target=out/'32b';target.mkdir(parents=True)
            jobs=[];meta=[];saved=[];table={k:[] for k in ('traj_id','turn','j','q_12','q_3456','q_gold')}
            for i,gap in enumerate((1/3,1/6,0.)):
                pid=f's{i:03d}_t01';tid=f's{i:03d}'
                meta.append(dict(point_id=pid,traj=tid,layer='I' if i==0 else 'U',swap_arm=1,q12_gap=gap))
                for j in range(5):
                    for k,value in dict(traj_id=tid,turn=1,j=j,q_12=gap if j==1 else 0.,
                                        q_3456=.5 if j==1 else 0.,q_gold=.8 if j==1 else 0.).items():table[k].append(value)
                for c in E.CONDS:
                    job=dict(request_id=pid+'__'+c,point_id=pid,condition=c,text='fixture',actions=['a','b','c','d','e'])
                    job['request_sha256']=E.sha(json.dumps(job,sort_keys=True,ensure_ascii=False).encode());jobs.append(job)
                    lp=np.zeros(5);lp[0]=1 if c.endswith('E') else 0;lp[1]=1 if c.endswith('X') else 0
                    p=np.exp(lp);p/=p.sum()
                    saved.append(dict(request_id=job['request_id'],point_id=pid,condition=c,request_sha256=job['request_sha256'],
                                      sequence_logp=lp.tolist(),candidate_prob=p.tolist(),argmax=int(p.argmax()),hardware_sha256='fp',session='one'))
            import gzip
            raw=('\n'.join(json.dumps(j) for j in jobs)+'\n').encode()
            (out/'requests.jsonl.gz').write_bytes(gzip.compress(raw));E.write_json(out/'pointmeta.json',meta)
            arms=root/'arms';arms.write_text('fixture')
            E.write_json(out/'manifest.json',dict(contexts=30,requests_sha256=E.sha(raw),pointmeta_sha256=E.digest(out/'pointmeta.json'),
                         input_sha256={'exploratory_deep_discovery/exploratory_arms.parquet':E.digest(arms)}))
            (target/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in saved))
            E.write_json(target/'run.json',dict(complete=True,contexts=30,requests_sha256=E.sha(raw),scores_sha256=E.digest(target/'scores.jsonl'),
                         scorer_script_sha256=E.digest(E.__file__),checks_script_sha256=E.digest(X.__file__),
                         protocol_sha256=E.digest(E.HERE/'EXPERIMENT1_PROTOCOL.md'),hardware_sha256='fp'))
            check_ids=[next(j['request_id'] for j in jobs if j['condition']==c) for c in E.CONDS]+[jobs[0]['request_id']]
            E.write_json(target/'sessions/one.json',dict(hardware_sha256='fp',checks=[dict(request_id=k,tv=0.,passed=True,
                         shared_sequence_logp=[0]*5,direct_sequence_logp=[0]*5) for k in check_ids]))
            with patch.object(E,'NPOINTS',3),patch.object(E,'read_parquet',return_value=table),contextlib.redirect_stdout(io.StringIO()):
                E.analyze(Namespace(repo=root,out=out,arms=arms,name='32b'))
            result=E.load_json(target/'analysis.json')
            self.assertEqual(result['metrics']['F/all']['identity_switch_logodds']['n'],3)
            self.assertEqual(result['metrics']['F/gold_q12_factual_worse']['identity_switch_logodds']['n'],2)
            self.assertEqual(result['metrics']['F/gold_q12_factual_worse_ge_1_3']['identity_switch_logodds']['n'],1)
            self.assertTrue(result['decisions']['identity_logodds_positive'])
            self.assertTrue(result['decisions']['value_harm_vs_no_label'])


if __name__=='__main__':unittest.main()
