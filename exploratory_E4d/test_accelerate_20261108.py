import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import accelerate_20261108 as A

class ProvenanceTests(unittest.TestCase):
    def test_frozen_parent_new_and_transition_records(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'new';prev=root/'prev';run.mkdir();prev.mkdir()
            (root/'e4d.py').write_text('frozen science')
            for folder in [run,prev]:A.E.save_json(folder/'plan.json',{'gpus':['gpu0','gpu1'],'revision':folder.name})
            p={'point_id':next(str(i) for i in range(100) if int(A.D.digest('E4d-gpu-shard:'+str(i)),16)%2==0)}
            path=A.D.run_path(root,'Azero','01',p['point_id'],0);path.parent.mkdir(parents=True);path.write_text('{}')
            rel=str(path.relative_to(root));frozen={rel:A.E.sha_file(path)}
            plan={'gpus':['gpu0','gpu1'],'parent_plans':{str(prev/'plan.json'):A.E.sha_file(prev/'plan.json')}}
            def rec(folder,gpu='gpu0'):
                return {'continuations':[{'rep':0}],'provenance':{'execution':{'gpu_uuid':gpu,'execution_plan_sha256':A.E.sha_file(folder/'plan.json'),'source_sha256':A.E.sha_file(root/'e4d.py')}}}
            with patch.multiple(A,ROOT=root,H=root,R=run,PREV=prev),patch.object(A.E,'DATA',root):
                A.execution_valid(rec(prev),path,p,plan,frozen)
                A.execution_valid(rec(run),path,p,plan,{})
                with self.assertRaises(AssertionError):A.execution_valid(rec(prev),path,p,plan,{})
                with self.assertRaises(AssertionError):A.execution_valid(rec(run),path,p,plan,frozen)
                with self.assertRaises(AssertionError):A.execution_valid(rec(run,'gpu1'),path,p,plan,{})
                A.E.save_json(run/'transition_tail_manifest.json',frozen)
                A.execution_valid(rec(prev),path,p,plan,{})
                with self.assertRaises(AssertionError):A.execution_valid(rec(run),path,p,plan,{})
                path.write_text('modified')
                with self.assertRaises(AssertionError):A.execution_valid(rec(prev),path,p,plan,frozen)

    def test_termination_requires_same_pid_identity_and_paused_state(self):
        receipt={'processes':{'42':{'start':'old'}}}
        with patch.object(A,'proc',return_value={'start':'new','state':'T'}),patch.object(A.os,'kill') as kill:
            A.terminate_scheduler(42,receipt);kill.assert_not_called()
        with patch.object(A,'proc',return_value={'start':'old','state':'S'}),patch.object(A.os,'kill') as kill:
            with self.assertRaises(AssertionError):A.terminate_scheduler(42,receipt)
            kill.assert_not_called()

if __name__=='__main__':unittest.main()
