import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import balance_20261109 as B
from register_balance_20261109 import assign

class BalanceTests(unittest.TestCase):
    def test_assignment_preserves_jobs_and_balances_frozen_costs(self):
        jobs=[dict(key=str(i),estimated_seconds=t) for i,t in enumerate([9,7,5,3,1])]
        a,loads=assign(jobs,[4,1]);self.assertEqual(set(a),{j['key'] for j in jobs})
        self.assertEqual(sum(loads),30);self.assertLessEqual(abs(loads[0]-loads[1]),1)
        self.assertEqual(assign(list(reversed(jobs)),[4,1]),(a,loads))

    def test_new_record_requires_registered_physical_card(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r=root/'run';r.mkdir();(root/'e4d.py').write_text('frozen')
            B.E.save_json(r/'plan.json',{'new':True});old=root/'old.json';B.E.save_json(old,{'gpus':['g0','g1']})
            point={'point_id':next(str(i) for i in range(100) if B.shard({'point_id':str(i)})==0)}
            path=B.D.run_path(root,'Dgap','11',point['point_id'],0);path.parent.mkdir(parents=True);path.write_text('{}')
            def record(sha,gpu):return {'provenance':{'execution':dict(execution_plan_sha256=sha,gpu_uuid=gpu,source_sha256=B.E.sha_file(root/'e4d.py'))}}
            plan={'gpus':['g0','g1'],'jobs':{'formal:Dgap:11:0':{'gpu':1}},'parent_plans':{str(old):B.E.sha_file(old)}}
            with patch.multiple(B,ROOT=root,H=root,R=r),patch.object(B.E,'DATA',root):
                good=record(B.E.sha_file(r/'plan.json'),'g1');B.execution_valid(good,path,point,plan,{})
                with self.assertRaises(AssertionError):B.execution_valid(record(B.E.sha_file(r/'plan.json'),'g0'),path,point,plan,{})
                previous=record(B.E.sha_file(old),'g0');manifest={str(path.relative_to(root)):B.E.sha_file(path)}
                B.execution_valid(previous,path,point,plan,manifest)
                with self.assertRaises(AssertionError):B.execution_valid(previous,path,point,plan,{})
                with self.assertRaises(AssertionError):B.execution_valid(good,path,point,plan,manifest)
                path.write_text('modified')
                with self.assertRaises(AssertionError):B.execution_valid(previous,path,point,plan,manifest)

if __name__=='__main__':unittest.main()
