import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import resume_devices_20261108 as M


class DeviceProvenance(unittest.TestCase):
    def test_checkpoint_and_new_records_use_distinct_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prev=root/'previous';run=root/'current';prev.mkdir();run.mkdir()
            M.E.save_json(prev/'plan.json',dict(gpus=['old0','old1']))
            M.E.save_json(run/'plan.json',dict(gpus=['new0','new1']))
            (root/'e4d.py').write_text('unchanged science')
            path=root/'record.json';path.write_text('{}');manifest={'record.json':M.E.sha_file(path)}
            point={'point_id':'s3000_t02'};shard=int(M.D.digest('E4d-gpu-shard:'+point['point_id']),16)%2
            def rec(gpu,plan):return {'provenance':{'execution':dict(gpu_uuid=gpu,
                execution_plan_sha256=M.E.sha_file(plan),source_sha256=M.E.sha_file(root/'e4d.py'))}}
            with patch.multiple(M,ROOT=root,H=root,PREV=prev,R=run):
                old=rec('old'+str(shard),prev/'plan.json');new=rec('new'+str(shard),run/'plan.json');plan=dict(gpus=['new0','new1'])
                M.execution_valid(old,path,point,plan,manifest)
                M.execution_valid(new,path,point,plan,{})
                with self.assertRaises(AssertionError):M.execution_valid(old,path,point,plan,{})
                with self.assertRaises(AssertionError):M.execution_valid(new,path,point,plan,manifest)
                wrong=rec('new'+str(1-shard),run/'plan.json')
                with self.assertRaises(AssertionError):M.execution_valid(wrong,path,point,plan,{})
                path.write_text('{"modified":true}')
                with self.assertRaises(AssertionError):M.execution_valid(old,path,point,plan,manifest)

    def test_configuration_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            prev=Path(td);old=dict(environment={'VLLM_BATCH_INVARIANT':'1'})
            M.E.save_json(prev/'plan.json',old)
            new=dict(previous_deterministic_plan_sha256=M.E.sha_file(prev/'plan.json'),environment={'VLLM_BATCH_INVARIANT':'0'})
            with patch.object(M,'PREV',prev),patch.object(M,'_checked_plan',return_value=new):
                with self.assertRaises(AssertionError):M.checked_plan()


if __name__=='__main__':unittest.main()
