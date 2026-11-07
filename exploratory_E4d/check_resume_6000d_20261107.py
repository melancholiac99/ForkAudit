"""CPU preflight with an isolated legacy lock-test fixture and resume checks."""
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
import e4d as E
import test_e4d as T
import test_server_execution as X
import resume_6000d_20261107 as M

# The original lock test patches DATA but omits FIRST_MANIFEST. Once real first
# actions exist it reads that real path and fails before exercising its fixture.
# Isolate only that test; do not edit fingerprinted science/test source or data.
original = T.E4dLockGate.test_lock_is_independent_of_pi1_and_rejects_change
def isolated_lock_test(self):
    with tempfile.TemporaryDirectory() as td, patch.object(E, 'FIRST_MANIFEST', Path(td)/'first_manifest.json'):
        original(self)
T.E4dLockGate.test_lock_is_independent_of_pi1_and_rejects_change = isolated_lock_test


class ResumeProvenance(unittest.TestCase):
    def test_old_and_new_records_require_respective_plan_and_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); old=root/'old'; run=root/'run'; old.mkdir();run.mkdir()
            E.save_json(old/'plan.json', {'gpus':['old0','old1']})
            E.save_json(run/'plan.json', {'gpus':['new0','new1']})
            (root/'e4d.py').write_text('frozen source')
            record=root/'record.json';record.write_text('{}')
            point={'point_id':'s3000_t02'}
            shard=int(M.D.digest('E4d-gpu-shard:'+point['point_id']),16)%2
            def rec(uuid, plan): return {'provenance':{'execution':dict(gpu_uuid=uuid,
                execution_plan_sha256=E.sha_file(plan),source_sha256=E.sha_file(root/'e4d.py'))}}
            with patch.multiple(M, ROOT=root, H=root, OLD=old, R=run):
                manifest={'record.json':E.sha_file(record)}
                M.execution_valid(rec('old'+str(shard),old/'plan.json'),record,point,{'gpus':['new0','new1']},manifest)
                M.execution_valid(rec('new'+str(shard),run/'plan.json'),record,point,{'gpus':['new0','new1']},{})
                with self.assertRaises(AssertionError):
                    M.execution_valid(rec('old'+str(shard),old/'plan.json'),record,point,{'gpus':['new0','new1']},{})
                with self.assertRaises(AssertionError):
                    M.execution_valid(rec('new'+str(1-shard),run/'plan.json'),record,point,{'gpus':['new0','new1']},{})
                record.write_text('{"changed":true}')
                with self.assertRaises(AssertionError):
                    M.execution_valid(rec('old'+str(shard),old/'plan.json'),record,point,{'gpus':['new0','new1']},manifest)


if __name__ == '__main__':
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([loader.loadTestsFromModule(T),loader.loadTestsFromModule(X),loader.loadTestsFromTestCase(ResumeProvenance)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
