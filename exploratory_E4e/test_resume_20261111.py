import ast,inspect,unittest
import four_arm_eval as R
import resume_eval_20261111 as E

class ResumeTests(unittest.TestCase):
    def test_rollout_body_is_frozen_except_declared_changes(self):
        expected=inspect.getsource(R.run).replace('old.get("provenance") != prov','not compatible_provenance(old.get("provenance"), prov)').replace('result.update(n_turns=len(x["steps"]), condition=cond, provenance=prov)','result.update(n_turns=len(x["steps"]), condition=cond, provenance=prov, hardware_resume=runtime_receipt())')
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse(inspect.getsource(E.run))))
    def test_only_commit_metadata_can_differ(self):
        a=dict(commit='old',argv=['fixed'],model='m',execution_plan_sha256='plan',source_sha256='source',adapter='a',vllm='v',condition='both')
        self.assertTrue(E.compatible_provenance(a,dict(a,commit='new')))
        for key in set(a)-{'commit'}:
            b=dict(a);b[key]='different';self.assertFalse(E.compatible_provenance(a,b),key)
        self.assertFalse(E.compatible_provenance(None,a))
if __name__=='__main__':unittest.main()
