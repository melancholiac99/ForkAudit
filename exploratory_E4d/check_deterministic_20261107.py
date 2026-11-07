"""Regression checks for isolated deterministic continuation routing."""
import sys, types, unittest
from unittest.mock import patch
import check_resume_6000d_20261107 as C
import run_deterministic_20261107 as N


class IsolatedExecution(unittest.TestCase):
    def test_data_root_and_frozen_first_actions(self):
        self.assertNotEqual(N.E.DATA, N.H/'data')
        for name in ('candidates.json','selected.json','lock.json','first_manifest.json'):
            self.assertEqual((N.E.DATA/name).read_bytes(), (N.H/'data'/name).read_bytes())
        for p in (N.H/'data/first').glob('*.json'):
            self.assertEqual(p.read_bytes(), (N.E.DATA/'first'/p.name).read_bytes())
        # The inventory contains design inputs only, never old formal outcomes.
        if (N.R/'preserved_manifest.json').exists():
            manifest=N.E.load(N.R/'preserved_manifest.json')
            self.assertFalse(any(str(N.DATA.relative_to(N.ROOT))+'/results/' in p for p in manifest))

    def test_engine_initialization_stays_after_original_pool_boundary(self):
        from e4_policy import LoraPolicy
        calls=[]
        class FakeEngine:
            def __init__(self, **kwargs): calls.append(kwargs)
            def get_tokenizer(self): return None
        fake=types.SimpleNamespace(LLM=FakeEngine,__version__='cpu-mock')
        with patch.object(LoraPolicy,'__init__',LoraPolicy.__init__), patch.dict(sys.modules, vllm=fake):
            N.install_engine_configuration()
            self.assertEqual(calls, [])
            LoraPolicy()
            self.assertEqual(len(calls),1)
            for k,v in N.ENGINE_CHANGES.items(): self.assertEqual(calls[0][k],v)
            self.assertEqual(calls[0]['dtype'],'bfloat16')
            self.assertTrue(calls[0]['enable_lora'])
            self.assertIs(fake.LLM,FakeEngine)


if __name__=='__main__':
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([loader.loadTestsFromModule(C.T),loader.loadTestsFromModule(C.X),
        loader.loadTestsFromTestCase(C.ResumeProvenance),loader.loadTestsFromTestCase(IsolatedExecution)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
