"""CPU-only orchestration checks; never downloads, starts a GPU or shuts down."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_with_gigpo as W

class ExtensionTests(unittest.TestCase):
    def test_original_success_requires_gigpo_before_shutdown(self):
        order=[]
        with tempfile.TemporaryDirectory() as d, patch.object(W,'H',Path(d)), patch.object(W,'G',Path(d)), \
             patch.object(W,'check_extension'), patch.object(W,'prepare_model',side_effect=lambda:order.append('model')), \
             patch.object(W.C,'state'), patch.object(W.C,'invoke',side_effect=lambda key,args:order.append(('eval',args))), \
             patch.object(W,'verify_and_analyze',side_effect=lambda:order.append('verify')), \
             patch('download_gigpo.verify',side_effect=lambda:order.append('weights')), \
             patch.object(W.C,'publish',side_effect=lambda:order.append('publish') or {'verified':True}), \
             patch.object(W,'_original_notify'), \
             patch.object(W,'_original_shutdown',side_effect=lambda why:order.append(('shutdown',why))):
            W.finish_with_gigpo('available_E4e_scope_verified_uploaded')
        self.assertEqual([x[0] if isinstance(x,tuple) else x for x in order],['model','eval','verify','weights','publish','shutdown'])
        args=order[1][1]
        self.assertNotIn('--adapter',args)
        for flag,value in [('--conditions','base'),('--reps','3'),('--workers','20'),('--policy-tag','gigpo7b')]:
            self.assertEqual(args[args.index(flag)+1],value)

    def test_failure_cannot_reach_success_shutdown(self):
        with patch.object(W,'check_extension'), patch.object(W,'prepare_model',side_effect=RuntimeError('bad checkpoint')), \
             patch.object(W,'_original_shutdown') as stop, patch.object(W.C,'publish') as publish:
            with self.assertRaises(RuntimeError):W.finish_with_gigpo('available_E4e_scope_verified_uploaded')
            stop.assert_not_called();publish.assert_not_called()

    def test_registered_terminal_failure_uses_original_preservation_path(self):
        with patch.object(W,'_original_shutdown') as stop, patch.object(W,'prepare_model') as model:
            W.finish_with_gigpo('unrecoverable_saved')
            stop.assert_called_once_with('unrecoverable_saved');model.assert_not_called()

if __name__=='__main__':unittest.main()
