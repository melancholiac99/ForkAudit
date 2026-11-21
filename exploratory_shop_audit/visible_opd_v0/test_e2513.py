import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .e2513_adopt import copy_state,recover_matched,verify_state
from .e2512_driver import train_arm
from exploratory_shop_audit.grpo_forced_audit.common import digest_file,save


class ResumeSignedPrefix(unittest.TestCase):
    def test_resumed_files_are_verified_and_cleanup_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'original';dst=root/'resumed'
            (src/'adapter').mkdir(parents=True)
            weight=src/'adapter/adapter_model.safetensors';weight.write_bytes(b'signed-weights')
            (src/'adam.pt').write_bytes(b'signed-moments')
            rec={'safetensors_sha256':digest_file(weight),'adam_sha256':digest_file(src/'adam.pt')}
            copy_state(src,dst,rec)
            self.assertTrue(verify_state(dst,rec))
            (dst/'adam.pt').unlink()
            self.assertTrue(verify_state(src,rec))
            with self.assertRaises(AssertionError):verify_state(src,{**rec,'adam_sha256':'wrong'})

    def test_no_matched_optimizer_replay_when_gate_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            l=SimpleNamespace(root=Path(temp))
            save(l.root/'controls_gate.json',dict(decision='NOT_RUN'))
            with patch(__package__+'.e2513_adopt.subprocess.run') as run:
                with self.assertRaises(AssertionError):recover_matched(l)
                run.assert_not_called()

    def test_first_live_job_is_update_six(self):
        with tempfile.TemporaryDirectory() as temp:
            l=SimpleNamespace(root=Path(temp),initialize=lambda *args,**kwargs:None)
            with patch(__package__+'.e2513_adopt.adopt_prefix') as adopt,\
                 patch(__package__+'.e2512_driver.launch',side_effect=RuntimeError('probe')) as launch:
                with self.assertRaisesRegex(RuntimeError,'probe'):train_arm(l,'P')
                adopt.assert_called_once_with(l,'P')
                launch.assert_called_once_with(l,'collect',6,'P')
