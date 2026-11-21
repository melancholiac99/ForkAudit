import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from exploratory_shop_audit.grpo_forced_audit.common import save
from . import e2512_worker as worker
from . import e25_refresh as refresh


class NativeTargets(unittest.TestCase):
    def test_prefetch_never_crosses_a_monitor_or_terminal_gate(self):
        self.assertFalse(any(worker.can_prepare_next(k) for k in (0,5,6,10,20,90,100)))
        self.assertTrue(all(worker.can_prepare_next(k) for k in (7,9,11,89,91,99)))

    def test_native_partition_requires_matching_snapshot_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = SimpleNamespace(folder=lambda k,a: root / str(k) / a)
            tasks = list(range(8))
            path = layout.folder(7, 'P')
            record = dict(checkpoint='file', routing_sha='route', rank=0,
                          goals=tasks, rollout_sha256=['r0','r1'])
            save(path/'native/rank0/complete.json',record)
            save(layout.folder(6,'P')/'optimizer.json', dict(safetensors_sha256='file',after_sha='weight'))
            manifest = dict(native_rollout_sha256=['r0','r1'], goals=tasks, scorer_sha='weight')
            save(path/'target_parts/0/target_manifest.json',manifest)
            with patch.object(worker,'digest_file',return_value='route'), \
                 patch.object(worker,'target_shards',return_value=[[(None,g) for g in tasks]]), \
                 patch(__package__+'.d_targets.build') as build:
                worker.native_targets(layout,7,'P',0)
                self.assertEqual(build.call_args.kwargs['rollout_root'],path/'native/rank0')
                self.assertTrue((path/'target_parts/0/pipeline_timing.json').exists())
                save(path/'target_parts/0/target_manifest.json',dict(manifest,native_rollout_sha256=['a','b']))
                with self.assertRaises(AssertionError):
                    worker.native_targets(layout,7,'P',0)


class RefreshReuse(unittest.TestCase):
    def test_keep_rows_classes_and_separate_privileged_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            split=dict(T_opd=[dict(goal_index=1,asin='a')],T_pre=[],T_cal=[],V=[],T_opd_eval=[])
            save(root/'asin_split_manifest.json',split)
            views=dict(none=[1],full=[2],placeholder=[3])
            def row(views,label):
                return dict(goal=1,na={},response=[4],views=views,classes=[label])
            first=row(views,'verb')
            import json
            records=[[first,first],[row(views,'syntax'),row(dict(views,placeholder=[7]),'other')]]
            for i,rows in enumerate(records):
                (root/f'rollout_{i}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            def features(model,views,answer):
                return {'h':torch.tensor(views['placeholder'])}
            with patch.object(refresh,'PARENT',root),patch.object(refresh,'actor',return_value=torch.nn.Identity()), \
                 patch.object(refresh,'source_dir',return_value=root),patch.object(refresh,'state_hash',return_value='actor'), \
                 patch(__package__+'.e25_layout.Layout.snapshot'),patch(__package__+'.e2512_routing.owners',return_value={1:0}), \
                 patch.object(refresh,'refresh_features',side_effect=features) as compute:
                refresh.score(0,'formal_2512',1,1,root)
            data=torch.load(root/'features_0.pt',weights_only=True)
            self.assertEqual(compute.call_count,3)
            self.assertEqual(data['classes'],['verb','verb','syntax','other'])
            self.assertEqual(data['h'].tolist(),[3,3,3,7])
