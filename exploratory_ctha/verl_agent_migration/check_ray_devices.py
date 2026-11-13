"""Verify Ray assigns distinct physical GPUs with the external worker hook."""
import json
import os
from pathlib import Path

import ray

ray.init(num_cpus=4, object_store_memory=1073741824,
         runtime_env={'worker_process_setup_hook': 'launch.setup_worker'})


@ray.remote(num_gpus=1)
class DeviceProbe:
    def inspect(self):
        import torch
        return dict(rank=os.environ.get('RANK'), visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    uuid=str(torch.cuda.get_device_properties(0).uuid))


actors = [DeviceProbe.options(runtime_env={'env_vars': {'RANK': str(i)}}).remote() for i in range(2)]
results = ray.get([actor.inspect.remote() for actor in actors])
assert len({r['uuid'] for r in results}) == 2, results
Path(os.environ['FORKAUDIT_VERL_OUTPUT'], 'ray_devices.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results))
ray.shutdown()
