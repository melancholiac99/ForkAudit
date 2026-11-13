"""Compare the live LoRA/AdamW state to the actual written files."""
import torch
from safetensors.torch import load_file
from peft import get_peft_model_state_dict
from protocol_common import save, sha, tensor_hash


def verify_live_checkpoint(model, optimizer, updates, folder):
    stored = torch.load(folder / 'optimizer.pt', map_location='cpu', weights_only=True)
    assert stored['total_updates'] == updates
    assert stored['adapter_sha'] == sha(folder / 'adapter_model.safetensors')
    live = optimizer.state_dict()
    assert live['param_groups'] == stored['optimizer']['param_groups']
    moments = {}
    for key, value in live['state'].items():
        moments[str(key)] = {}
        for name, tensor in value.items():
            other = stored['optimizer']['state'][key][name]
            assert torch.equal(tensor.cpu(), other) and torch.isfinite(other).all()
            moments[str(key)][name] = tensor_hash(tensor)
        assert value['step'].item() == updates
    disk = load_file(str(folder / 'adapter_model.safetensors'))
    adapter = get_peft_model_state_dict(model)
    assert adapter.keys() == disk.keys()
    weights = {}
    for name, value in adapter.items():
        assert torch.equal(value.cpu(), disk[name]) and torch.isfinite(disk[name]).all()
        weights[name] = tensor_hash(value)
    save(folder / 'live_checkpoint_verification.json', dict(passed=True, total_updates=updates,
         live_vs_disk_weights_equal=True, live_vs_disk_moments_equal=True,
         parameter_hashes=weights, moment_hashes=moments,
         weights_sha256=sha(folder / 'adapter_model.safetensors'), optimizer_sha256=sha(folder / 'optimizer.pt')))

