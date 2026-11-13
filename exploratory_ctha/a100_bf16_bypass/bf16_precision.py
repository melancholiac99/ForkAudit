"""BF16 backbone and FP32 head/log-softmax; no autocast over the head."""
import importlib.metadata
import os


def check_environment():
    assert importlib.metadata.version('vllm') == '0.32.0'
    assert os.environ['VLLM_BATCH_INVARIANT'] == '1'
    assert os.environ['NVIDIA_TF32_OVERRIDE'] == '0'


def setup_head(model):
    head = model.get_output_embeddings()
    head.float()
    head.register_forward_pre_hook(lambda module, args: (args[0].float(), *args[1:]))


def selected_logprobs(model, row, temperature):
    import torch
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    assert model.get_output_embeddings().weight.dtype == torch.float32
    pi, ri = row['prompt_ids'], row['completion_ids']
    device = model.get_input_embeddings().weight.device
    with torch.autocast('cuda', enabled=False):
        logits = model(input_ids=torch.tensor([pi + ri], device=device), use_cache=False).logits
        assert logits.dtype == torch.float32
        window = logits[0, len(pi)-1:len(pi)+len(ri)-1]
        return (window / temperature).log_softmax(-1).gather(-1, torch.tensor(ri, device=device)[:, None]).squeeze(-1)
