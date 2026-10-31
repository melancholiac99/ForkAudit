"""vLLM policy wrapper for E4: identical sampling parameters to scripts/policy.py (D-013), plus optional LoRA adapters.

The engine is always created with enable_lora=True in E4 so that the base model and every adapter are sampled by the
same engine configuration (no adapter request = base weights).
"""
import common
from policy import Policy


class LoraPolicy(Policy):
    def __init__(self, model_path=common.MODEL_PATH, gpu_memory_utilization=0.85, max_model_len=6144, max_lora_rank=16):
        from vllm import LLM
        import vllm
        self.vllm_version = vllm.__version__
        self.llm = LLM(model=model_path, dtype="bfloat16", seed=0, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_prefix_caching=True, generation_config="vllm",
                       enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
        self.tok = self.llm.get_tokenizer()
        self.lora = None

    def set_adapter(self, path, int_id):
        """path=None -> base weights."""
        if path is None:
            self.lora = None
        else:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest(f"e4_{int_id}", int_id, path)

    def sample(self, prompts, seeds, n=1, logprobs=False):
        sps = [self.params(s, n=n, logprobs=logprobs) for s in seeds]
        kw = {} if self.lora is None else {"lora_request": self.lora}
        outs = self.llm.generate([self.chat_text(p) for p in prompts], sps, use_tqdm=False, **kw)
        res = []
        for o in outs:
            res.append([{"text": c.text, "token_ids": list(c.token_ids), "logprobs": None,
                         "finish_reason": c.finish_reason} for c in o.outputs])
        return res
