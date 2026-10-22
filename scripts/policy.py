"""vLLM policy wrapper shared by rollout and replay (DECISIONS D-013: pure temperature sampling)."""
import common


class Policy:
    def __init__(self, model_path=common.MODEL_PATH, gpu_memory_utilization=0.85, max_model_len=6144):
        from vllm import LLM
        import vllm
        self.vllm_version = vllm.__version__
        self.llm = LLM(model=model_path, dtype="bfloat16", seed=0, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_prefix_caching=True, generation_config="vllm")
        self.tok = self.llm.get_tokenizer()

    def chat_text(self, prompt):
        return self.tok.apply_chat_template(common.chat_messages(prompt), tokenize=False, add_generation_prompt=True)

    def sample(self, prompts, seeds, n=1, logprobs=True):
        """prompts: list[str] (filled template), seeds: list[int]. Returns per prompt a list of n dicts
        {text, token_ids, logprobs(list[float]) }."""
        from vllm import SamplingParams
        sps = [SamplingParams(n=n, temperature=common.TEMPERATURE, top_p=1.0, top_k=-1, min_p=0.0,
                              repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                              max_tokens=common.MAX_RESPONSE_TOKENS, seed=s, logprobs=0 if logprobs else None)
               for s in seeds]
        outs = self.llm.generate([self.chat_text(p) for p in prompts], sps, use_tqdm=False)
        res = []
        for o in outs:
            cs = []
            for c in o.outputs:
                lp = None
                if logprobs and c.logprobs is not None:
                    lp = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs)]
                cs.append({"text": c.text, "token_ids": list(c.token_ids), "logprobs": lp,
                           "finish_reason": c.finish_reason})
            res.append(cs)
        return res

    def close(self):
        """Shut the engine core down explicitly; otherwise the EngineCore child can linger holding GPU memory."""
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001
            print("engine shutdown warning:", e)
        del self.llm
        import gc
        gc.collect()
