"""GPU ceiling for replay generation: steps/s when vLLM gets N real stored prompts at once (continuous batching)."""
import glob, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
from vllm import LLM, SamplingParams
P = sorted(glob.glob(os.path.join(common.ROOT, "prompts", "s0*", "t*.txt"))); random.seed(0)
llm = LLM(model=common.MODEL_PATH, dtype="bfloat16", seed=0, gpu_memory_utilization=0.90, max_model_len=6144,
          enable_prefix_caching=True, generation_config="vllm", max_num_seqs=int(os.environ.get("MNS", "1024")),
          max_num_batched_tokens=int(os.environ.get("MNBT", "16384")))
tok = llm.get_tokenizer()
for N, seeded in [(1500, False)]:
    ps = [tok.apply_chat_template(common.chat_messages(open(p).read()), tokenize=False, add_generation_prompt=True)
          for p in random.sample(P, N)]
    sps = [SamplingParams(temperature=0.7, top_p=1.0, top_k=-1, max_tokens=512, seed=(i if seeded else None)) for i in range(N)]
    t = time.time(); out = llm.generate(ps, sps, use_tqdm=False); dt = time.time() - t
    ntok = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"N={N} seeded={seeded} MNS={os.environ.get('MNS','1024')}: {dt:.1f}s -> {N/dt:.1f} steps/s, {ntok/dt:.0f} out tok/s", flush=True)
