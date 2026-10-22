"""Phase 0.2 smoke test: vLLM loads Qwen2.5-7B-Instruct, samples 32 drafts at task-0 step 1, seed reproducibility."""
import collections
import json
import os
import time

import alfenv
import common
from policy import Policy

tl = json.load(open(os.path.join(common.ROOT, "data", "task_list.json")))["tasks"]
g = alfenv.AlfGame(os.path.join(alfenv.ALFWORLD_DATA, tl[0]["game_file"]))
obs, adm, won = g.reset()
p = common.build_prompt(common.extract_task(obs), [], obs, adm)
t0 = time.time()
pol = Policy()
print("vllm", pol.vllm_version, "load_s", round(time.time() - t0, 1))
print("prompt_tokens", len(pol.tok(pol.chat_text(p))["input_ids"]))
print("chat_text_head", repr(pol.chat_text(p)[:160]))
t0 = time.time()
o = pol.sample([p], [1], n=32, logprobs=True)[0]
print("32_samples_s", round(time.time() - t0, 2))
acts = [common.parse(c["text"])[:2] for c in o]
print("admissible_counts", collections.Counter(a for a, v in acts if v and a in adm).most_common(8))
print("invalid", sum(not v for a, v in acts), "valid_nonadmissible", sum(v and a not in adm for a, v in acts))
print("mean_resp_tokens", sum(len(c["token_ids"]) for c in o) / 32)
print("example_response:\n" + o[0]["text"])
o2 = pol.sample([p], [1], n=32, logprobs=False)[0]
print("seed_reproducible", [c["text"] for c in o2] == [c["text"] for c in o])
pol.close()
