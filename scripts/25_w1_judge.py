"""FINAL RULING 2026-10-26 §3.2 features 32-33 (W1-J): the frozen policy model judges each peek observation as
{progress, neutral, regress}. Fixed prompt prompts/w1_judge_template.txt, temperature 0, chat template, vLLM bf16.

GATE: the prompt must be approved by the researcher before any call. The script refuses to run unless
logs/W1J_APPROVED exists and contains the sha256 of the template file (so any later edit of the prompt voids the
approval). Points: D (bank-1 cf_defined), factual + 4 alternative peeks (data/peek_alts, D-029).
Output data/w1_judge.parquet: traj_id, turn, role, alt_index, action, raw, label (progress|neutral|regress|unparsed).
"""
import json
import os
import re
import sys
import time

import pandas as pd

import common

TEMPLATE = os.path.join(common.ROOT, "prompts", "w1_judge_template.txt")
APPROVAL = os.path.join(common.ROOT, "logs", "W1J_APPROVED")
LABELS = ("progress", "neutral", "regress")


def parse(text):
    m = re.search(r"[a-z]+", text.lower())
    return m.group(0) if m and m.group(0) in LABELS else "unparsed"


def main():
    sha = common.file_sha256(TEMPLATE)
    if not os.path.exists(APPROVAL) or sha not in open(APPROVAL).read():
        sys.exit(f"W1-J prompt not approved: write the template sha256 {sha} into {APPROVAL} after researcher review")
    tpl = open(TEMPLATE).read()
    jobs = []
    for p in sorted(os.listdir(os.path.join(common.ROOT, "data", "rollout"))):
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", p)))
        for stp in tr["steps"]:
            b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", tr["traj_id"], f"t{stp['turn']:02d}.json")))
            if not b1["cf_defined"]:
                continue
            pk = json.load(open(os.path.join(common.ROOT, "data", "peek_alts", tr["traj_id"], f"t{stp['turn']:02d}.json")))
            for q in pk["peeks"]:
                jobs.append({"traj_id": tr["traj_id"], "turn": stp["turn"], "role": q["role"],
                             "alt_index": -1 if q["alt_index"] is None else int(q["alt_index"]), "action": q["action"],
                             "text": tpl.format(task=tr["task"], obs_before=stp["obs_before"], action=q["action"],
                                                peek_obs=q["obs"])})
    print(f"{len(jobs)} judge calls", flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    llm = LLM(model=common.MODEL_PATH, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=6144,
              enable_prefix_caching=True, generation_config="vllm")
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    t0 = time.time()
    prompts = [tok.apply_chat_template(common.chat_messages(j["text"]), tokenize=False, add_generation_prompt=True)
               for j in jobs]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    rows = [{k: j[k] for k in ("traj_id", "turn", "role", "alt_index", "action")}
            | {"raw": o.outputs[0].text, "label": parse(o.outputs[0].text)} for j, o in zip(jobs, outs)]
    df = pd.DataFrame(rows)
    out = os.path.join(common.ROOT, "data", "w1_judge.parquet")
    df.to_parquet(out)
    common.write_json(out.replace(".parquet", ".provenance.json"),
                      {"provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))},
                       "template_sha256": sha, "n_calls": len(df), "label_counts": df.label.value_counts().to_dict(),
                       "seconds": time.time() - t0})
    print("label counts", df.label.value_counts().to_dict(), f"{time.time() - t0:.0f}s", flush=True)
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as ex:  # noqa: BLE001
        print("engine shutdown warning:", ex)


if __name__ == "__main__":
    main()
