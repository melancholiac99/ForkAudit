"""Teacher-forced scoring constructions shared by the vLLM and HF backends (CLAUDE.md §4; FEATURES_FROZEN.md §2).

Token construction (CWGT `score_implicit.py`: "the separately tokenized action delimiters are masked out"):
    ids = tok(context + "<action>") ++ tok(a) ++ tok("</action>")
Only the tok(a) span (the command tokens) is scored. `context` is the chat-templated user turn with the generation
prompt, optionally followed by an assistant-side prefix (empty think, realized think).
Log-probabilities are raw model log-probabilities (temperature 1).
"""
import json
import os

import common

VARIANTS = ("bare", "notask", "emptythink", "realized", "hindsight")
EMPTY_THINK = "<think></think>\n"
TASK_PREFIX = "Your task is to: "


def detask(prompt, task):
    """Remove every statement of the task goal: the template line and the copy inside observations
    (initial observation at turn 1 and in the action history)."""
    line = f"{TASK_PREFIX}{task}."
    lines = prompt.split("\n")
    assert lines[1] == line, "template task line is not line 2"
    p = "\n".join(lines[:1] + lines[2:])
    p = p.replace(f"\n\n{TASK_PREFIX}{task}", "").replace(f"{TASK_PREFIX}{task}", "")
    assert TASK_PREFIX not in p, "task statement left in de-tasked prompt"
    return p


def hindsight_line(won, final_obs):
    return f"Final outcome: {'SUCCESS' if won else 'FAILURE'}. Final observation: {final_obs[:500]}"


def realized_prefix(response):
    """Realized response up to (excluding) the opening tag of the parsed action span; None if not locatable."""
    k = response.rfind("</think>")
    if k == -1:
        return None
    tail = response[k + len("</think>"):]
    m = list(common._TAGGED.finditer(tail)) or list(common._UNCLOSED.finditer(tail))
    if not m:
        return None
    return response[:k + len("</think>") + m[-1].start()]


class Builder:
    """Builds token ids + command-token spans; tokenizer = HF tokenizer of the policy."""

    def __init__(self, tok):
        self.tok = tok
        self.open_ids = None

    def chat(self, user_text):
        return self.tok.apply_chat_template(common.chat_messages(user_text), tokenize=False, add_generation_prompt=True)

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def build(self, context, action):
        """-> (ids, start, end): command tokens are ids[start:end]."""
        pre = self.enc(context + "<action>")
        cmd = self.enc(action)
        post = self.enc("</action>")
        return pre + cmd + post, len(pre), len(pre) + len(cmd)

    def context(self, variant, prompt, task=None, response=None, won=None, final_obs=None):
        if variant == "bare":
            return self.chat(prompt)
        if variant == "notask":
            return self.chat(detask(prompt, task))
        if variant == "emptythink":
            return self.chat(prompt) + EMPTY_THINK
        if variant == "realized":
            rp = realized_prefix(response)
            return None if rp is None else self.chat(prompt) + rp
        if variant == "hindsight":
            return self.chat(prompt + "\n" + hindsight_line(won, final_obs))
        raise ValueError(variant)


def candidate_set(admissible, chosen):
    """𝒜_t := admissible ∪ {chosen} \\ {help}, admissible order, chosen appended if not admissible (D-021)."""
    c = [a for a in admissible if a != "help"]
    if chosen not in c:
        c.append(chosen)
    return c


def iter_points(seeds=None):
    """Every rollout step (all 3,818 points; candidate features need t-1 and t+1..t+5 neighbours)."""
    for p in sorted(os.listdir(os.path.join(common.ROOT, "data", "rollout"))):
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", p)))
        if seeds is not None and tr["seed"] not in seeds:
            continue
        final_obs = tr["steps"][-1]["obs_after"]
        for s in tr["steps"]:
            prompt = open(os.path.join(common.ROOT, s["prompt_path"])).read()
            yield {"traj_id": tr["traj_id"], "turn": s["turn"], "task": tr["task"], "prompt": prompt,
                   "chosen": s["action"], "admissible": s["admissible"], "response": s["response"],
                   "won": tr["won"], "final_obs": final_obs}
