"""WebShop prompt construction and policy helpers (PREREG_E5 §2).

The template text is NOT written here: gate S0 copies verl-agent's WebShop template verbatim (pinned commit recorded in
DECISIONS) into exploratory_E5/prompts/webshop_template.txt, and checks format_available() against verl-agent's own
formatting of the same state. Placeholders filled: task_description, step_count, history_length, action_history,
current_step, current_observation, available_actions (admissible_actions is filled with the same string).
"""
import os
import random
import re

import common

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(HERE, "prompts", "webshop_template.txt")
MAX_STEPS = 15
HISTORY_LENGTH = 2


_TEMPLATE = None


def load_template():
    global _TEMPLATE
    if _TEMPLATE is None:
        with open(TEMPLATE_PATH) as f:
            _TEMPLATE = f.read().rstrip("\n")
    return _TEMPLATE


def extract_task(initial_obs):
    """'WebShop [SEP] Instruction: [SEP] <instruction> [SEP] Search' -> '<instruction>'."""
    m = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]", initial_obs, re.S)
    if not m:
        raise ValueError("instruction not found")
    return m.group(1).strip()


def format_available(adm):
    return ",\n".join(f"'{a}'" for a in adm)


def build_prompt(task, memory, current_obs, admissible, template=None):
    t = template or load_template()
    hist, n_hist = common.format_history(memory, HISTORY_LENGTH)
    av = format_available(admissible)
    return t.format(task_description=task, step_count=len(memory), history_length=n_hist, action_history=hist,
                    current_step=len(memory) + 1, current_observation=current_obs, available_actions=av,
                    admissible_actions=av)


def patch_common():
    """Make scripts/11_replay (run_async, _apply) build WebShop prompts and stop at 15 steps."""
    common.build_prompt = build_prompt
    common.MAX_STEPS = MAX_STEPS


class DryPolicy:
    """CPU stand-in: a random action among those listed in the prompt; search slot -> 'search[test query]'."""
    vllm_version = "dry"

    def sample(self, prompts, seeds, n=1, logprobs=False):
        out = []
        for p, sd in zip(prompts, seeds):
            item = r"'(?:search|click)\[[^\]]*\]'"
            blocks = re.findall(item + r"(?:,\n" + item + r")*", p)     # the available-actions block comes last
            acts = re.findall(r"'([^']*)'", blocks[-1]) if blocks else []
            rng = random.Random(sd)
            res = []
            for _ in range(n):
                a = rng.choice(acts) if acts else "search[test query]"
                a = "search[test query]" if a.startswith("search[<") else a
                res.append({"text": f"<think>dry</think>\n<action>{a}</action>", "token_ids": [], "logprobs": None,
                            "finish_reason": "stop"})
            out.append(res)
        return out

    def close(self):
        pass
