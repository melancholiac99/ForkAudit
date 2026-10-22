"""Shared prompt construction, action parsing and provenance helpers (DECISIONS D-009).

Every script that builds an agent prompt or parses an agent output imports this module,
so rollout, replay and teacher-forced scoring see byte-identical prompts.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(ROOT, "prompts", "hcapo_alfworld_template.txt")
MODEL_PATH = os.environ.get("M0_MODEL", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")

HISTORY_LENGTH = 2
MAX_STEPS = 50
MAX_RESPONSE_TOKENS = 512
TEMPERATURE = 0.7


def load_template():
    """HCAPO App. C.2.1 template as bundled by CWGT, minus the leading '#' provenance comment lines."""
    with open(TEMPLATE_PATH) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i] == ""):
        i += 1
    return "\n".join(lines[i:]).rstrip("\n")


TEMPLATE = load_template()


def extract_task(initial_obs):
    k = initial_obs.find("Your task is to: ")
    if k == -1:
        raise ValueError("Task description not found in initial observation.")
    return initial_obs[k + len("Your task is to: "):].strip()


def format_admissible(admissible):
    return "\n ".join(f"'{s}'" for s in admissible if s != "help")


def format_history(memory, history_length=HISTORY_LENGTH):
    """memory: list of {'text_obs', 'action'}; verl-agent SimpleMemory.fetch format."""
    recent = memory[-history_length:] if history_length > 0 else []
    start = len(memory) - len(recent)
    lines = [f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
             for j, r in enumerate(recent)]
    return "\n".join(lines), len(recent)


def build_prompt(task, memory, current_obs, admissible, template=TEMPLATE):
    hist, n_hist = format_history(memory)
    return template.format(
        task_description=task,
        step_count=len(memory),
        history_length=n_hist,
        action_history=hist,
        current_step=len(memory) + 1,
        current_observation=current_obs,
        admissible_actions=format_admissible(admissible),
    )


def build_prompt_notask(task, memory, current_obs, admissible):
    """Arm-B 'de-tasked' prompt: the task-goal sentence is removed; everything else identical.
    (Exact construction to be frozen in FEATURES_FROZEN.md before Phase 2.)"""
    t = TEMPLATE.replace("Your task is to: {task_description}.\n", "")
    assert t != TEMPLATE
    return build_prompt(task, memory, current_obs, admissible, template=t)


def chat_messages(prompt):
    return [{"role": "user", "content": prompt}]


_CJK = re.compile(r"[一-鿿]")
# Tag-variant-tolerant action extraction (DECISIONS D-014). CWGT's harness parser handled the same
# `<action>` / `[action]` / `[action>` drift (KS1_XFAM_PREREG §5, A3B_PREREG "Normalization").
_OPEN = r"[<\[]\s*action\s*[>\]]"
_CLOSE = r"[<\[]\s*/\s*action\s*[>\]]"
_TAGGED = re.compile(_OPEN + r"(.*?)" + _CLOSE, re.S | re.I)
_UNCLOSED = re.compile(_OPEN + r"([^\n<\[]*)", re.I)


def parse_action(text):
    """Returns (action_str_sent_to_env, fmt). fmt in {"xml", "variant", "unclosed", "none"}.
    Only the text after the last </think> is searched (so tags quoted inside the reasoning are ignored);
    if there is no </think>, the whole text is searched. The LAST tagged span wins.
    Unparseable output -> the last 30 chars, lowercased (verl-agent behaviour; env answers "Nothing happens.")."""
    k = text.rfind("</think>")
    tail = text[k + len("</think>"):] if k != -1 else text
    m = list(_TAGGED.finditer(tail))
    if m:
        raw = m[-1].group(0)
        fmt = "xml" if raw.lower().startswith("<action>") and raw.lower().endswith("</action>") else "variant"
        return " ".join(m[-1].group(1).strip().lower().split()), fmt
    m = list(_UNCLOSED.finditer(tail))
    if m and m[-1].group(1).strip():
        return " ".join(m[-1].group(1).strip().lower().split()), "unclosed"
    return text.lower()[-30:], "none"


def is_valid(fmt, text):
    """Valid = an action was extracted, <think>...</think> present, no CJK (verl-agent criteria, tag-tolerant)."""
    return fmt != "none" and "<think>" in text and "</think>" in text and not _CJK.search(text)


def parse(text):
    """-> (action, valid, fmt)"""
    a, fmt = parse_action(text)
    return a, is_valid(fmt, text), fmt


ACTION_TYPES = ["goto", "take", "put", "open", "close", "toggle", "use", "examine/look", "inventory", "other"]


def action_type(a):
    a = a.strip().lower()
    for pat, t in [(r"^go to ", "goto"), (r"^take ", "take"), (r"^(put|move) ", "put"), (r"^open ", "open"),
                   (r"^close ", "close"), (r"^toggle ", "toggle"), (r"^use ", "use"),
                   (r"^(examine|look)\b", "examine/look"), (r"^inventory\b", "inventory")]:
        if re.match(pat, a):
            return t
    return "other"


# ---------------------------------------------------------------- provenance

def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def git_dirty():
    try:
        return bool(subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--", "scripts"], text=True).strip())
    except Exception:
        return True


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def provenance(script_file, inputs=()):
    return {
        "script": os.path.relpath(os.path.abspath(script_file), ROOT),
        "commit": git_commit(),
        "scripts_dirty": git_dirty(),
        "argv": sys.argv,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {os.path.relpath(p, ROOT): file_sha256(p) for p in inputs},
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
