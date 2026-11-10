"""CPU-only ALFWorld state tracking and per-task milestone extraction.

All inferred events are based on agent-visible feedback, not hidden simulator state.
Milestones in failed trajectories are conservative evidence, not oracle truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Keep task parsing self-contained; aligns with E4's task-word convention.
TARGET_PATS = (
    r"^(?:clean|heat|cool) (?:a |an |some |two |the )?(\w+) and put",
    r"^find (?:a |an |some |two |the )?(\w+) and put",
    r"^put (?:a |an |some |two |the )?(?:clean |hot |cool |cold )?(\w+) (?:in|on|into)\b",
    r"^(?:look at|examine) (?:a |an |some |two |the )?(\w+) (?:under|with)\b",
)
OBJECT = r"[a-z][a-z0-9_-]*(?:\s+\d+)?"
PICK = re.compile(rf"\byou pick up (?:the |a |an )?({OBJECT})\b", re.I)
RELEASE = re.compile(rf"\byou (?:move|put|place) (?:the |a |an )?({OBJECT})\b", re.I)
HELD_TEXT = re.compile(rf"\byou are (?:carrying|holding) (?:the |a |an )?({OBJECT})\b", re.I)
NEGATIVE = re.compile(r"\b(?:nothing happens|can't|cannot|not possible|fail to|unable to|don't have)\b", re.I)
PROCESS_VERBS = {"clean": "clean", "heat": "heat", "cool": "cool"}


def normalize(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def target_word(task: str) -> Optional[str]:
    text = normalize(task).rstrip(".")
    for p in TARGET_PATS:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return None


def ids_in(text: str, word: str) -> set[str]:
    if not word:
        return set()
    return {f"{word} {n}" for n in re.findall(rf"\b{re.escape(word)}\s+(\d+)\b", text or "", re.I)}


def is_target(obj: Optional[str], word: Optional[str]) -> bool:
    return bool(obj and word and re.match(rf"^{re.escape(word)}(?:\s+\d+)?$", normalize(obj)))


def task_kind(task_type: str) -> str:
    t = normalize(task_type)
    if "pick_two_obj" in t:
        return "two"
    if "look_at_obj" in t:
        return "look"
    for k in PROCESS_VERBS:
        if f"pick_{k}_then" in t:
            return k
    if "pick_and_place" in t:
        return "place"
    return "unknown"


@dataclass
class CarryState:
    """Tracks only feedback-confirmed hand occupancy; unknown is not empty."""
    item: Optional[str] = None
    known: bool = False

    def update(self, obs: str) -> None:
        obs = obs or ""
        pick = PICK.search(obs)
        if pick and not NEGATIVE.search(obs):
            self.item = normalize(pick.group(1))
            self.known = True
            return
        release = RELEASE.search(obs)
        if release and not NEGATIVE.search(obs):
            if self.item is None or normalize(release.group(1)) == self.item:
                self.item = None
                self.known = True
            return
        hold = HELD_TEXT.search(obs)
        if hold:
            self.item = normalize(hold.group(1))
            self.known = True
        if re.search(r"\b(?:your hands are empty|you are not carrying anything)\b", obs, re.I):
            self.item, self.known = None, True

    def description(self) -> str:
        if not self.known:
            return "Carrying status: unknown from observed feedback."
        if self.item:
            return f"Carrying status: currently holding {self.item}."
        return "Carrying status: hands empty."


def put_candidates(admissible: list[str], held: Optional[str]) -> list[str]:
    """Only already-admissible put actions mentioning the exact held object."""
    if not held:
        return []
    h = normalize(held)
    return sorted(a for a in admissible
                  if re.match(r"^(?:put|move)\s+", normalize(a))
                  and re.match(rf"^(?:put|move)\s+(?:the\s+)?{re.escape(h)}\s+(?:in|on|into|to)\b", normalize(a)))


def forced_release_action(held: CarryState, task: str, task_type: str, admissible: list[str]) -> Optional[str]:
    """Observation-history intervention; never synthesizes an illegal command."""
    w = target_word(task)
    if task_kind(task_type) == "unknown" or not held.known or not held.item or not w or is_target(held.item, w):
        return None
    candidates = put_candidates(admissible, held.item)
    return candidates[0] if candidates else None


def milestones(ep: dict) -> dict:
    """Monotonic reachability milestones. Stage evidence is a lower bound for failures.

    For pick-two, 'found' and 'picked' require two different target instances.
    A confirmed win implies all required prerequisites, even if feedback did not name them.
    """
    task, kind = ep.get("task", ""), task_kind(ep.get("task_type", ""))
    word = target_word(task)
    if kind == "unknown" or not word:
        return {"recognized": False, "kind": kind, "target": word, "stages": {}, "evidence": {}}
    required = 2 if kind == "two" else 1
    visible = ids_in(ep.get("initial_obs", ""), word)
    picked: set[str] = set()
    processed: set[str] = set()
    for step in ep.get("steps", []):
        obs = step.get("obs_after", "") or ""
        visible.update(ids_in(obs, word))
        p = PICK.search(obs)
        if p and is_target(p.group(1), word) and not NEGATIVE.search(obs):
            obj = normalize(p.group(1))
            picked.add(obj)
            visible.add(obj)
        if kind in PROCESS_VERBS and not NEGATIVE.search(obs):
            verb = PROCESS_VERBS[kind]
            if re.search(rf"\byou\s+{verb}\b", obs, re.I):
                # Credit a process event only when feedback confirms the target instance.
                processed.update(ids_in(obs, word))
    won = bool(ep.get("won", False))
    done = bool(ep.get("steps", []) and ep["steps"][-1].get("won_after", False))
    if done:
        won = True
    found = len(visible) >= required or len(picked) >= required or len(processed) >= required or won
    picked_ok = len(picked) >= required or len(processed) >= required or won
    steps = ["found", "picked"]
    reached = {"found": found, "picked": picked_ok}
    if kind in PROCESS_VERBS:
        steps.append("processed")
        reached["processed"] = bool(processed) or won
    steps.append("completed")
    reached["completed"] = won
    if not found and (picked_ok or bool(processed)):
        reached["found"] = True
    return {"recognized": True, "kind": kind, "target": word, "required_instances": required,
            "stages": {k: reached[k] for k in steps},
            "evidence": {"visible_ids": sorted(visible), "picked_ids": sorted(picked),
                         "processed_ids": sorted(processed), "won": won}}
