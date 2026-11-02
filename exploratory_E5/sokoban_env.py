"""Sokoban, text observations (fully observable; predicted-absence control for "information vs computation",
PREREG_E5 §13).

Semantics follow verl-agent @ 20bd331bdbc9026a5668e11362178e10ab7400c8 (env_package/sokoban: SokobanEnv on gym_sokoban,
sokoban_projection, SokobanEnvironmentManager with SimpleMemory, SOKOBAN_TEMPLATE[_NO_HIS]):
  * rooms: third_party/verl_agent_room_utils.generate_room(dim, num_steps=int(1.7*(h+w)), num_boxes, search_depth) with
    python and numpy RNGs seeded by the room seed (verl-agent set_seed). verl-agent retries a failed generation with
    abs(hash(str(seed))), which is process-randomised; here the retry seed is seed + 1_000_003 * attempt (D-061). Rooms
    are generated once at freeze and stored, so replay never regenerates.
  * dynamics: gym_sokoban push / move rules (actions 1-4 = push up / down / left / right, falling back to a move);
    projected action 0 (no <action> block, no direction word, or 'still') leaves the room unchanged.
  * observation: tiny_rgb_array rendering, cell ' # \\t' etc., rows joined by '\\n' (player on target = S).
  * projection: lower-cased; the substring between the first '<action>' and the first '</action>'; the first of
    up / down / left / right / still contained in it; valid also needs '<think>' and '</think>'.
  * memory: (observation before, action name Up / Down / Left / Right / Still), history_length 2, format
    "[Observation n: '...', Action n: '...']".
  * the episode ends when every box is on a target or after max_steps agent turns (examples: max_steps 15).
"""
import json
import multiprocessing as mp
import os
import random
import traceback
from collections import deque

import numpy as np

import envpool

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["Up", "Down", "Left", "Right"]
CODE = {"up": 1, "down": 2, "left": 3, "right": 4, "still": 0}
NAME = {0: "Still", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}
CHANGE = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
GRID = {0: " # \t", 1: " _ \t", 2: " O \t", 3: " √ \t", 4: " X \t", 5: " P \t", 6: " S \t"}
HISTORY_LENGTH = 2
# Difficulty ladder (PREREG_E5 §13.5). L1 = verl-agent defaults (ppo_trainer.yaml: 6x6, 1 box, search_depth 30;
# run_sokoban.sh: max_steps 15). L2 / L3 shorten the reverse-play search (boxes start closer to their targets).
LADDER = [dict(name="L1", dim=(6, 6), num_boxes=1, search_depth=30, max_steps=15),
          dict(name="L2", dim=(6, 6), num_boxes=1, search_depth=15, max_steps=15),
          dict(name="L3", dim=(6, 6), num_boxes=1, search_depth=6, max_steps=15)]


def generate(level, seed):
    """-> (room_fixed, room_state, generation seed actually used) as nested int lists."""
    from third_party.verl_agent_room_utils import generate_room
    h, w = level["dim"]
    for attempt in range(20):
        s = seed + 1_000_003 * attempt
        state_r, state_np = random.getstate(), np.random.get_state()
        try:
            random.seed(s)
            np.random.seed(s)
            fixed, state, _, _ = generate_room(dim=tuple(level["dim"]), num_steps=int(1.7 * (h + w)), num_boxes=level["num_boxes"],
                                               search_depth=level["search_depth"])
            return np.asarray(fixed).astype(int).tolist(), np.asarray(state).astype(int).tolist(), s
        except (RuntimeError, RuntimeWarning):
            continue
        finally:
            random.setstate(state_r)
            np.random.set_state(state_np)
    raise RuntimeError(f"room generation failed for seed {seed}")


class Room:
    """gym_sokoban dynamics on (room_fixed, room_state); codes 0 wall, 1 floor, 2 target, 3 box on target, 4 box, 5 player."""

    def __init__(self, fixed, state):
        self.fixed = np.array(fixed, dtype=int)
        self.state = np.array(state, dtype=int)
        self.player = np.argwhere(self.state == 5)[0]
        self.num_boxes = int(np.sum((self.state == 3) | (self.state == 4)))

    def copy(self):
        return Room(self.fixed, self.state)

    def _move(self, change):
        new = self.player + change
        if self.state[new[0], new[1]] in (1, 2):
            self.state[new[0], new[1]] = 5
            self.state[self.player[0], self.player[1]] = self.fixed[self.player[0], self.player[1]]
            self.player = new
            return True
        return False

    def act(self, code):
        """code 1-4 (push up / down / left / right); 0 = no-op."""
        if code == 0:
            return False
        change = np.array(CHANGE[(code - 1) % 4])
        new = self.player + change
        box = new + change
        if box[0] >= self.state.shape[0] or box[1] >= self.state.shape[1]:
            return False
        if self.state[new[0], new[1]] in (3, 4) and self.state[box[0], box[1]] in (1, 2):
            self.state[new[0], new[1]] = 5
            self.state[self.player[0], self.player[1]] = self.fixed[self.player[0], self.player[1]]
            self.player = new
            self.state[box[0], box[1]] = 3 if self.fixed[box[0], box[1]] == 2 else 4
            return True
        return self._move(change)

    def boxes_on_target(self):
        empty = self.state == 2
        hiding = (self.fixed == 2) & (self.state == 5)
        return self.num_boxes - int(np.sum(empty | hiding))

    def solved(self):
        return self.boxes_on_target() == self.num_boxes

    def render(self):
        st = np.where((self.state == 5) & (self.fixed == 2), 6, self.state)
        return "\n".join("".join(GRID[int(c)] for c in row) for row in st)

    def key(self):
        return self.state.tobytes()


def solve_distances(room, budget):
    """BFS over room states up to `budget` moves. -> distance (moves) to a solved state from `room`, or None."""
    start = room.copy()
    if start.solved():
        return 0
    seen, q = {start.key()}, deque([(start, 0)])
    while q:
        r, d = q.popleft()
        if d >= budget:
            continue
        for code in (1, 2, 3, 4):
            n = r.copy()
            if not n.act(code):
                continue
            k = n.key()
            if k in seen:
                continue
            if n.solved():
                return d + 1
            seen.add(k)
            q.append((n, d + 1))
    return None


def optimal_arms(room, arms, budget):
    """Planner rule (PREREG_E5 §13.3 K-0): arm a is optimal iff the room is solvable within the budget and the state
    after a is solvable in exactly one move fewer."""
    d0 = solve_distances(room, budget)
    out = []
    for a in arms:
        if d0 is None or a not in ACTIONS:
            out.append(False)
            continue
        n = room.copy()
        if not n.act(CODE[a.lower()]):
            out.append(False)
            continue
        out.append(solve_distances(n, budget - 1) == d0 - 1)
    return out


# ----------------------------------------------------------------------------- agent output (sokoban_projection)

def parse(text):
    """-> (action name, valid, fmt). The action name is what verl-agent stores in memory (ACTION_LOOKUP)."""
    s = text.lower()
    i, j = s.find("<action>"), s.find("</action>")
    if i == -1 or j == -1:
        return "Still", False, "none"
    inner = s[i + len("<action>"):j].strip().lower()
    code, valid = 0, False
    for k in ("up", "down", "left", "right", "still"):
        if k in inner:
            code, valid = CODE[k], True
            break
    if text.find("<think>") == -1 or text.find("</think>") == -1:
        valid = False
    return NAME[code], valid, "xml" if valid else "unparsed"


def is_admissible(action, adm):
    return action in adm


def display(action):
    return action


def _load(name):
    with open(os.path.join(HERE, "prompts", name)) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i] == ""):
        i += 1
    return "\n".join(lines[i:]).rstrip("\n")


_T = {}


def build_prompt(task, memory, current_obs, admissible):
    import common
    if not _T:
        _T["no_his"], _T["his"] = _load("sokoban_template_no_his.txt"), _load("sokoban_template.txt")
    if not memory:
        return _T["no_his"].format(current_observation=current_obs)
    hist, n = common.format_history(memory, HISTORY_LENGTH)
    return _T["his"].format(step_count=len(memory), history_length=n, action_history=hist, current_step=len(memory) + 1,
                            current_observation=current_obs)


class Game:
    def __init__(self, task, level):
        self.room = Room(task["fixed"], task["state"])
        self.max_steps = level["max_steps"]
        self.n, self.done, self.won = 0, False, False

    def initial_obs(self):
        return self.room.render()

    def admissible(self):
        return [] if self.done else list(ACTIONS)

    def step(self, action):
        assert not self.done
        self.room.act(CODE.get(action.lower(), 0))
        self.n += 1
        self.won = self.room.solved()
        self.done = self.won or self.n >= self.max_steps
        return self.room.render(), self.admissible(), self.won, self.done


TASKS, LEVEL = {}, {}


def load_tasks(path):
    t = json.load(open(path))
    TASKS.clear()
    LEVEL.clear()
    LEVEL.update(t["level"])
    for x in t["tasks"]:
        TASKS[str(x["seed"])] = x


def _worker(conn):
    games = {}
    while True:
        cmd, payload = conn.recv()
        if cmd == "quit":
            break
        out = {}
        try:
            def open_(key, tid, prefix, expected):
                g = Game(TASKS[str(tid)], LEVEL)
                obs, div = g.initial_obs(), 0
                for i, a in enumerate(prefix):
                    obs, _, _, _ = g.step(a)
                    div += expected is not None and obs != expected[i]
                games[key] = g
                return obs, g.admissible(), g.won, int(div)
            if cmd == "open":
                for key, tid, prefix, expected in payload:
                    out[key] = open_(key, tid, prefix, expected)
            elif cmd == "step":
                for key, action in payload:
                    out[key] = games[key].step(action)
            elif cmd == "open_step":
                for key, tid, prefix, expected, first in payload:
                    out[key] = (open_(key, tid, prefix, expected), games[key].step(first))
            elif cmd == "close":
                for key in payload:
                    games.pop(key, None)
            conn.send(("ok", out))
        except Exception:
            conn.send(("err", traceback.format_exc()))


class SokobanPool(envpool.EnvPool):
    def __init__(self, n_workers=8):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.owner = {}
        self._rr = 0


class DryPolicy:
    vllm_version = "dry"

    def sample(self, prompts, seeds, n=1, logprobs=False):
        out = []
        for sd in seeds:
            rng = random.Random(sd)
            out.append([{"text": f"<think>dry</think>\n<action>{rng.choice(['up', 'down', 'left', 'right'])}</action>", "token_ids": [],
                         "logprobs": None, "finish_reason": "stop"} for _ in range(n)])
        return out

    def close(self):
        pass
