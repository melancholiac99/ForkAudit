"""WebShop adapter for the M0 replay machinery (PREREG_E5 §2).

A WebShop text environment holds one browser state at a time and is expensive to instantiate, so each worker process
owns ONE environment and keeps, per caller key, only the action history. Serving a key whose state is not the one
currently loaded = reset(session) + re-feed its history (WebShop is deterministic given session and actions; gate S1).
The worker speaks the scripts/envpool.py protocol (open / step / open_step / close) with game_file = session id, so
scripts/11_replay.run_async and the E4-style loops run unchanged.

Items marked VERIFY are WebShop API details that gate S1/S3 check on the server before any rollout.
"""
import multiprocessing as mp
import os
import sys
import traceback

import envpool

WEBSHOP_ROOT = os.environ.get("WEBSHOP_ROOT", "/root/autodl-tmp/WebShop")
SEARCH_SLOT = "search[<your query>]"


def make_env():
    sys.path.insert(0, WEBSHOP_ROOT)
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # VERIFY import path
    return WebAgentTextEnv(observation_mode="text", human_goals=True)


def admissible(env):
    av = env.get_available_actions()  # VERIFY: {'has_search_bar': bool, 'clickables': [...]}
    return ([SEARCH_SLOT] if av["has_search_bar"] else []) + [f"click[{c}]" for c in av["clickables"]]


def is_admissible(action, adm):
    """Free-text search is admissible whenever the search bar is present; clicks must be listed."""
    if action in adm:
        return True
    return SEARCH_SLOT in adm and action.startswith("search[") and action.endswith("]") and len(action) > len("search[]")


class Session:
    """One environment, many logical games (keys)."""

    def __init__(self):
        self.env = make_env()
        self.current = None      # key whose state is loaded
        self.games = {}          # key -> dict(session, actions)

    def _reset(self, session):
        self.env.server.user_sessions.pop(str(session), None)   # VERIFY: fresh goal/done state for this session
        obs, _ = self.env.reset(session=int(session))
        return obs

    def _load(self, key):
        g = self.games[key]
        obs = self._reset(g["session"])
        for a in g["actions"]:
            obs, _, _, _ = self.env.step(a)
        self.current = key
        return obs

    def goal_asin(self):
        return self.env.server.user_sessions[self.env.session]["goal"]["asin"]   # VERIFY

    def open(self, key, session, prefix, expected):
        self.games[key] = dict(session=str(session), actions=[])
        obs = self._reset(session)
        div = 0
        for i, a in enumerate(prefix):
            obs, _, _, _ = self.env.step(a)
            self.games[key]["actions"].append(a)
            div += expected is not None and obs != expected[i]
        self.current = key
        return obs, admissible(self.env), False, int(div)

    def step(self, key, action):
        if self.current != key:
            self._load(key)
        obs, reward, done, info = self.env.step(action)
        self.games[key]["actions"].append(action)
        won = bool(done) and reward == 1.0           # PREREG_E5 §2: success = reward 1 at episode end
        return obs, admissible(self.env), won, bool(done)

    def close(self, key):
        self.games.pop(key, None)
        if self.current == key:
            self.current = None


def _worker(conn):
    s = Session()
    while True:
        cmd, payload = conn.recv()
        if cmd == "quit":
            break
        out = {}
        try:
            if cmd == "open":
                for key, session, prefix, expected in payload:
                    out[key] = s.open(key, session, prefix, expected)
            elif cmd == "step":
                for key, action in payload:
                    out[key] = s.step(key, action)
            elif cmd == "open_step":
                for key, session, prefix, expected, first in payload:
                    out[key] = (s.open(key, session, prefix, expected), s.step(key, first))
            elif cmd == "close":
                for key in payload:
                    s.close(key)
            elif cmd == "goal":
                for key, session in payload:
                    s._reset(session)
                    s.current = None
                    out[key] = s.goal_asin()
            conn.send(("ok", out))
        except Exception:
            conn.send(("err", traceback.format_exc()))


class WSPool(envpool.EnvPool):
    """envpool.EnvPool with WebShop workers (same API). Create before CUDA initialisation."""

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

    def goals(self, sessions):
        """{session: goal asin} using worker 0."""
        self.conns[0].send(("goal", [(str(x), x) for x in sessions]))
        status, res = self.conns[0].recv()
        if status != "ok":
            raise RuntimeError(res)
        return {int(k): v for k, v in res.items()}
