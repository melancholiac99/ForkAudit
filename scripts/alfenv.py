"""Single-game ALFWorld (TextWorld backend) wrapper with deterministic prefix re-feed.

Mirrors alfworld 0.4.2 AlfredTWEnv: AlfredDemangler(shuffle=False) + AlfredInfos wrappers,
request_infos won/admissible_commands/gamefile. State restore = reset + re-feed the action prefix
(TextWorld PDDL games are deterministic); `refeed` verifies observations against the stored ones.
"""
import json
import os

# Must be read before importing alfworld, which sets ALFWORLD_DATA to ~/.cache/alfworld if unset.
ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", "/root/autodl-tmp/alfworld_data")
os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA

import textworld  # noqa: E402
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, TASK_TYPES  # noqa: E402


def collect_games(split="train"):
    """Same filters as AlfredTWEnv.collect_game_files, sorted for determinism. Returns [(game_file, task_type)]."""
    sub = {"train": "train", "eval_in_distribution": "valid_seen", "eval_out_of_distribution": "valid_unseen"}[split]
    root_dir = os.path.join(ALFWORLD_DATA, "json_2.1.1", sub)
    types = set(TASK_TYPES.values())
    out = []
    for root, dirs, files in os.walk(root_dir):
        if "traj_data.json" not in files or "movable" in root or "Sliced" in root:
            continue
        gf = os.path.join(root, "game.tw-pddl")
        if not os.path.exists(gf):
            continue
        with open(os.path.join(root, "traj_data.json")) as f:
            tt = json.load(f)["task_type"]
        if tt not in types:
            continue
        with open(gf) as f:
            if not json.load(f).get("solvable", False):
                continue
        out.append((gf, tt))
    return sorted(out)


class AlfGame:
    def __init__(self, game_file):
        self.game_file = game_file
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        self.env = textworld.start(game_file, request_infos=infos,
                                   wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])

    def reset(self):
        gs = self.env.reset()
        return self._unpack(gs)

    def load(self, game_file):
        """Reuse this native-library slot for another game via the full wrapper stack."""
        if game_file != self.game_file:
            self.env.load(game_file)
            self.game_file = game_file

    def step(self, action):
        gs, score, done = self.env.step(action)
        obs, adm, won = self._unpack(gs)
        return obs, adm, won, done

    @staticmethod
    def _unpack(gs):
        return gs.feedback, list(gs["admissible_commands"]), bool(gs["won"])

    def refeed(self, actions, expected_obs=None):
        """Reset and replay `actions`. Returns (obs, admissible, won, n_divergences).
        expected_obs[i] is the observation stored after actions[i] in the original run."""
        obs, adm, won = self.reset()
        div = 0
        for i, a in enumerate(actions):
            obs, adm, won, done = self.step(a)
            if expected_obs is not None and obs != expected_obs[i]:
                div += 1
        return obs, adm, won, div

    def close(self):
        self.env.close()
