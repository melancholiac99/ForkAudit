"""Environment adapters for the E5 pipeline (PREREG_E5 §13). Each adapter supplies everything environment-specific;
e5_run.py is environment-agnostic. Environments: search (open retrieval), search_ctx (same questions, context given),
sokoban, webshop (deferred).
"""
import json
import os

import numpy as np

import common

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = {2: "two", 3: "three", 4: "four", 5: "five"}


class Adapter:
    name = None
    K_ALT = 4
    MAX_STEPS = None
    N_MAIN = None
    presence = False
    draft_alternatives = True

    def tasks_path(self, pilot=False, level=None):
        return os.path.join(HERE, f"tasks_{self.task_set}{'_pilot' if pilot else ''}.json")

    def prepare_workers(self, tasks_path):
        pass

    def patch_common(self):
        common.build_prompt = self.build_prompt
        common.parse = self.parse
        common.MAX_STEPS = self.MAX_STEPS

    def fixed_alternatives(self, factual):
        return None

    def display(self, action):
        return action

    def conf_split(self, action):
        """-> (opening tag, command) for the -conf signal (command tokens only, teacher-forced after the tag)."""
        return "<action>", action

    def judge_template(self, kind):
        if kind == "base":
            return open(os.path.join(HERE, "prompts", f"judge_base_{self.judge_name}.txt")).read()
        t = open(os.path.join(HERE, "prompts", f"judge_{kind}.txt")).read()
        return t.replace("five labeled", f"{WORD[self.K_ALT + 1]} labeled") if kind == "feedback" else t


class SearchAdapter(Adapter):
    task_set, judge_name = "search", "search"
    N_MAIN = 1000

    def __init__(self, variant):
        import search_env as S
        self.S, self.variant = S, variant
        self.name = "search" if variant == "open" else "search_ctx"
        self.presence = variant == "open"
        self.MAX_STEPS = S.MAX_STEPS
        self.build_prompt, self.parse, self.is_admissible, self.display = S.build_prompt, S.parse, S.is_admissible, S.display

    def conf_split(self, action):
        for tag in ("search", "answer"):
            if action.startswith(f"<{tag}>"):
                return f"<{tag}>", action[len(tag) + 2:-(len(tag) + 3)]
        return None

    def prepare_workers(self, tasks_path):
        self.S.load_questions(tasks_path)

    def pool(self, n):
        return self.S.SearchPool(n, self.variant)

    def dry_policy(self):
        return self.S.DryPolicy()

    def game_id(self, task):
        return str(task["qid"])

    def task_text(self, task, initial_obs):
        return initial_obs          # verl-agent: the task description is the reset observation

    def decisive(self, tr, i, peeks):
        hist = "\n".join([tr["initial_obs"]] + [s["obs_after"] for s in tr["steps"][:i]])
        z, reveal = self.S.stratum(tr["steps"][i]["prompt"], hist, [p["obs"] for p in peeks], tr["meta"]["supporting_titles"])
        return z, reveal, reveal

    def freeze(self, args):
        """HotpotQA dev, distractor setting (HOTPOT_DEV; answers, supporting facts, context). Forward-only filters: answer
        not yes / no; both supporting titles present in wiki-18 (WIKI18 = wiki-18.jsonl). Disjoint seeded samples: main
        1000 (seeds 6000-6999), pilot 50 (seeds 8000-8049)."""
        assert not os.path.exists(self.tasks_path()), "already frozen"
        dev = json.load(open(os.environ["HOTPOT_DEV"]))
        wiki_titles = set()
        with open(os.environ["WIKI18"]) as f:
            for line in f:
                c = json.loads(line)["contents"]
                wiki_titles.add(self.S.norm_title(c.partition("\n")[0]))
        keep = [q for q in dev if q["answer"].strip().lower() not in ("yes", "no")
                and all(self.S.norm_title(t) in wiki_titles for t in {t for t, _ in q["supporting_facts"]})]
        idx = np.random.default_rng(20261031).permutation(len(keep))
        rec = lambda q, s: dict(seed=s, qid=q["_id"], question=q["question"], answers=[q["answer"]], type=q["type"], level=q["level"],
                                supporting_titles=sorted({t for t, _ in q["supporting_facts"]}), context=q["context"])
        main = [rec(keep[k], 6000 + n) for n, k in enumerate(idx[:self.N_MAIN])]
        pilot = [rec(keep[k], 8000 + n) for n, k in enumerate(idx[self.N_MAIN:self.N_MAIN + 50])]
        meta = dict(n_dev=len(dev), n_eligible=len(keep), seed=20261031, filters="answer not yes/no; supporting titles in wiki-18")
        json.dump(dict(**meta, tasks=main), open(self.tasks_path(), "w"), indent=1)
        json.dump(dict(**meta, tasks=pilot), open(self.tasks_path(pilot=True), "w"), indent=1)
        print(meta, len(main), len(pilot))


class SokobanAdapter(Adapter):
    name = task_set = judge_name = "sokoban"
    K_ALT = 3
    N_MAIN = 150
    draft_alternatives = False

    def __init__(self):
        import sokoban_env as K
        self.K = K
        self.build_prompt, self.parse, self.is_admissible, self.display = K.build_prompt, K.parse, K.is_admissible, K.display

    def tasks_path(self, pilot=False, level=None):
        if pilot:
            assert level, "--level is required for the Sokoban pilot"
            return os.path.join(HERE, f"tasks_sokoban_pilot_{level}.json")
        return os.path.join(HERE, "tasks_sokoban.json")

    def conf_split(self, action):
        return "<action>", action.lower()

    def prepare_workers(self, tasks_path):
        self.K.load_tasks(tasks_path)
        self.MAX_STEPS = self.K.LEVEL["max_steps"]

    def pool(self, n):
        return self.K.SokobanPool(n)

    def dry_policy(self):
        return self.K.DryPolicy()

    def game_id(self, task):
        return str(task["seed"])

    def task_text(self, task, initial_obs):
        return "push every box onto a target"          # the Sokoban template has no task slot; shown to judges only

    def fixed_alternatives(self, factual):
        return [a for a in self.K.ACTIONS if a != factual] if factual in self.K.ACTIONS else None

    def decisive(self, tr, i, peeks):
        """Fully observable: every deciding fact is in the prompt (stratum U, no reveal). Rule = planner (BFS on the
        prompt's own board, remaining budget): arms that start a shortest solution."""
        room = self.K.Room(tr["meta"]["fixed"], tr["meta"]["state"])
        for s in tr["steps"][:i]:
            room.act(self.K.CODE.get(s["action"].lower(), 0))
        budget = tr["meta"]["max_steps"] - i
        return "U", [False] * len(peeks), self.K.optimal_arms(room, [p["action"] for p in peeks], budget)

    def judge_template(self, kind):
        t = super().judge_template(kind)
        return t.replace("{max_steps}", str(self.K.LEVEL["max_steps"])) if kind == "base" else t

    def freeze(self, args):
        """--pilot --level Lk: 50 pilot rooms of that level (seeds 7500 + 100 * ladder index + n). Without --pilot: 150
        main rooms (seeds 7000-7149) at --level, which must be the level chosen by the pilot rule (§13.5)."""
        level = next(l for l in self.K.LADDER if l["name"] == args.level)
        path = self.tasks_path(pilot=args.pilot, level=args.level)
        assert not os.path.exists(path), "already frozen"
        base = 7500 + 100 * self.K.LADDER.index(level) if args.pilot else 7000
        n = 50 if args.pilot else self.N_MAIN
        tasks = []
        for k in range(n):
            fixed, state, gen_seed = self.K.generate(level, base + k)
            tasks.append(dict(seed=base + k, gen_seed=gen_seed, fixed=fixed, state=state, max_steps=level["max_steps"],
                              opt_len=self.K.solve_distances(self.K.Room(fixed, state), level["max_steps"])))
        json.dump(dict(level=level, tasks=tasks), open(path, "w"), indent=1)
        print(path, len(tasks), "solvable within budget:", sum(t["opt_len"] is not None for t in tasks))


class WebShopAdapter(Adapter):
    name = task_set = judge_name = "webshop"
    N_MAIN = 150
    presence = True

    def __init__(self):
        import ws_common as W
        import ws_env
        self.W, self.E = W, ws_env
        self.MAX_STEPS = W.MAX_STEPS
        self.build_prompt, self.is_admissible = W.build_prompt, ws_env.is_admissible
        self.parse = common.parse

    def tasks_path(self, pilot=False, level=None):
        return os.path.join(HERE, "tasks.json")

    def pool(self, n):
        return self.E.WSPool(n)

    def dry_policy(self):
        return self.W.DryPolicy()

    def game_id(self, task):
        return str(task["session"])

    def task_text(self, task, initial_obs):
        return self.W.extract_task(initial_obs)

    def decisive(self, tr, i, peeks):
        import e5_run as X
        s = tr["steps"]
        hist = "\n".join([tr["initial_obs"]] + [x["obs_before"] + "\n" + "\n".join(x["admissible"]) for x in s[:i + 1]])
        asin = tr["meta"]["goal_asin"]
        z = X.stratum(s[i]["prompt"], hist, [p["obs"] for p in peeks], asin)
        reveal = [asin.lower() in p["obs"].lower() for p in peeks]
        return z, reveal, reveal

    def freeze(self, args):
        import e5_run as X
        X.stage_freeze_webshop(args)


def get(name):
    return {"search": lambda: SearchAdapter("open"), "search_ctx": lambda: SearchAdapter("ctx"),
            "sokoban": SokobanAdapter, "webshop": WebShopAdapter}[name]()
