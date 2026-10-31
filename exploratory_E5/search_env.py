"""Multi-hop retrieval agent (HotpotQA questions; wiki-18 e5 retriever) for the M0 replay machinery (PREREG_E5 §13).

Semantics follow verl-agent @ 20bd331bdbc9026a5668e11362178e10ab7400c8 (agent_system/environments/env_package/search,
its skyrl_gym SearchEnv / SearchToolGroup, SearchEnvironmentManager, SearchMemory, search_projection):
  * action projection: text after the first </search> or </answer> is cut; the first <search>..</search> block wins,
    else the first <answer>..</answer>; the action sent to the environment is '<search>q</search>' /
    '<answer>a</answer>' / '' (no block). valid = a block was found, not both tags, no repeated tag.
  * search: POST {query, topk: 3, return_scores: true} to the retriever; documents formatted 'Doc i: {contents}\\n';
    observation = '<information>' + json.dumps({'result': ...}) + '</information>'. No <search> block -> ''.
  * the episode ends at an <answer> block or after max_turns = 4 actions; reward = EM of the last <answer> span in the
    concatenated history (SQuAD normalisation); won = reward >= 1. A terminal step returns the empty observation.
  * prompt: SEARCH_TEMPLATE_NO_HIS at step 1, SEARCH_TEMPLATE afterwards; history = last 4 records
    'Step {n}:{action} {observation_after}\\n' joined by '\\n' (history_length 4 as in examples/gigpo_trainer/run_search.sh).
Deviations (DECISIONS D-061): a retrieval error aborts the run instead of being shown to the agent; every query's
documents are fixed the first time it is retrieved (SQLite cache shared by all workers, stages, and both variants;
first writer wins), so replay is exact even if the retriever is not bit-deterministic.

Two variants (same questions, same agent, same budget):
  open : the task description is the question                                      (E5-S, effect predicted)
  ctx  : the task description also carries the ten HotpotQA context paragraphs (two gold, eight distractors) in
         the retriever's own document format, so the deciding documents are in the prompt from step 1
                                                                                   (E5-C, effect predicted absent)
The environment is stateless apart from the action history; a logical game is (question id, actions so far).
"""
import json
import multiprocessing as mp
import os
import re
import string
import traceback

import envpool

HERE = os.path.dirname(os.path.abspath(__file__))
VERL_AGENT_COMMIT = "20bd331bdbc9026a5668e11362178e10ab7400c8"
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
TOPK = 3
MAX_STEPS = 4
HISTORY_LENGTH = 4
ADM = ["<search>", "<answer>"]          # nominal: the prompt lists no admissible actions
_RETRIEVE = None                        # tests replace this with a fake retriever
CACHE_PATH = os.environ.get("SEARCH_CACHE", os.path.join(HERE, "data", "retrieval_cache.sqlite"))   # None: no cache
_DB = {}


# ----------------------------------------------------------------------------- retrieval and formatting

def retrieve(query):
    """-> list of document dicts ({'contents': ...}) for one query (topk = 3)."""
    if _RETRIEVE is not None:
        return _RETRIEVE(query)
    import urllib.request
    req = urllib.request.Request(RETRIEVER_URL, data=json.dumps(dict(query=query, topk=TOPK, return_scores=True)).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]
    return [d["document"] for d in res[0]]


def init_cache():
    """Create the cache (WAL mode) in the parent process, before the workers are forked."""
    if CACHE_PATH is None:
        return
    import sqlite3
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    db = sqlite3.connect(CACHE_PATH, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS c (q TEXT PRIMARY KEY, docs TEXT NOT NULL)")
    db.commit()
    db.close()


def _db():
    import sqlite3
    pid = os.getpid()
    if pid not in _DB:                  # one connection per process (workers are forked)
        if not os.path.exists(CACHE_PATH):
            init_cache()
        _DB[pid] = sqlite3.connect(CACHE_PATH, timeout=120)
    return _DB[pid]


def cached_retrieve(query):
    if CACHE_PATH is None:
        return retrieve(query)
    db = _db()
    row = db.execute("SELECT docs FROM c WHERE q = ?", (query,)).fetchone()
    if row is None:
        docs = [dict(contents=d["contents"]) for d in retrieve(query)]
        db.execute("INSERT OR IGNORE INTO c VALUES (?, ?)", (query, json.dumps(docs)))
        db.commit()
        row = db.execute("SELECT docs FROM c WHERE q = ?", (query,)).fetchone()
    return json.loads(row[0])


def passages2string(docs):
    return "".join(f"Doc {i + 1}: {d['contents'].strip()}\n" for i, d in enumerate(docs))


def information(docs):
    """Observation text for one search (verl-agent: '\\n<information>' + result_text + '</information>\\n', stripped)."""
    body = json.dumps({"result": passages2string(docs)}) if docs else json.dumps({"result": "No search results found."})
    return "<information>" + body + "</information>"


def norm_title(t):
    return " ".join(t.strip().strip('"').lower().split())


_INFO = re.compile(r"<information>(.*?)</information>", re.S)
_DOC = re.compile(r"(?:^|\n)Doc \d+: (.*)")


def titles_in(text):
    """Normalised titles of all documents inside <information> blocks of a prompt or observation."""
    out = set()
    for blob in _INFO.findall(text):
        try:
            s = json.loads(blob)["result"]
        except (ValueError, KeyError, TypeError):
            continue
        out |= {norm_title(t) for t in _DOC.findall(s)}
    return out


def context_block(context):
    """HotpotQA context [[title, [sentences]], ...] in the retriever's document format."""
    return information([dict(contents=f'"{t}"\n' + "".join(s).strip()) for t, s in context])


# ----------------------------------------------------------------------------- answers (Search-R1 qa_em)

def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):
    return any(normalize_answer(pred) == normalize_answer(g) for g in golds)


def extract_solution(s):
    m = list(re.finditer(r"<answer>(.*?)</answer>", s, re.S))
    return m[-1].group(1).strip() if m else None


# ----------------------------------------------------------------------------- agent output (search_projection)

_SB = re.compile(r"<search>(.*?)</search>", re.I | re.S)
_AB = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def parse(text):
    """-> (action sent to the environment, valid, fmt)."""
    t = text
    if "</search>" in t:
        t = t.split("</search>", 1)[0] + "</search>"
    elif "</answer>" in t:
        t = t.split("</answer>", 1)[0] + "</answer>"
    m = _SB.search(t)
    if m:
        a, fmt = f"<search>{m.group(1).strip()}</search>", "search"
    else:
        m = _AB.search(t)
        a, fmt = (f"<answer>{m.group(1).strip()}</answer>", "answer") if m else ("", "none")
    ns, na = len(re.findall(r"<search>", text, re.I)), len(re.findall(r"<answer>", text, re.I))
    valid = fmt != "none" and not (ns and na) and ns <= 1 and na <= 1
    return a, valid, fmt


def is_admissible(action, adm):
    return bool(adm) and (action.startswith("<search>") or action.startswith("<answer>"))


def display(action):
    """Judge-facing form of an action: search[q] / answer[a] / (no action)."""
    if action.startswith("<search>"):
        return f"search[{action[len('<search>'):-len('</search>')]}]"
    if action.startswith("<answer>"):
        return f"answer[{action[len('<answer>'):-len('</answer>')]}]"
    return "(no action)"


# ----------------------------------------------------------------------------- prompt

def _load(name):
    with open(os.path.join(HERE, "prompts", name)) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i] == ""):
        i += 1
    return "\n".join(lines[i:]).rstrip("\n")


_T = {}


def build_prompt(task, memory, current_obs, admissible):
    """memory: [{'text_obs': observation before the action, 'action'}] (the E5 / 11_replay convention). verl-agent
    shows (action, observation after the action); the observation after record j is memory[j+1]['text_obs'], or the
    current observation for the last record."""
    if not _T:
        _T["no_his"], _T["his"] = _load("search_template_no_his.txt"), _load("search_template.txt")
    if not memory:
        return _T["no_his"].format(task_description=task)
    after = [m["text_obs"] for m in memory[1:]] + [current_obs]
    recs = list(zip([m["action"] for m in memory], after))
    recent = recs[-HISTORY_LENGTH:]
    start = len(recs) - len(recent)
    ctx = "\n".join(f"Step {start + j + 1}:{a} {o}\n" for j, (a, o) in enumerate(recent))
    return _T["his"].format(task_description=task, memory_context=ctx, step_count=len(memory))


# ----------------------------------------------------------------------------- game logic (pure; unit-tested)

class Game:
    """One question; replay of the action list is exact (retrieval results are cached per worker)."""

    def __init__(self, q, variant):
        self.q, self.variant = q, variant
        self.turns, self.done, self.won, self.history = 0, False, False, []

    def initial_obs(self):
        if self.variant == "ctx":
            return f"{self.q['question']}\nRetrieved documents: {context_block(self.q['context'])}"
        return self.q["question"]

    def admissible(self):
        return [] if self.done else list(ADM)

    def step(self, action, cache):
        assert not self.done
        self.turns += 1
        self.history.append(action)
        done = self.turns >= MAX_STEPS or ("<answer>" in action and "</answer>" in action)
        if done:
            ans = extract_solution("".join(self.history))
            self.done, self.won = True, ans is not None and exact_match(ans, self.q["answers"])
            return "", [], self.won, True
        m = re.search(r"<search>(.*?)</search>", action, re.S) if ("<search>" in action and "</search>" in action) else None
        if m is None:
            return "", self.admissible(), False, False
        query = m.group(1).strip()
        if query not in cache:
            cache[query] = cached_retrieve(query)
        obs = information(cache[query])
        self.history.append("\n" + obs + "\n")
        return obs, self.admissible(), False, False


# ----------------------------------------------------------------------------- worker (envpool protocol)

QUESTIONS = {}


def load_questions(path):
    init_cache()
    QUESTIONS.clear()
    for q in json.load(open(path))["tasks"]:
        QUESTIONS[str(q["qid"])] = q


def _worker(conn, variant):
    games, cache = {}, {}
    while True:
        cmd, payload = conn.recv()
        if cmd == "quit":
            break
        out = {}
        try:
            def open_(key, qid, prefix, expected):
                g = Game(QUESTIONS[str(qid)], variant)
                obs, div = g.initial_obs(), 0
                for i, a in enumerate(prefix):
                    obs, _, _, _ = g.step(a, cache)
                    div += expected is not None and obs != expected[i]
                games[key] = g
                return obs, g.admissible(), g.won, int(div)
            if cmd == "open":
                for key, qid, prefix, expected in payload:
                    out[key] = open_(key, qid, prefix, expected)
            elif cmd == "step":
                for key, action in payload:
                    out[key] = games[key].step(action, cache)
            elif cmd == "open_step":
                for key, qid, prefix, expected, first in payload:
                    out[key] = (open_(key, qid, prefix, expected), games[key].step(first, cache))
            elif cmd == "close":
                for key in payload:
                    games.pop(key, None)
            conn.send(("ok", out))
        except Exception:
            conn.send(("err", traceback.format_exc()))


class SearchPool(envpool.EnvPool):
    def __init__(self, n_workers=8, variant="open"):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, variant), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.owner = {}
        self._rr = 0


# ----------------------------------------------------------------------------- strata (decisive facts = gold supporting titles)

def stratum(prompt, history_text, arm_obs, gold_titles):
    """I: a gold supporting title is absent from the prompt and some arm's one-step response retrieves an absent gold
    title not seen in the history; U: every gold title is in the prompt; N: absent titles, no arm retrieves any of them;
    other: the rest (an arm retrieves only titles already seen). -> (stratum, reveal bit per arm)."""
    gold = {norm_title(t) for t in gold_titles}
    missing = gold - titles_in(prompt)
    seen = titles_in(history_text)
    reveal = [bool((titles_in(o) & missing) - seen) for o in arm_obs]
    if not missing:
        return "U", reveal
    if any(reveal):
        return "I", reveal
    if not any(titles_in(o) & missing for o in arm_obs):
        return "N", reveal
    return "other", reveal


class DryPolicy:
    """CPU stand-in emitting verl-agent-style search / answer outputs."""
    vllm_version = "dry"

    def sample(self, prompts, seeds, n=1, logprobs=False):
        import random
        out = []
        for p, sd in zip(prompts, seeds):
            rng = random.Random(sd)
            res = []
            for _ in range(n):
                if rng.random() < 0.6:
                    t = f"<think>dry</think>\n<search> q{rng.randint(0, 5)} </search>"
                else:
                    t = f"<think>dry</think>\n<answer>a{rng.randint(0, 2)}</answer>"
                res.append({"text": t, "token_ids": [], "logprobs": None, "finish_reason": "stop"})
            out.append(res)
        return out

    def close(self):
        pass
