"""DRAFT implementation of E5 amendment 3 v2. No rewards, hidden states or future peeks.
Only pilot-independent synthetic validation is allowed until pre-registration freezes this file.
"""
import json
import re
import unicodedata

INFO = re.compile(r"<information>(.*?)</information>", re.S)
DOC = re.compile(r"(?m)(?:^|\n)Doc\s+\d+:\s*([^\n]*)")
QUESTION = re.compile(r"(?m)^Your question:\s*([^\n]*)", re.I)
COUNT = re.compile(r"already taken\s+(\d+)\s+step\(s\)", re.I)
EXECUTED = re.compile(r"(?m)^Step\s+\d+\s*:\s*<search>(.*?)</search>", re.I)
SEARCH = re.compile(r"<search>(.*?)</search>", re.I | re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
QUOTED = re.compile(r"[\x22\u201c]([^\x22\u201d]{2,100})[\x22\u201d]|[\x27\u2018]([^\x27\u2019]{2,100})[\x27\u2019]", re.S)
PROPER = re.compile(r"\b(?:[A-Z][a-z0-9]+|[A-Z]{2,})(?:[\s-]+(?:[A-Z][a-z0-9]+|[A-Z]{2,}))*\b")
STOP = set(("a an and are as at answer about born book by city country date did do does "
            "director directed find first for from get give how in is last list movie name "
            "novel of on or person search show tell the their this to was were what when "
            "where which who why with year").split())
K = 5
MAX_STEPS = 4


def norm(s):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", s).casefold(), re.UNICODE))


def titles_in_prompt(prompt):
    titles = set()
    for blob in INFO.findall(prompt):
        try:
            result = json.loads(blob).get("result", "")
        except (ValueError, TypeError, AttributeError):
            continue
        if isinstance(result, str):
            for raw in DOC.findall(result):
                title = norm(raw.strip().strip('"“”'))
                if title:
                    titles.add(title)
    return titles


def surface_entities(s):
    """Quoted phrases and capitalized spans, split at explicit generic question words."""
    out = set()
    for q in QUOTED.findall(s):
        value = norm(next((v for v in q if v), ""))
        if value and not all(w in STOP for w in value.split()):
            out.add(value)
    for match in PROPER.finditer(s):
        chunk = []
        for w in norm(match.group(0)).split() + [""]:
            if w and w not in STOP:
                chunk.append(w)
            elif chunk:
                out.add(" ".join(chunk))
                chunk = []
    return out


def query_entities(query, question):
    out = surface_entities(query)
    q = " " + norm(query) + " "
    for ent in surface_entities(question):
        if " " + ent + " " in q:
            out.add(ent)
    return out


def parse_action(action):
    a = action.strip()
    s = SEARCH.fullmatch(a)
    if s and s.group(1).strip():
        return "search", s.group(1).strip()
    b = ANSWER.fullmatch(a)
    if b and b.group(1).strip():
        return "answer", b.group(1).strip()
    return "invalid", ""


def turn_of(prompt):
    found = COUNT.search(prompt)
    if found:
        turn = int(found.group(1)) + 1
    elif QUESTION.search(prompt) and not EXECUTED.search(prompt):
        turn = 1
    else:
        raise ValueError("Turn absent in visible prompt")
    if not 1 <= turn <= MAX_STEPS:
        raise ValueError("Turn exceeds max four actions")
    return turn


def uniform(indices, k=K):
    out = [0.0] * k
    for i in indices:
        out[i] = 1.0 / len(indices)
    return out


def rule_dist(prompt, candidates, full_history=None):
    """R-window by default, or R-full if history is the complete (action, observation-after) prefix.

    Returns (distribution or None, diagnostics). No-answer at t=4 is not evaluable:
    the original candidate pool is never silently changed.
    """
    if len(candidates) != K or len({a.strip() for a in candidates}) != K:
        raise ValueError("Expected 5 distinct candidates")
    turn = turn_of(prompt)
    types = [parse_action(x) for x in candidates]
    answer_ids = [i for i, (kind, _) in enumerate(types) if kind == "answer"]
    search_ids = [i for i, (kind, _) in enumerate(types) if kind == "search"]
    seen = titles_in_prompt(prompt)
    old_queries = {norm(q) for q in EXECUTED.findall(prompt)}
    if full_history is not None:
        if len(full_history) != turn - 1:
            raise ValueError("R-full must contain factual actions only before the current step")
        for act, obs in full_history:
            kind, query = parse_action(act)
            if kind == "search":
                old_queries.add(norm(query))
            seen |= titles_in_prompt(obs)
    meta = dict(turn=turn, search_ids=search_ids, novel_ids=[], titles=len(seen))
    if turn == MAX_STEPS:
        if not answer_ids:
            meta["status"] = "RULE_UNAVAILABLE_NO_ANSWER_ARM"
            return None, meta
        meta["status"] = "ANSWER_ONLY"
        return uniform(answer_ids), meta
    m = QUESTION.search(prompt)
    if m is None:
        raise ValueError("Question absent in visible prompt")
    novel = []
    for i in search_ids:
        query = types[i][1]
        if norm(query) in old_queries:
            continue
        ents = query_entities(query, m.group(1))
        if any(not any(" " + ent + " " in " " + t + " " for t in seen) for ent in ents):
            novel.append(i)
    meta["novel_ids"] = novel
    if novel:
        meta["status"] = "NOVEL_ENTITY"
        return uniform(novel), meta
    meta["status"] = "UNIFORM_FALLBACK"
    return uniform(range(K)), meta


def coverage(rows):
    """Each row=(visible_prompt, five candidate actions). NO rewards/labels read."""
    records = [rule_dist(p, a) for p, a in rows]
    early = [(d, m) for d, m in records if m["turn"] <= 3 and d is not None]
    mid = [m for d, m in early if m["turn"] in (2, 3) and len(m["search_ids"]) >= 2]
    changed = sum(any(abs(p - .2) > 1e-12 for p in d) for d, m in early)
    between = sum(0 < len(m["novel_ids"]) < len(m["search_ids"]) for m in mid)
    return dict(early_points=len(early), changed_from_uniform=changed,
                early_coverage=changed / len(early) if early else None,
                search_mult_points=len(mid), within_search_discriminated=between,
                within_search_coverage=between / len(mid) if mid else None,
                last_step_points=sum(m["turn"] == 4 for _, m in records),
                last_step_without_answer=sum(m["turn"] == 4 and d is None for d, m in records))
