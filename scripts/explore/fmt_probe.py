"""Exploratory (not a result): which action-tag formats does Qwen2.5-7B emit under the HCAPO template?"""
import collections, json, os, re, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alfenv, common

def chat(prompt, n, seed):
    body = {"model": "qwen", "messages": common.chat_messages(prompt), "n": n, "seed": seed, "temperature": 0.7,
            "top_p": 1.0, "top_k": -1, "max_tokens": 512}
    req = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return [c["message"]["content"] for c in json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req))["choices"]]

def fmt(t):
    if re.search(r"<action>.*?</action>", t, re.S): return "xml"
    if re.search(r"\[action\].*?\[/action\]", t, re.S | re.I): return "bracket"
    if re.search(r"<action>", t): return "xml_unclosed"
    return "other"

tl = json.load(open(os.path.join(common.ROOT, "data", "task_list.json")))["tasks"]
tot = collections.Counter()
for x in tl[:10]:
    g = alfenv.AlfGame(os.path.join(alfenv.ALFWORLD_DATA, x["game_file"])); obs, adm, _ = g.reset()
    outs = chat(common.build_prompt(common.extract_task(obs), [], obs, adm), 32, x["seed"])
    c = collections.Counter()
    for o in outs:
        a, valid, f = common.parse(o)
        c[f] += 1; c["valid"] += valid; c["valid_admissible"] += valid and a in adm
        if f == "none": print("UNPARSED:", repr(o[-200:]))
    tot += c; print(x["seed"], dict(c))
print("TOTAL", dict(tot), "valid_rate", tot["valid"] / 320, "admissible_rate", tot["valid_admissible"] / 320)
