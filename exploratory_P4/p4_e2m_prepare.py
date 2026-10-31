"""E2-M: build memory (M) and memory+feedback (MF) judge contexts from the committed E2 requests.

Deterministic, CPU-only, reads no outcomes. For each of the 873 E2 points:
  M  = S text + '\n' + memory template + JSON(own factual history before t)
  MF = S text + '\n' + memory template + JSON(...) + (F text after the S prefix)
The history is taken from the full factual episode embedded in the committed `L` request (turns < t only:
initial observation, executed action and observation per turn; no responses, no later turns, no reward).
Writes exploratory_P4/e2m_requests.jsonl.gz and exploratory_P4/e2m_preparation.json.
"""
from pathlib import Path
import gzip, hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()


def build(rows_by_cond, memory_template):
    """rows_by_cond: {'S','L','F'} -> E2 request rows for one point. Returns the M and MF rows."""
    S, L, F = rows_by_cond['S'], rows_by_cond['L'], rows_by_cond['F']
    assert S['actions'] == L['actions'] == F['actions'] and F['text'].startswith(S['text'])
    t = int(S['turn'])
    text = L['text']
    ep = json.JSONDecoder().raw_decode(text[text.find('{"task":'):])[0]
    steps = ep['steps']
    assert all(int(st['turn']) == n + 1 for n, st in enumerate(steps))
    assert steps[t - 1]['executed_action'] == S['actions'][0]
    memory = dict(initial_observation=ep['initial_observation'],
                  turns=[dict(turn=int(st['turn']), executed_action=st['executed_action'],
                              observation_after=st['observation_after']) for st in steps[:t - 1]])
    block = '\n' + memory_template + json.dumps(memory, ensure_ascii=False)
    out = []
    for cond, txt in [('M', S['text'] + block), ('MF', S['text'] + block + F['text'][len(S['text']):])]:
        row = dict(request_id=S['point_id'] + '__' + cond, point_id=S['point_id'], traj_id=S['traj_id'], turn=t,
                   condition=cond, text=txt, actions=S['actions'])
        row['request_sha256'] = sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        out.append(row)
    return out, len(memory['turns'])


def main():
    inputs = {}

    def read(p):
        b = (ROOT / p).read_bytes(); inputs[p] = sha(b); return b

    prep = json.loads(read('exploratory_P4/e2_preparation.json'))
    raw = gzip.decompress(read('exploratory_P4/e2_requests.jsonl.gz'))
    assert sha(raw) == prep['requests_sha256']
    template = read('exploratory_P4/prompts/memory.txt').decode()
    by_point = {}
    for line in raw.decode().splitlines():
        r = json.loads(line)
        by_point.setdefault(r['point_id'], {})[r['condition']] = r
    assert len(by_point) == 873
    lines, hist_len = [], []
    for pid in sorted(by_point):
        rows, n = build(by_point[pid], template)
        hist_len.append(n)
        lines += [json.dumps(r, ensure_ascii=False) for r in rows]
    body = ('\n'.join(lines) + '\n').encode()
    (OUT / 'e2m_requests.jsonl.gz').write_bytes(gzip.compress(body, mtime=0))
    read('exploratory_P4/p4_e2m_prepare.py')
    out = dict(points=len(by_point), contexts=len(lines), conditions=['M', 'MF'], requests_sha256=sha(body),
               history_turns=dict(min=min(hist_len), max=max(hist_len), mean=sum(hist_len) / len(hist_len)),
               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), inputs=inputs)
    (OUT / 'e2m_preparation.json').write_text(json.dumps(out, indent=2) + '\n')
    print({k: v for k, v in out.items() if k != 'inputs'})


if __name__ == '__main__':
    main()
