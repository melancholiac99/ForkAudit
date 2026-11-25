"""Paired, question-level post hoc analysis of the long thinking B and R runs."""
import json
from pathlib import Path
import numpy as np
from launch import read, sha, save

root = Path('/mnt/data2/ForkAudit/math-opsd')
dataset = root/'math500/test.jsonl'
rows = [json.loads(line) for line in dataset.read_text().splitlines()]
problems = [r['problem'] for r in rows]
assert len(set(problems)) == len(rows) == 500
levels = np.array([r['level'] for r in rows])
scores = []
inputs = {}
for arm in ('B','R@100'):
    folder = root/'run/math500-long-seed1001-n2'/('thinking-'+arm)
    receipt = read(folder/'receipt.json')
    inputs[arm] = sha(folder/'answers.json')
    assert inputs[arm] == receipt['answers_sha256'] and sha(dataset) == receipt['data_sha256']
    result = read(folder/'answers.json')
    assert result['enable_thinking'] and result['seed'] == 1001 and result['max_new_tokens'] == 32768
    assert [r['problem'] for r in result['results']] == problems
    assert all(len(r['generations']) == 2 for r in result['results'])
    scores.append(np.array([sum(g['correct'] for g in r['generations']) for r in result['results']]))
b,r = scores
delta = (r-b)*50
rng = np.random.default_rng(20261124)
groups = {}
draws = {}
for name,mask in [(str(i),levels==i) for i in range(1,6)] + [('1–4',levels<5)]:
    d = delta[mask]
    draws[name] = rng.choice(d,(10000,len(d)),replace=True).mean(axis=1)
    groups[name] = dict(questions=len(d), B_correct=int(b[mask].sum()), R_correct=int(r[mask].sum()),
                        delta_pp=d.mean(), ci95=np.percentile(draws[name],[2.5,97.5]).tolist(),
                        improved=int((d>0).sum()), worsened=int((d<0).sum()))
contrast = dict(delta_pp=delta[levels==5].mean()-delta[levels<5].mean(),
                ci95=np.percentile(draws['5']-draws['1–4'],[2.5,97.5]).tolist())
report=dict(data_sha256=sha(dataset),answers_sha256=inputs, bootstrap_seed=20261124,
            paired_question_draws=10000,groups=groups, level5_minus_levels1_to4=contrast)
save(Path(__file__).resolve().parent/'evidence/math500_long/difficulty_thinking.json',report)
print(json.dumps(report,ensure_ascii=False))
