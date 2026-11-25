"""Summarize archived MATH-500 samples without running inference."""
import csv
import hashlib
import json
import re
from pathlib import Path

ARCHIVE = Path('/mnt/data2/ForkAudit/math-opsd/run/math500-trajectory-k20')
OUTPUT = Path(__file__).resolve().parent / 'evidence/math500_trajectory/response_stats.csv'
TERMS = ('Wait', 'But', 'Alternatively')
PATTERNS = [re.compile(r'\b' + term + r'\b', re.IGNORECASE) for term in TERMS]
MODELS = ['B'] + [f'{arm}@{step}' for arm in ('R', 'T0', 'P', 'Rres', 'Rw', 'Rnt')
                  for step in (25, 50, 75, 100)]


def summarize(model):
    folder = ARCHIVE / model
    receipt = json.loads((folder / 'receipt.json').read_text())
    raw = (folder / 'answers.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['output_sha256']
    rows = json.loads(raw)['results']
    assert len({r['problem'] for r in rows}) == len(rows) == 500
    assert all(len(r['generations']) == 4 for r in rows)
    answers = [g for r in rows for g in r['generations']]
    assert {g['finish_reason'] for g in answers} <= {'stop', 'length'}
    assert sum(g['correct'] for g in answers) == receipt['metrics']['correct']
    kept = [g for g in answers if g['finish_reason'] != 'length']
    n = len(answers)
    correct = sum(g['correct'] for g in kept)
    texts = [g['full_generation'] for g in answers]
    boxed = sum(r'\boxed' in text for text in texts)
    counts = [sum(len(p.findall(text)) for text in texts) for p in PATTERNS]
    return [model, n, n-len(kept), len(kept), correct, boxed, *counts,
            100*(n-len(kept))/n, 100*correct/len(kept), 100*boxed/n,
            *(c/n for c in counts)]


if __name__ == '__main__':
    values = [summarize(m) for m in MODELS]
    with OUTPUT.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','answers','truncated','untruncated','untruncated_correct',
                         'contains_boxed',*['total_'+t for t in TERMS],'truncated_pct',
                         'untruncated_accuracy_pct','contains_boxed_pct',*['mean_'+t for t in TERMS]])
        writer.writerows(values)
    for row in values:
        print(row[0], *(f'{n:.2f}' for n in row[-6:-3]), *(f'{n:.4f}' for n in row[-3:]))
