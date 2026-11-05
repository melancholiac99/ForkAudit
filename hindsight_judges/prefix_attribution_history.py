"""Read-only historical KEEP/DROP audits. No old result is changed or reused as new scores."""
import json
from pathlib import Path
import numpy as np
import exp2_common as C
import exp3_action_diagnostic as D

BASE = Path(__file__).resolve().parent / 'results/action_diagnostic'
REVIEW_TV = .005  # Conservative review trigger, not a scientific significance threshold.


def historical_paths():
    paths = [BASE / 'requests.jsonl.gz', BASE / 'prepare.json']
    for name in ('7b', '32b'):
        paths += [BASE / name / 'full' / f for f in ('run.json', 'scores.jsonl')]
    return paths


def original():
    requests, h = C.decode_rows(BASE / 'requests.jsonl.gz')
    assert h == C.read_json(BASE / 'prepare.json')['request_hash']
    return {(r['point_id'], r['condition'], r['think']): r for r in requests}


def scores(name):
    folder = BASE / name / 'full'
    run = C.read_json(folder / 'run.json')
    assert run['complete'] and C.sha_file(folder / 'scores.jsonl') == run['scores_sha256']
    result = {}
    requests = original()
    for line in (folder / 'scores.jsonl').read_text().splitlines():
        z = json.loads(line)
        key = (z['point_id'], z['condition'], z['think'])
        assert key not in result and z['sha256'] == requests[key]['sha256']
        ll = np.array(z['sequence_logp'], float)
        p = np.exp(ll - ll.max()); p /= p.sum()
        assert ll.shape == (5,) and np.isfinite(ll).all()
        assert np.allclose(p, z['prob'], atol=1e-12, rtol=0)
        result[key] = z
    assert len(result) == run['n'] == 3088
    return result


def prepare_audit(out, rows, tok):
    from exp4_prefix_attribution import tokenize
    old = original()
    anchors = [r for r in rows if r['condition'] in ('KEEP', 'DROP')]
    for r in anchors:
        prior = old[r['point_id'], r['evidence'], r['condition'].lower()]
        for field in ('prompt', 'reply_prefix', 'action_lead', 'action_tail', 'actions'):
            assert r[field] == prior[field], (r['id'], field)
        a, b = tokenize(tok, r), D.tokenize(tok, prior)
        assert a['prefix_ids'] == b['prefix_ids'] and a['candidates'] == b['candidates'], r['id']
    assert len(anchors) == 1544
    headline = {}
    slopes = {}
    for model in ('7b', '32b'):
        old_scores = scores(model)
        for e in ('M', 'F'):
            for c in ('KEEP', 'DROP'):
                matched = [r for r in anchors if r['evidence'] == e and r['condition'] == c]
                headline[f'{model}_{e}_{c}'] = float(np.mean([
                    np.array(old_scores[r['point_id'], e, c.lower()]['prob']) @ np.array(r['q3456']) * 100
                    for r in matched]))
        # Recompute the historical centered definition; never assume quoted 0.55/0.36 use the same formula.
        for c in ('KEEP', 'DROP'):
            num = den = 0.
            for r in anchors:
                if r['condition'] != c or r['evidence'] != 'M': continue
                q = np.array(r['q3456']); q -= q.mean()
                delta = np.array(old_scores[r['point_id'], 'F', c.lower()]['sequence_logp']) - np.array(old_scores[r['point_id'], 'M', c.lower()]['sequence_logp'])
                delta -= delta.mean()
                num += float(q @ delta); den += float(q @ q)
            slopes[f'{model}_{c}'] = num / den
        slopes[f'{model}_KEEP_over_DROP'] = slopes[f'{model}_KEEP'] / slopes[f'{model}_DROP']
    audit = dict(passed=True, anchors=1544, historical_raw_scores_hash_validated=True,
                 exact_prompt_prefix_actions_and_reconstructed_token_ids=True,
                 historical_headline_value_pp=headline, historical_centered_slopes=slopes,
                 note='CPU source/token reconstruction and historical value reproduction; fresh GPU agreement is checked after each run.',
                 review_rule='Fresh vs historical candidate TV >= 0.005 requires investigation before full progression or combined interpretation.',
                 sources={str(p): C.sha_file(p) for p in historical_paths()})
    C.save_json(Path(out) / 'historical_cpu_audit.json', audit)
    return audit


def compare(rows, new_scores, model):
    historical = scores(model)
    records = []
    for r in rows:
        if r['condition'] not in ('KEEP', 'DROP'): continue
        old = historical[r['point_id'], r['evidence'], r['condition'].lower()]
        new = new_scores[r['id']]
        p, q = np.array(new['prob']), np.array(old['prob'])
        delta = np.array(new['sequence_logp']) - np.array(old['sequence_logp'])
        tv = float(abs(p - q).sum() / 2)
        records.append(dict(id=r['id'], evidence=r['evidence'], condition=r['condition'],
            historical_sequence_logp=old['sequence_logp'], new_sequence_logp=new['sequence_logp'],
            logp_delta=delta.tolist(), historical_prob=old['prob'], new_prob=new['prob'], tv=tv,
            max_abs_logp_delta=float(abs(delta).max()),
            historical_value_pp=float(q @ np.array(r['q3456']) * 100),
            new_value_pp=float(p @ np.array(r['q3456']) * 100)))
    assert len(records) == len(rows) // 4
    summary = dict(n=len(records), mean_tv=float(np.mean([r['tv'] for r in records])),
                   max_tv=max(r['tv'] for r in records), max_abs_logp_delta=max(r['max_abs_logp_delta'] for r in records),
                   nonidentical_ids=[r['id'] for r in records if r['max_abs_logp_delta'] != 0 or r['tv'] != 0],
                   review_ids=[r['id'] for r in records if r['tv'] >= REVIEW_TV],
                   compatible=all(r['tv'] < REVIEW_TV for r in records), review_tv=REVIEW_TV,
                   review_rule='Numerical review trigger only, not an outcome/significance threshold', cells={})
    for e in ('M', 'F'):
        for c in ('KEEP', 'DROP'):
            rr = [r for r in records if r['evidence'] == e and r['condition'] == c]
            summary['cells'][f'{e}_{c}'] = dict(n=len(rr), mean_tv=float(np.mean([r['tv'] for r in rr])),
                max_tv=max(r['tv'] for r in rr), historical_value_pp=float(np.mean([r['historical_value_pp'] for r in rr])),
                new_value_pp=float(np.mean([r['new_value_pp'] for r in rr])))
    return dict(model=model, summary=summary, records=records)
