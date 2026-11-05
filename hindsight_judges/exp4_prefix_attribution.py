"""Frozen P4 prefix provenance/placement diagnostic; no training or automatic GPU launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import exp1c as E
import exp2_common as C
import exp3_action_diagnostic as D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = HERE / 'PROTOCOL_PREFIX_ATTRIBUTION_20261105.md'
CONDITIONS = ('AS', 'AO', 'QS', 'QO', 'QN', 'AP', 'KEEP', 'DROP')
EVIDENCE = ('M', 'F')
CUES = {
    'neutral': '[DRAFT] Below is draft reasoning for the current step. [/DRAFT]',
    'self': '[DRAFT SOURCE] The draft reasoning below for the current step was written by you. [/DRAFT SOURCE]',
    'other': '[DRAFT SOURCE] The draft reasoning below for the current step was written by another agent. [/DRAFT SOURCE]',
}
QUOTE_OPEN = '[QUOTED DRAFT — context, not an environment observation]\n'
QUOTE_CLOSE = '\n[/QUOTED DRAFT]'
REQUEST = 'Output the next legal action.'
SEED = 20261105
NBOOT = 4000


def digest(value):
    return hashlib.sha256(C.json_bytes(value)).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def implementation():
    names = ('exp4_prefix_attribution.py', 'prefix_attribution_analysis.py',
             'exp3_action_diagnostic.py', 'exp1c.py', 'exp2_common.py',
             'token_diagnostic_backend.py', 'exp1c_backend.py', 'exp1c_precision.py',
             'test_prefix_attribution.py', 'prefix_attribution_history.py', PROTOCOL.name)
    return {name: C.sha_file(HERE / name) for name in names}


def safe_out(path):
    out = Path(path).resolve()
    # Never allow this implementation to write into earlier scientific artifacts.
    forbidden = [ROOT / 'exploratory_P4', ROOT / 'exploratory_E4',
                 HERE / 'results/prefix_attribution_20261105', HERE / 'results/action_diagnostic', HERE / 'results/think_localization']
    forbidden += list((HERE / 'results').glob('token_diagnostic*'))
    forbidden += list((HERE / 'results').glob('exp1*'))
    for p in forbidden:
        if out == p.resolve() or p.resolve() in out.parents:
            raise ValueError(f'Protected earlier experiment: {out}')
    if out in (ROOT, HERE, HERE / 'results'):
        raise ValueError('Use a dedicated new experiment output directory')
    return out


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def tokenizer_identity(tok):
    return dict(vocab_sha256=digest(tok.get_vocab()), chat_template_sha256=text_hash(tok.chat_template),
                eos_token=tok.eos_token, eos_token_id=tok.eos_token_id,
                special_tokens_map=tok.special_tokens_map)


def make_cases(S, L, F, template, feedback_template):
    """Reuse old anchors literally. Only append new interventions to original user text."""
    original = D.make_cases(S, L, F, template, feedback_template)
    anchors = {(r['condition'], r['think']): r for r in original}
    keep = anchors['M', 'keep']['reply_prefix']
    match = list(D.OPEN_ACTION.finditer(keep))[-1]
    assert match.end() == len(keep)
    reasoning, tag = keep[:match.start()], keep[match.start():]
    assert reasoning and E.THINK_BLOCK.search(reasoning)
    assert len(S['actions']) == len(set(S['actions'])) == 5
    assert not any(x in reasoning for x in (QUOTE_OPEN, QUOTE_CLOSE, '<|im_start|>', '<|im_end|>'))
    rows = []
    for evidence in EVIDENCE:
        base = anchors[evidence, 'keep']
        for condition in CONDITIONS:
            source = {'AS': 'self', 'AO': 'other', 'QS': 'self', 'QO': 'other', 'QN': 'neutral', 'AP': 'self'}.get(condition)
            user = base['prompt']
            quote_span = None
            if source:
                user += '\n\n' + CUES[source]
            if condition in ('QS', 'QO', 'QN'):
                user += '\n' + QUOTE_OPEN
                quote_span = [len(user), len(user) + len(reasoning)]
                user += reasoning + QUOTE_CLOSE
                prefix = tag
                assert user[slice(*quote_span)] == reasoning
                assert prefix == tag and not D.CLOSE_THINK.search(prefix)
            elif condition == 'AP':
                prefix = tag
            elif condition == 'DROP':
                prefix = anchors[evidence, 'drop']['reply_prefix']
            else:
                prefix = keep
            # No characters at or beyond the action tag enter the new quote.
            if condition in ('AS', 'AO', 'KEEP'):
                assert prefix == reasoning + tag
            messages = [dict(role='user', content=user)]
            if condition == 'AP':
                messages.append(dict(role='assistant', content=reasoning))
            if condition in ('AP', 'QS', 'QO', 'QN'):
                messages.append(dict(role='user', content=REQUEST))
            outside_think_action = bool(D.OPEN_ACTION.search(E.THINK_BLOCK.sub('', reasoning)))
            row = dict(id=f"{S['point_id']}__{evidence}__{condition}", point_id=S['point_id'],
                       traj=S['traj_id'], turn=int(S['turn']), evidence=evidence, condition=condition,
                       prompt=user, messages=messages, messages_sha256=digest(messages), reply_prefix=prefix, actions=list(S['actions']),
                       preexisting_action_marker_outside_think=outside_think_action,
                       action_lead=base['action_lead'], action_tail=base['action_tail'],
                       cue=source, reasoning_role='user_quote' if quote_span else
                       ('completed_assistant' if condition == 'AP' else 'absent' if condition == 'DROP' else 'active_assistant'),
                       quote_span=quote_span, reasoning_sha256=text_hash(reasoning),
                       prompt_sha256=text_hash(user), reply_prefix_sha256=text_hash(prefix),
                       candidate_serializations_sha256=digest([tag + base['action_lead'] + a + base['action_tail']
                                                              for a in S['actions']]),
                       presentation_sha256=digest(dict(cues=CUES, open=QUOTE_OPEN, close=QUOTE_CLOSE, request=REQUEST)))
            if condition in ('KEEP', 'DROP'):
                old = anchors[evidence, condition.lower()]
                assert all(row[k] == old[k] for k in ('prompt', 'reply_prefix', 'actions', 'action_lead', 'action_tail'))
            rows.append(row)
    return rows


def render(tok, row):
    messages = row['messages']
    for message in messages:
        assert message['role'] in ('user', 'assistant')
        assert not any(x in message['content'] for x in ('<|im_start|>', '<|im_end|>'))
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered.endswith('<|im_start|>assistant\n')
    if row['condition'] == 'AP':
        assert [m['role'] for m in messages] == ['user', 'assistant', 'user']
        assert messages[-1]['content'] == REQUEST
        assert '<|im_start|>assistant\n' + messages[1]['content'] + '<|im_end|>\n<|im_start|>user\n' + REQUEST + '<|im_end|>' in rendered
    if row['condition'] in ('QS', 'QO', 'QN'):
        assert [m['role'] for m in messages] == ['user', 'user']
        assert messages[-1]['content'] == REQUEST
    return rendered


def tokenize(tok, row):
    # Exact exp3 canonical shared-boundary scoring, with structured chat rendering.
    prefix = render(tok, row) + row['reply_prefix']
    pre = tok.encode(prefix, add_special_tokens=False)
    full = [tok.encode(prefix + row['action_lead'] + action + row['action_tail'] + tok.eos_token,
                       add_special_tokens=False) for action in row['actions']]
    start = len(pre)
    for ids in full:
        i = 0
        while i < min(len(pre), len(ids)) and pre[i] == ids[i]: i += 1
        start = min(start, i)
    assert len(pre) - start <= 3
    suffixes = [ids[start:] for ids in full]
    assert all(len(ids) > 1 for ids in suffixes)
    return dict(**row, prefix_ids=full[0][:start], candidates=[dict(suffix_ids=ids) for ids in suffixes],
                max_tokens=max(map(len, full)))


def token_audit(tok, row):
    j = tokenize(tok, row)
    rendered = render(tok, row)
    ids = tok.encode(rendered, add_special_tokens=False)
    return dict(prefix_ids_sha256=digest(j['prefix_ids']),
                candidate_suffix_ids_sha256=digest([c['suffix_ids'] for c in j['candidates']]),
                rendered_chat_sha256=text_hash(rendered), max_tokens=j['max_tokens'],
                message_roles=[m['role'] for m in row['messages']],
                actual_role_tokens={s: tok.encode(s, add_special_tokens=False)
                                    for s in ('<|im_start|>user\n', '<|im_end|>', '<|im_start|>assistant\n')},
                chat_generation_boundary_ids=ids[-5:])


def source_paths(arms):
    paths = [ROOT / 'exploratory_P4' / x for x in (
        'e2_requests.jsonl.gz', 'e2_preparation.json', 'e2m_requests.jsonl.gz', 'e2m_preparation.json',
        'prompts/log.txt', 'prompts/feedback.txt', 'e2.json')]
    paths += [HERE / 'results/exp1_p4' / x for x in ('pointmeta.json', 'manifest.json')]
    paths += [ROOT / 'data/task_list.json', Path(arms).resolve()]
    import prefix_attribution_history as H
    paths += H.historical_paths()
    return {str(p): C.sha_file(p) for p in paths}


def prepare(args):
    out = safe_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(args.model)
    meta, source, _, template, hashes = E.original_sources(ROOT)
    expected = C.read_json(ROOT / 'exploratory_P4/e2.json')['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    q = E.read_arms(ROOT, Path(args.arms), expected)
    feedback_template = (ROOT / 'exploratory_P4/prompts/feedback.txt').read_text()
    rows, errors = [], []
    for m in meta:
        pid = m['point_id']
        try:
            cases = make_cases(*[source[pid][e] for e in ('S', 'L', 'F')], template, feedback_template)
            for row in cases:
                assert row['traj'] == m['traj']
                row['q3456'] = [q[pid][j]['q3456'] for j in range(5)]
                assert np.isfinite(row['q3456']).all() and all(0 <= v <= 1 for v in row['q3456'])
                row['source_input_sha256'] = digest({e: source[pid][e] for e in ('S', 'L', 'F')})
                row['token_audit'] = token_audit(tok, row)
                assert row['token_audit']['max_tokens'] <= 32768, 'No truncation permitted'
                row['sha256'] = digest(row)
            rows.extend(cases)
        except (AssertionError, ValueError, KeyError) as exc:
            errors.append(dict(point_id=pid, error=repr(exc)))
    if errors:
        C.save_json(out / 'prepare_errors.json', dict(errors=errors, status='blocked; no exclusions or source repair'))
        raise RuntimeError(f'{len(errors)} parse/audit failures; see {out / "prepare_errors.json"}')
    assert len(rows) == 6176 and len({r['traj'] for r in rows}) == 60
    sources = source_paths(args.arms)
    manifest = dict(points=386, clusters=60, contexts_per_model=6176, conditions=CONDITIONS,
                    tokenizer=tokenizer_identity(tok), sources=sources, original_source_checks=hashes,
                    protocol_sha256=C.sha_file(PROTOCOL), implementation=implementation(), excluded=[],
                    known_preexisting_action_marker_points=sorted({r['point_id'] for r in rows if r['preexisting_action_marker_outside_think']}),
                    max_tokens=max(r['token_audit']['max_tokens'] for r in rows),
                    anchor_policy='rescore; historical runs lack immutable token/hardware identity required for reuse')
    path = out / 'requests.jsonl.gz'
    if path.exists():
        assert C.decode_rows(path)[0] == rows, 'Different preparation; use a fresh output directory'
        previous = C.read_json(out / 'prepare.json')
        assert all(previous[k] == json.loads(json.dumps(v)) for k, v in manifest.items()), 'Preparation fingerprint mismatch'
    else:
        C.encode_rows(path, rows)
    manifest['request_hash'] = C.decode_rows(path)[1]
    C.save_json(out / 'prepare.json', manifest)
    audit = [dict(id=r['id'], evidence=r['evidence'], condition=r['condition'], cue=r['cue'],
                  reasoning_role=r['reasoning_role'], quote_span=r['quote_span'],
                  prompt_sha256=r['prompt_sha256'], reply_prefix_sha256=r['reply_prefix_sha256'],
                  reasoning_sha256=r['reasoning_sha256'], **r['token_audit']) for r in rows]
    C.save_json(out / 'condition_matrix_audit.json', audit)
    C.save_json(out / 'exact_prompt_examples.json', [dict(id=r['id'], user=r['prompt'], messages=r['messages'],
                active_assistant_prefix=r['reply_prefix'], actions=r['actions'], lead=r['action_lead'],
                tail=r['action_tail'], chat=render(tok, r), token_audit=r['token_audit']) for r in rows[:16]])
    import prefix_attribution_history as H
    H.prepare_audit(out, rows, tok)
    print(f'CPU PREPARED: 386 points, 60 clusters, {len(rows)} contexts/model; max {manifest["max_tokens"]} tokens')


def prepared(out, limit=0):
    out = safe_out(out)
    prep = C.read_json(out / 'prepare.json')
    assert prep['implementation'] == implementation(), 'Implementation changed; use a new preparation'
    assert prep['protocol_sha256'] == C.sha_file(PROTOCOL)
    for path, sha in prep['sources'].items():
        assert C.sha_file(path) == sha, f'Frozen source changed: {path}'
    rows, h = C.decode_rows(out / 'requests.jsonl.gz')
    assert h == prep['request_hash'] and len(rows) == 6176
    for row in rows:
        assert row['sha256'] == digest({k: v for k, v in row.items() if k != 'sha256'})
    pids = sorted({r['point_id'] for r in rows})
    assert len(pids) == 386 and len({r['traj'] for r in rows}) == 60
    if limit:
        rows = [r for r in rows if r['point_id'] in set(pids[:limit])]
    assert len({r['id'] for r in rows}) == len(rows)
    return rows, prep


def run_dir(args):
    return safe_out(args.out) / args.name / ('full' if args.limit == 0 else f'first{args.limit}')


def softmax(ll):
    ll = np.asarray(ll, dtype=float)
    assert ll.shape == (5,) and np.isfinite(ll).all()
    p = np.exp(ll - ll.max())
    return p / p.sum()


def read_scores(folder, rows, complete=True):
    run = C.read_json(folder / 'run.json')
    assert run['fingerprint']['implementation'] == implementation()
    if complete:
        assert run['complete'] and C.sha_file(folder / 'scores.jsonl') == run['scores_sha256']
        assert C.sha_file(folder / 'numerical_gate.json') == run['gate_sha256']
    expected = {r['id']: r for r in rows}
    result = {}
    path = folder / 'scores.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        z = json.loads(line)
        assert z['id'] in expected and z['id'] not in result, 'Unknown/duplicate row'
        r = expected[z['id']]
        assert z['sha256'] == r['sha256'] and z['request'] == r
        assert z['fingerprint_sha256'] == digest(run['fingerprint'])
        assert np.asarray(z['prob']).shape == (5,)
        assert np.allclose(softmax(z['sequence_logp']), z['prob'], atol=1e-12, rtol=0)
        assert z['record_sha256'] == digest({k: v for k, v in z.items() if k != 'record_sha256'})
        result[z['id']] = z
    if complete:
        assert set(result) == set(expected) and run['n'] == len(rows)
        gate = C.read_json(folder / 'numerical_gate.json')
        assert gate and all(g['passed'] and np.isfinite(g['tv']) and g['tv'] < .005 for g in gate)
    return result, run


def score(args):
    import torch
    import transformers
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import token_diagnostic_backend as B
    assert not args.limit or 32 <= args.limit <= 64, 'Pilot must use first 32–64 points'
    # Refuse to allocate a model before the CPU construction/statistics checks and
    # full-run pilot audit have passed. The subprocess is explicitly CPU-only.
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                    '-p', 'test_prefix_attribution.py', '-v'], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    rows, prep = prepared(args.out, args.limit)
    historical_cpu = C.read_json(safe_out(args.out) / 'historical_cpu_audit.json')
    assert historical_cpu['passed'] and historical_cpu['anchors'] == 1544
    assert all(prep['sources'][path] == sha for path, sha in historical_cpu['sources'].items())
    pilot_run = None
    if not args.limit:
        pilot = safe_out(args.out) / args.name / f'first{args.pilot_size}'
        pilot_rows, _ = prepared(args.out, args.pilot_size)
        _, pilot_run = read_scores(pilot, pilot_rows)
        verification = C.read_json(pilot / 'verification.json')
        assert verification['passed'] and verification['historical_anchors_compatible'] and pilot_run['complete']
        assert verification['run_sha256'] == C.sha_file(pilot / 'run.json')
        assert verification['scores_sha256'] == C.sha_file(pilot / 'scores.jsonl')
        assert verification['analysis_sha256'] == C.sha_file(pilot / 'analysis.json')
    pin = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert pin.startswith('GPU-') and ',' not in pin, 'Pin one recorded GPU UUID with CUDA_VISIBLE_DEVICES'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    folder = run_dir(args)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'score.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tok = tokenizer(args.model)
        assert tokenizer_identity(tok) == prep['tokenizer'], 'Different tokenizer/template'
        jobs = []
        for row in rows:
            assert token_audit(tok, row) == row['token_audit']
            jobs.append(tokenize(tok, row))
        # Fixed eight-condition order; every condition scored regardless of outcomes.
        jobs.sort(key=lambda j: (CONDITIONS.index(j['condition']), j['point_id'], j['evidence']))
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            scorer, model = B.load_model(args.name, args.model, ROOT)
            runtime = B.runtime()
            runtime['transformers'] = transformers.__version__
            runtime['selected_gpu_uuid'] = pin
            runtime['fp16_reduced_precision_reduction'] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            fingerprint = dict(name=args.name, request_hash=prep['request_hash'], implementation=implementation(),
                               tokenizer=prep['tokenizer'], model=model, runtime=runtime, limit=args.limit,
                               model_config_files={p.name: C.sha_file(p) for p in sorted(Path(args.model).glob('*.json'))})
            if pilot_run is not None:
                assert {k: v for k, v in pilot_run['fingerprint'].items() if k != 'limit'} == {
                    k: v for k, v in fingerprint.items() if k != 'limit'}, 'Pilot/full execution configuration differs'
            runfile = folder / 'run.json'
            completed = {}
            if runfile.exists():
                old = C.read_json(runfile)
                assert old['fingerprint'] == fingerprint, 'Resume fingerprint changed (including GPU UUID)'
                completed, _ = read_scores(folder, rows, complete=old['complete'])
                if old['complete']:
                    print('Already complete and hash checked'); return
            else:
                assert not (folder / 'scores.jsonl').exists(), 'Orphan scores'
            C.save_json(runfile, dict(fingerprint=fingerprint, complete=False, started_utc=utc()))
            gates = []
            chosen = [min(jobs, key=lambda j: j['max_tokens']), max(jobs, key=lambda j: j['max_tokens'])]
            chosen += [min((j for j in jobs if j['condition'] == c and j['evidence'] == 'F'),
                           key=lambda j: j['max_tokens']) for c in CONDITIONS]
            seen = set()
            for j in chosen:
                if j['id'] in seen: continue
                seen.add(j['id'])
                a = [sum(v) for v in scorer.score(j)]
                b = [sum(scorer.direct(j, k)) for k in range(5)]
                tv = float(np.abs(softmax(a) - softmax(b)).sum() / 2)
                gates.append(dict(id=j['id'], tv=tv, passed=tv < .005, cached=a, direct=b))
            C.save_json(folder / 'numerical_gate.json', gates)
            assert all(g['passed'] for g in gates), 'TV gate failed; formal scoring blocked'
            request = {r['id']: r for r in rows}
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            fresh = 0
            with (folder / 'scores.jsonl').open('a', buffering=1) as f:
                for j in jobs:
                    if j['id'] in completed: continue
                    tick = time.monotonic()
                    ll = [sum(v) for v in scorer.score(j)]
                    torch.cuda.synchronize()
                    record = dict(id=j['id'], sha256=j['sha256'], request=request[j['id']],
                                  sequence_logp=ll, prob=softmax(ll).tolist(), utc=utc(),
                                  seconds=time.monotonic() - tick, fingerprint_sha256=digest(fingerprint))
                    record['record_sha256'] = digest(record)
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush(); os.fsync(f.fileno())
                    completed[j['id']] = record
                    fresh += 1
                    C.save_json(folder / 'progress.json', dict(done=len(completed), total=len(jobs),
                                new_contexts=fresh, elapsed_seconds=time.monotonic() - start,
                                seconds_per_context=(time.monotonic() - start) / fresh,
                                peak_vram_bytes=torch.cuda.max_memory_allocated(), utc=utc()))
                    if fresh % 20 == 0: print(args.name, len(completed), '/', len(jobs), flush=True)
            assert len(completed) == len(rows)
            C.save_json(runfile, dict(fingerprint=fingerprint, complete=True, n=len(rows), finished_utc=utc(),
                        scores_sha256=C.sha_file(folder / 'scores.jsonl'),
                        gate_sha256=C.sha_file(folder / 'numerical_gate.json'),
                        total_scoring_seconds=sum(z['seconds'] for z in completed.values()),
                        peak_vram_bytes=torch.cuda.max_memory_allocated()))


def analyze(args):
    import prefix_attribution_analysis as A
    rows, prep = prepared(args.out, args.limit)
    folder = run_dir(args)
    scores, run = read_scores(folder, rows)
    assert run['fingerprint']['request_hash'] == prep['request_hash']
    import prefix_attribution_history as H
    history = H.compare(rows, scores, args.name)
    C.save_json(folder / 'historical_anchor_comparison.json', history)
    result, points = A.analyze_rows(rows, scores)
    result.update(historical_anchors=history['summary'], known_preexisting_action_marker_points=prep['known_preexisting_action_marker_points'], model=args.name, cohort=folder.name, run_sha256=C.sha_file(folder / 'run.json'),
                  scores_sha256=run['scores_sha256'], implementation=implementation(),
                  numerical_gate=C.read_json(folder / 'numerical_gate.json'),
                  runtime=run['fingerprint']['runtime'], model_fingerprint=run['fingerprint']['model'],
                  total_scoring_seconds=run['total_scoring_seconds'], peak_vram_bytes=run['peak_vram_bytes'],
                  source_hashes=prep['sources'], excluded=prep['excluded'])
    C.encode_rows(folder / 'point_metrics.jsonl.gz', points)
    C.save_json(folder / 'analysis.json', result)
    (folder / 'REPORT.txt').write_text(A.report(result))
    print(folder / 'REPORT.txt')


def verify(args):
    """Separate audit: reconstruct primary estimands directly from raw log likelihoods."""
    rows, prep = prepared(args.out, args.limit)
    folder = run_dir(args)
    scores, run = read_scores(folder, rows)
    assert run['fingerprint']['request_hash'] == prep['request_hash']
    assert run['fingerprint']['limit'] == args.limit and run['fingerprint']['name'] == args.name
    analysis = C.read_json(folder / 'analysis.json')
    assert analysis['run_sha256'] == C.sha_file(folder / 'run.json')
    assert analysis['scores_sha256'] == run['scores_sha256'] and analysis['implementation'] == implementation()
    values, clusters = {}, {}
    for row in rows:
        ll = np.asarray(scores[row['id']]['sequence_logp'], dtype=float)
        weights = np.exp(ll - np.max(ll)); weights /= sum(weights)
        values[row['point_id'], row['evidence'], row['condition']] = float(sum(weights * np.array(row['q3456'])))
        clusters[row['point_id']] = row['traj']
    pids = sorted(clusters)
    primary = {k: [] for k in ('SLOT_F', 'SLOT_M', 'I_SLOT', 'P_Q_F', 'P_Q_M', 'I_Q')}
    for pid in pids:
        f = values[pid, 'F', 'QO'] - values[pid, 'F', 'QS']
        m = values[pid, 'M', 'QO'] - values[pid, 'M', 'QS']
        sf = values[pid, 'F', 'QS'] - values[pid, 'F', 'AP']
        sm = values[pid, 'M', 'QS'] - values[pid, 'M', 'AP']
        for key, value in zip(primary, (sf, sm, sf-sm, f, m, f - m)):
            primary[key].append(value * 100)
    checks = {}
    # Independent cluster bootstrap implementation; exact same fixed resampling protocol.
    labels = sorted(set(clusters.values()))
    counts = np.array([sum(clusters[p] == c for p in pids) for c in labels])
    draws = np.random.default_rng(SEED).multinomial(len(labels), [1 / len(labels)] * len(labels), size=NBOOT)
    for key, vector in primary.items():
        sums = np.array([sum(v for p, v in zip(pids, vector) if clusters[p] == c) for c in labels])
        estimate = float(np.mean(vector))
        ci = np.percentile((draws @ sums) / (draws @ counts), [2.5, 97.5])
        reported = analysis['primary'][key]
        assert abs(estimate - reported['estimate']) < 1e-10
        assert np.allclose(ci, reported['ci95'], atol=1e-10, rtol=0)
        checks[key] = dict(estimate=estimate, ci95=ci.tolist())
    import prefix_attribution_history as H
    history = H.compare(rows, scores, args.name)
    assert history == C.read_json(folder / 'historical_anchor_comparison.json')
    assert history['summary']['compatible'], 'Historical anchors require investigation; no mixed interpretation/full progression'
    verify_centered(rows, scores, analysis, clusters, draws, labels)
    assert analysis['n_points'] == len(pids) and analysis['n_clusters'] == len(labels)
    C.save_json(folder / 'verification.json', dict(passed=True, utc=utc(), points=len(pids),
                contexts=len(rows), clusters=len(labels), primary_recomputed=checks,
                historical_anchors_compatible=True, centered_slopes_recomputed=True,
                raw_scores_validated=True, frozen_sources_unchanged=True,
                run_sha256=C.sha_file(folder / 'run.json'), scores_sha256=C.sha_file(folder / 'scores.jsonl'),
                analysis_sha256=C.sha_file(folder / 'analysis.json')))
    print('VERIFIED', folder)


def verify_centered(rows, scores, analysis, clusters, draws, labels):
    """Independently form all five-arm centered products and paired bootstrap ratios."""
    indexed = {(r['point_id'], r['evidence'], r['condition']): r for r in rows}
    pids = sorted(clusters)
    by_condition = {}
    estimates = {}
    for condition in CONDITIONS:
        numerator = np.zeros(len(labels)); denominator = np.zeros(len(labels))
        for pid in pids:
            r = indexed[pid, 'M', condition]
            q = np.asarray(r['q3456'], float)
            d = np.asarray(scores[indexed[pid, 'F', condition]['id']]['sequence_logp']) - np.asarray(scores[r['id']]['sequence_logp'])
            group = labels.index(clusters[pid])
            for j in range(5):
                numerator[group] += (q[j] - q.mean()) * (d[j] - d.mean())
                denominator[group] += (q[j] - q.mean()) ** 2
        den = denominator.sum()
        estimate = float(numerator.sum() / den) if den else None
        boot_den = draws @ denominator
        boot = np.divide(draws @ numerator, boot_den, out=np.full(NBOOT, np.nan), where=boot_den != 0)
        reported = analysis['centered']['conditions'][condition]['slope']
        if estimate is None: assert reported['estimate'] is None
        else: assert abs(estimate - reported['estimate']) < 1e-10
        if np.isfinite(boot).all():
            assert np.allclose(np.percentile(boot, [2.5, 97.5]), reported['ci95'], atol=1e-10, rtol=0)
        else: assert reported['ci95'] is None
        by_condition[condition] = boot; estimates[condition] = estimate
    for condition in CONDITIONS:
        report = analysis['centered']['conditions'][condition]['ratio_to_DROP']
        ratio = estimates[condition] / estimates['DROP'] if estimates['DROP'] not in (0, None) else None
        if ratio is None: assert report['estimate'] is None
        else: assert abs(ratio - report['estimate']) < 1e-10
        if report['ci95'] is not None:
            ref = by_condition['DROP']
            dist = np.divide(by_condition[condition], ref, out=np.full(NBOOT, np.nan), where=ref != 0)
            assert np.isfinite(dist).all()
            assert np.allclose(np.percentile(dist, [2.5, 97.5]), report['ci95'], atol=1e-10, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'score', 'analyze', 'verify'))
    parser.add_argument('--out', default=str(HERE / 'results/prefix_attribution_v2_20261105'))
    parser.add_argument('--model', help='Local frozen model/tokenizer directory; required for prepare/score')
    parser.add_argument('--arms', default=str(ROOT / 'exploratory_deep_discovery/exploratory_arms.parquet'))
    parser.add_argument('--name', choices=('7b', '32b'), default='7b')
    parser.add_argument('--limit', type=int, default=0, help='0=full; otherwise first 32–64 points (separate directory)')
    parser.add_argument('--pilot-size', type=int, choices=range(32, 65), default=32)
    args = parser.parse_args()
    if args.limit != 0 and not 32 <= args.limit <= 64: parser.error('--limit must be 0 or 32–64')
    if args.command in ('prepare', 'score') and not args.model: parser.error('--model is required')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
