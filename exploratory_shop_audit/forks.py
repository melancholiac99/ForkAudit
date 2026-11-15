"""Outcome-blind nested point selection, 16 drafts, independent 6+6 banks."""
from collections import Counter, defaultdict
import json
import re
import time

from exploratory_shop_audit.data import stable_seed, write_once
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.run_pilot import assert_same, public_state
from exploratory_shop_audit.runtime import append, make_input, parse_generation, read_rows, utc
from exploratory_shop_audit.judge_prompt import build_choice_messages

PAGES = ['search_home', 'search_results', 'product_detail']


def action_key(action):
    name, argument = parse_action_fixed(action)
    return name, argument.lower() if isinstance(argument, str) else argument


def legal(action, state):
    name, argument = action_key(action)
    return bool((name == 'search' and argument) or name == 'finish' or
                (name == 'click' and argument in state['available']['clickables'] and argument != 'search'))


def select_points(steps, task_order, count, phase):
    pools = defaultdict(lambda: defaultdict(list))
    trajectories = defaultdict(list)
    for row in steps:
        if row['phase'] != phase or row['repeat'] != 0:
            continue
        trajectories[row['trajectory_id']].append(row)
        page = row['before']['public']['page_type']
        if row['remaining'] < 3 or not row['legal'] or page not in PAGES:
            continue
        if page == 'product_detail' and not row['before']['stop_available']:
            continue
        pools[page][row['task_id']].append(row)
    selected, used = [], set()
    # Incremental quotas preserve 30 ⊂ 60 ⊂ 120, with distinct tasks across all pages.
    levels = [1] if count == 3 else [10, 20, 40]
    counts = Counter()
    for quota in levels:
        for page in PAGES:
            for task_id in task_order:
                if counts[page] >= quota or len(selected) >= count:
                    break
                if task_id in used or task_id not in pools[page]:
                    continue
                row = min(pools[page][task_id], key=lambda r: (stable_seed(20261115, 'point', phase, task_id, r['step']), r['step']))
                prefix_rows = sorted(trajectories[row['trajectory_id']], key=lambda r: r['step'])[:row['step']]
                selected.append({'state_id': f'{phase}:{task_id}:step{row["step"]}', 'task_id': task_id,
                    'goal_index': int(task_id.split(':')[1]), 'trajectory_id': row['trajectory_id'],
                    'step': row['step'], 'page_type': page, 'remaining': row['remaining'],
                    'original_action': row['action'], 'input': row['input'], 'before': row['before'],
                    'prefix': [r['action'] for r in prefix_rows],
                    'history': [(r['before']['observation'], r['generation']['text']) for r in prefix_rows]})
                used.add(task_id)
                counts[page] += 1
        if len(selected) >= count:
            break
    return {'requested': count, 'selected': len(selected), 'counts': dict(counts),
            'eligible_tasks': {p: len(pools[p]) for p in PAGES}, 'points': selected,
            'rule': 'native legal original action; >=3 remaining; stable task order and uniform hash state; no reward'}


def clean_feedback(result):
    if result['status']['done']:
        return '该动作终止回合。'
    return result['state']['observation']


def audit_state(run, engine, model, tokenizer, point, phase):
    sid = point['state_id']
    root = run / phase
    root.mkdir(exist_ok=True)
    begin = time.monotonic()
    existing = {r['state_id']: r for r in read_rows(root / 'state_results.jsonl')}
    if sid in existing and (existing[sid]['defined'] or phase != 'pilot_forks' or existing[sid].get('throughput_continuations_complete')):
        return existing[sid]
    input_ids = point['input']['token_ids']
    draft_records = [r for r in read_rows(root / 'drafts.jsonl') if r['state_id'] == sid]
    done_drafts = {r['draft'] for r in draft_records}
    pending = [i for i in range(16) if i not in done_drafts]
    if pending:
        requests = [{'token_ids': input_ids, 'seed': stable_seed(20261115, sid, 'draft', i)} for i in pending]
        outputs = model.generate(requests)
        for i, req, out in zip(pending, requests, outputs):
            action = parse_generation(out['text'])
            row = {'state_id': sid, 'draft': i, 'seed': req['seed'], 'generation': out,
                   'action': action, 'parsed_action': action_key(action), 'legal': legal(action, point['before'])}
            append(root / 'drafts.jsonl', row)
            draft_records.append(row)
    draft_records.sort(key=lambda r: r['draft'])
    frequencies = Counter(action_key(r['action']) for r in draft_records if r['legal'])
    candidates = [point['original_action']]
    if point['before']['stop_available'] and action_key(candidates[0]) != ('click', 'buy now'):
        candidates.append('click[buy now]')
    seen = {action_key(a) for a in candidates}
    for row in draft_records:
        key = action_key(row['action'])
        if row['legal'] and key not in seen and len(candidates) < 4:
            candidates.append(row['action'])
            seen.add(key)
    definitions = []
    existing_candidates = {(r['state_id'], r['candidate']): r for r in read_rows(root / 'candidates.jsonl')}
    restore_seconds = 0
    for cid, action in enumerate(candidates):
        key = f'{sid}:candidate:{cid}'
        tick = time.monotonic()
        state = engine.call(cmd='open', key=key, goal_index=point['goal_index'], prefix=point['prefix'])
        restore_seconds += time.monotonic() - tick
        assert_same(public_state(state), point['before'], 'fork prefix')
        first = engine.call(cmd='step', key=key, action=action)
        row = {'state_id': sid, 'task_id': point['task_id'], 'candidate': cid, 'action': action,
               'parsed_action': action_key(action), 'draft_frequency': frequencies[action_key(action)],
               'draft_denominator': 16, 'original': cid == 0, 'terminal': first['status']['done'],
               'one_step': first, 'clean_feedback': clean_feedback(first)}
        old = existing_candidates.get((sid, cid))
        if old:
            # JSON roundtrip normalizes tuple action keys.
            assert_same(json.loads(json.dumps(row)), old, 'resumed deterministic candidate')
        else:
            append(root / 'candidates.jsonl', row)
        definitions.append(row)
        engine.call(cmd='close', key=key)
    defined = len(definitions) >= 2
    repeats = []
    completed = {}
    for bank in ['A', 'B']:
        completed.update({r['replay_id']: r for r in read_rows(root / f'replay_bank_{bank.lower()}.jsonl')})
    # A single legal candidate is undefined for action comparisons, but its
    # continuations remain a valid cost measurement at a pilot-only point.
    if defined or phase == 'pilot_forks':
        for candidate in definitions:
            if candidate['terminal']:
                continue
            for bank in ['A', 'B']:
                for repeat in range(6):
                    rid = f'{sid}:c{candidate["candidate"]}:{bank}:{repeat}'
                    if rid not in completed:
                        repeats.append({'replay_id': rid, 'candidate': candidate, 'bank': bank, 'repeat': repeat,
                                        'seed': stable_seed(20261115, sid, candidate['candidate'], bank, repeat)})
        for offset in range(0, len(repeats), 16):
            active = []
            for replay in repeats[offset:offset + 16]:
                rid, candidate = replay['replay_id'], replay['candidate']
                tick = time.monotonic()
                state = engine.call(cmd='open', key=rid, goal_index=point['goal_index'],
                                    prefix=point['prefix'] + [candidate['action']])
                restore_seconds += time.monotonic() - tick
                assert_same(state, candidate['one_step']['state'], 'candidate replay start')
                replay.update({'state': state, 'length': 0,
                    'history': list(point['history']) + [(point['before']['observation'], 'Action: ' + candidate['action'])]})
                active.append(replay)
            while active:
                inputs = [make_input(tokenizer, r['state']['task'], r['history'], r['state']['observation']) for r in active]
                requests = [{'token_ids': inp['token_ids'], 'seed': stable_seed(r['seed'], 'decision', r['length'])} for r, inp in zip(active, inputs)]
                outputs = model.generate(requests)
                survivors = []
                for replay, inp, req, out in zip(active, inputs, requests, outputs):
                    action = parse_generation(out['text'])
                    before = replay['state']
                    result = engine.call(cmd='step', key=replay['replay_id'], action=action)
                    append(root / 'replay_steps.jsonl', {'replay_id': replay['replay_id'], 'state_id': sid,
                        'step': replay['length'], 'seed': req['seed'], 'remaining': before['remaining'],
                        'input': inp, 'generation': out, 'action': action, 'legal': result['legal'],
                        'before': public_state(before), 'after': public_state(result['state']), 'done': result['status']['done']})
                    append(root / 'replay_backend_steps.jsonl', {'replay_id': replay['replay_id'], 'step': replay['length'],
                           'before': before['backend'], 'after': result['state']['backend'],
                           'stop_score': before['stop_score'], 'status': result['status'], 'no_budget_gate': result['no_budget_gate']})
                    replay['length'] += 1
                    replay['history'].append((before['observation'], out['text']))
                    replay['state'] = result['state']
                    if result['status']['done']:
                        row = {k: replay[k] for k in ['replay_id', 'bank', 'repeat', 'seed', 'length']}
                        row.update({'state_id': sid, 'task_id': point['task_id'], 'candidate': replay['candidate']['candidate'],
                            'initial_remaining': point['remaining'] - 1, 'reward': result['status']['reward'],
                            'no_budget_gate': result['no_budget_gate'], 'status': result['status'],
                            'success': result['status'].get('reward_detail', {}).get('reward_type') == 'gold_purchase' and result['status'].get('reward_valid', False)})
                        append(root / f'replay_bank_{replay["bank"].lower()}.jsonl', row)
                        engine.call(cmd='close', key=replay['replay_id'])
                    else:
                        if result['state']['remaining'] <= 0:
                            raise RuntimeError('Continuation exceeded native action horizon')
                        survivors.append(replay)
                active = survivors
    valid_drafts = sum(r['legal'] for r in draft_records)
    covered = sum(frequencies[action_key(a)] for a in candidates)
    query_sets = []
    for candidate in definitions:
        if action_key(candidate['action'])[0] == 'search':
            ids = {str(p['asin']) for p in candidate['one_step']['state']['public'].get('products', [])}
            # Engine session search_results are the full native ranked result set, not merely one page.
            session = candidate['one_step']['state']['backend']['session']
            full = session.get('search_result_asins')
            query_sets.append({'candidate': candidate['candidate'], 'query': action_key(candidate['action'])[1],
                               'displayed_ids': sorted(ids), 'full_result_ids': full})
    overlaps = []
    for i, left in enumerate(query_sets):
        for right in query_sets[i + 1:]:
            x, y = set(left['full_result_ids'] or left['displayed_ids']), set(right['full_result_ids'] or right['displayed_ids'])
            overlaps.append({'a': left['candidate'], 'b': right['candidate'], 'jaccard': len(x & y) / len(x | y) if x | y else 1.0,
                             'identical': x == y, 'full_results_available': left['full_result_ids'] is not None and right['full_result_ids'] is not None})
    row = {'state_id': sid, 'task_id': point['task_id'], 'page_type': point['page_type'], 'defined': defined,
           'candidate_count': len(candidates), 'drafts': 16, 'valid_drafts': valid_drafts,
           'covered_drafts': covered, 'omitted_valid_drafts': valid_drafts - covered,
           'query_sets': query_sets, 'query_overlaps': overlaps, 'restore_seconds': restore_seconds,
           'total_seconds': time.monotonic() - begin, 'gpu_hours': model.hours(),
           'throughput_continuations_complete': phase == 'pilot_forks'}
    append(root / 'state_results.jsonl', row)
    append(run / 'progress.jsonl', {'time': utc(), 'phase': phase, **row})
    print(json.dumps({k: row[k] for k in ['state_id', 'defined', 'candidate_count', 'total_seconds', 'gpu_hours']}), flush=True)
    return row


def judge_state(run, model, tokenizer, point, candidates):
    if len(candidates) < 2:
        return
    order = sorted(candidates, key=lambda c: stable_seed(20261115, point['state_id'], 'display', c['candidate']))
    existing = {(r['state_id'], r['condition']) for r in read_rows(run / 'judge_outputs.jsonl')}
    for condition in ['S', 'F']:
        if (point['state_id'], condition) in existing:
            continue
        messages = build_choice_messages(point['input']['messages'], order, condition)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        request = {'state_id': point['state_id'], 'condition': condition, 'messages': messages,
                   'token_ids': ids, 'seed': stable_seed(20261115, point['state_id'], 'judge', condition),
                   'order': [c['candidate'] for c in order], 'temperature': 0, 'max_tokens': 256,
                   'cleaning': 'terminal => uniform termination marker; otherwise native public renderer only'}
        append(run / 'judge_requests.jsonl', request)
        if len(ids) > 8192:
            result = {'state_id': point['state_id'], 'condition': condition, 'selected': None, 'error': 'INPUT_OVERFLOW'}
        else:
            out = model.generate([request])[0]
            match = re.fullmatch(r'\s*([1-4])\s*', out['text'])
            index = int(match.group(1)) - 1 if match else -1
            selected = order[index]['candidate'] if 0 <= index < len(order) else None
            result = {'state_id': point['state_id'], 'condition': condition, 'selected': selected,
                      'generation': out, 'error': None if selected is not None else 'PARSE_FAILURE'}
        append(run / 'judge_outputs.jsonl', result)
