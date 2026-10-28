"""Five-arm extension with 8 old Fork, 8 old Indifferent, 8 old all-failure Dead.

Existing cases have already been inspected. Extension selection is deterministic
from old strata, task type, and a hash, never from observed intervention benefit.
Dead cases use original regime flags and feedback eligibility, not new outcomes.
This enriched sample cannot estimate population regime proportions.
"""
import argparse
import copy
from collections import Counter
import re
import pandas as pd
from protocol import (HERE, ROOT, VERSION, ARMS, admissible, digest, file_hash,
                      location_clues, post_facts, read, relation_fact, write,
                      verify_environment)

PAIR_DIR = HERE.parent/'opportunity_completion'
PAIR_RUN = PAIR_DIR/'runs/main_32x16'
TABLE = HERE.parent/'exploratory_point_mechanisms.parquet'
SEED = 2026102807


def all_arms(point):
    p = copy.deepcopy(point)
    pk = read(ROOT/p['peek_path'])
    if len(pk['peeks']) != 5 or not pk['checks']['factual_obs_match'] or pk['checks']['refeed_div_total']:
        return None
    arms = {}
    for i, a in enumerate(pk['peeks']):
        facts = post_facts(pk, a)
        if a['done'] or a['won'] or a['refeed_div'] or p['clue']['fact'] not in facts:
            return None
        arms[f'a{i}'] = dict(action=a['action'], observation_after=a['obs'],
            admissible_after=admissible(a['admissible_after']),
            post_facts_sha256=digest(sorted(facts)), role=a['role'],
            target_word_in_feedback=bool(re.search(r'\b'+re.escape(p['target'])+r' \d+\b', a['obs'])),
            clue_object_in_feedback=p['clue']['object'] in a['obs'])
    assert arms['a0']['action'] == pk['chosen_action']
    p['arms'] = arms
    return p


def balanced(pool, n, used_tasks):
    counts = Counter(); chosen = []
    for _ in range(n):
        eligible = [p for p in pool if p['task_id'] not in used_tasks]
        if not eligible:
            raise ValueError('Insufficient distinct eligible tasks')
        p = min(eligible, key=lambda p: (counts[p['task_type']], digest([SEED, p['id']])))
        chosen.append(p); used_tasks.add(p['task_id']); counts[p['task_type']] += 1
    return chosen


def dead_candidates():
    tab = pd.read_parquet(TABLE)
    tab = tab[(~tab.live) & tab.any_new_to_history & (tab.turn <= 40)
              & (tab.pool_b1 == 0) & (tab.pool_b2 == 0)]
    pool = []
    for row in tab.itertuples():
        tid, turn = row.traj_id, int(row.turn)
        pk_path = f'data/peek_alts/{tid}/t{turn:02d}.json'
        pk = read(ROOT/pk_path)
        if not pk['checks']['factual_obs_match'] or pk['checks']['refeed_div_total']:
            continue
        candidates = []
        for source in pk['peeks']:
            if source['done'] or source['won'] or source['refeed_div']:
                continue
            facts = post_facts(pk, source)
            for clue in location_clues(source['obs'], source['action'], row.target):
                if clue['object'] in set(row.ever_target_ids):
                    continue
                fact = relation_fact(clue['object'], clue['location'], facts)
                if fact is None:
                    continue
                if not all(fact in post_facts(pk, a) for a in pk['peeks']):
                    continue
                candidates.append(dict(clue=dict(clue, fact=fact), source_action=source['action']))
        if not candidates:
            continue
        clue = min(candidates, key=lambda c: digest([SEED, tid, turn, c]))
        tr_path = f'data/rollout/{tid}.json'; tr = read(ROOT/tr_path); step = tr['steps'][turn-1]
        p = dict(id=f'{tid}_t{turn:02d}', traj_id=tid, task_id=tr['task_id'],
                 task_type=tr['task_type'], task=tr['task'], turn=turn, old_regime='Dead',
                 old_fork=False, old_masked=False, target=row.target, game_file=tr['game_file'],
                 observation_before=step['obs_before'], admissible_before=admissible(step['admissible']),
                 prefix=[s['action'] for s in tr['steps'][:turn-1]],
                 expected_prefix_observations=[s['obs_after'] for s in tr['steps'][:turn-1]],
                 memory=[dict(text_obs=s['obs_before'], action=s['action']) for s in tr['steps'][:turn-1]],
                 peek_path=pk_path, rollout_path=tr_path, reuse_arms={}, **clue)
        p = all_arms(p)
        if p:
            pool.append(p)
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-env', action='store_true')
    args = parser.parse_args()
    if (HERE/'design.json').exists():
        parser.error('Design exists; do not overwrite it')
    original = read(PAIR_DIR/'design.json')
    old = []
    for p in original['points']:
        new = all_arms(p)
        if not new:
            continue
        new['old_regime'] = 'Fork' if p['old_fork'] else 'Indifferent'
        new['reuse_arms'] = {name: old_name for name, a in new['arms'].items()
                            for old_name, old_a in p['arms'].items() if a['action'] == old_a['action']}
        new['source_action'] = p['arms']['revealing']['action']
        new['pair_source_index'] = p['index']
        old.append(new)
    used = set()
    points = balanced([p for p in old if p['old_regime']=='Indifferent'], 8, used)
    points += balanced([p for p in old if p['old_regime']=='Fork'], 8, used)
    dead_pool = dead_candidates()
    points += balanced(dead_pool, 8, used)
    points.sort(key=lambda p: digest([SEED, 'run-order', p['id']]))
    inputs = {}
    for path in [TABLE, PAIR_DIR/'design.json', PAIR_RUN/'run_manifest.json']:
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    for i, p in enumerate(points):
        p['index'] = i
        nonrevealing = [a for a in ARMS if not p['arms'][a]['clue_object_in_feedback']]
        p['selective_arm'] = min(nonrevealing or list(ARMS), key=lambda a: digest([SEED,p['id'],a]))
        for key in ['peek_path', 'rollout_path']:
            inputs[p[key]] = file_hash(ROOT/p[key])
        for source_arm in p['reuse_arms'].values():
            for condition in ['native','completed']:
                for rep in range(16):
                    path=PAIR_RUN/'results'/f"{p['id']}__{source_arm}__{condition}__r{rep:03d}.json"
                    inputs[str(path.relative_to(ROOT))] = file_hash(path)
    n_reused=sum(len(p['reuse_arms'])*2*16 for p in points)
    design=dict(protocol=VERSION, status='exploratory; original frozen labels unchanged',
        selection=__doc__, seed=SEED, counts=dict(points=24,tasks=24,per_old_regime=8,
            total_continuations=3840,reused=n_reused,new=3840-n_reused),
        dead_candidate_points=len(dead_pool), dead_candidate_tasks=len({p['task_id'] for p in dead_pool}),
        pair_source_run=str(PAIR_RUN.relative_to(ROOT)),
        intervention='Same true, peek-derived target location sentence appended once after each original '
                     'forced action; original physical state, actions, budget, memory and policy preserved.',
        primary='Within old Fork, broadcast completion lowers unbiased five-arm C=Var_a(q) '
                'while raising uniform-arm mean success. All other old strata reported, including failures.',
        secondary='Selective information goes only to a metadata-fixed nonrevealing arm. '
                  'An action-independent 0.2-probability information lottery has equal expected injection rate. '
                  'Both are identified from measured native/completed cells, not extra independent experiments. '
                  'No claim that random selective allocation increasing variance is a new theorem.',
        variance_identity='C1-C0 = 2 Cov(q0, q1-q0) + Var(q1-q0); use distinct-repeat U-statistics.',
        reporting='Do not infer equivalence from a failed significance test. Frozen historical labels stay fixed; '
                  'same-budget exploratory outcome profiles are descriptive. Do not remove negative or physical cases.',
        reps=16, inputs=inputs, points=points)
    design['design_sha256']=digest(design)
    write(HERE/'design.json',design)
    lines=['# Five-arm regime mechanism design','',__doc__,'',json_counts(design['counts']),'',
           '| Point | Old regime | Task type | Clue | Selective arm |','|---|---|---|---|---|']
    for p in points:
        lines.append(f"| {p['id']} | {p['old_regime']} | {p['task_type']} | {p['clue']['text']} | "
                     f"{p['selective_arm']}: {p['arms'][p['selective_arm']]['action']} |")
    (HERE/'design.md').write_text('\n'.join(lines)+'\n')
    print(design['counts'], 'dead eligible',len(dead_pool),flush=True)
    print([(p['id'],p['old_regime']) for p in points],flush=True)
    if args.verify_env:
        records=verify_environment(points)
        write(HERE/'design.preflight.json',dict(design_sha256=design['design_sha256'],checked_arms=len(records),records=records))


def json_counts(c):
    import json
    return json.dumps(c,ensure_ascii=False)


if __name__=='__main__':
    main()
