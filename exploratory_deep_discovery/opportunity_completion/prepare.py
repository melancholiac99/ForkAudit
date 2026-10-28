"""Select an enriched, outcome-independent-within-stratum set of information cases.

Selection uses the existing Live/Fork flags for 24/8 strata, but no action success
rates or Gold outcomes. Arm orientation is determined solely by target feedback.
One prefix per task; target instance must be absent from recorded factual history
(including admissible actions), and its peek-derived location must stay true in
both branches. This is an enriched mechanism sample, not a population estimate.
"""
import argparse
from collections import Counter
from pathlib import Path
import random
import re

import pandas as pd

from protocol import (HERE, ROOT, VERSION, ARMS, admissible, digest, file_hash,
                      location_clues, post_facts, read, relation_fact,
                      verify_environment, write)


def candidates():
    table = ROOT / 'exploratory_deep_discovery/exploratory_point_mechanisms.parquet'
    points = pd.read_parquet(table)
    points = points[points.live & points.any_new_to_history]
    result = []
    trajectories = {}
    for row in points.itertuples():
        tid, turn = row.traj_id, int(row.turn)
        if turn > 45:
            continue
        peek_path = ROOT / f'data/peek_alts/{tid}/t{turn:02d}.json'
        pk = read(peek_path)
        if not pk['checks']['factual_obs_match'] or pk['checks']['refeed_div_total']:
            continue
        tr = trajectories.setdefault(tid, None)
        if tr is None:
            tr = trajectories[tid] = read(ROOT / f'data/rollout/{tid}.json')
        step = tr['steps'][turn - 1]
        assert step['turn'] == turn
        history = set(row.ever_target_ids)
        pairs = []
        for source in pk['peeks']:
            if source['done'] or source['won'] or source['refeed_div']:
                continue
            fs = post_facts(pk, source)
            for clue in location_clues(source['obs'], source['action'], row.target):
                if clue['object'] in history:
                    continue
                fact = relation_fact(clue['object'], clue['location'], fs)
                if fact is None:
                    continue
                for other in pk['peeks']:
                    if other['done'] or other['won'] or other['refeed_div']:
                        continue
                    if re.search(r'\b' + re.escape(row.target) + r' \d+\b', other['obs']):
                        continue
                    fo = post_facts(pk, other)
                    if fact not in fo:
                        continue
                    arms = {}
                    for name, a, facts in [('revealing', source, fs), ('other', other, fo)]:
                        arms[name] = dict(action=a['action'], observation_after=a['obs'],
                                          admissible_after=admissible(a['admissible_after']),
                                          post_facts_sha256=digest(sorted(facts)))
                    pairs.append(dict(clue=dict(clue, fact=fact), arms=arms,
                                      same_post_world_facts=fs == fo,
                                      same_post_admissible=admissible(source['admissible_after']) ==
                                                           admissible(other['admissible_after'])))
        if not pairs:
            continue
        # No best-q or positive-gap filtering: one deterministic random pair per point.
        pair = min(pairs, key=lambda x: digest([tid, turn, x['arms'], x['clue']]))
        result.append(dict(id=f'{tid}_t{turn:02d}', traj_id=tid, task_id=tr['task_id'],
                           task_type=tr['task_type'], task=tr['task'], turn=turn,
                           old_fork=bool(row.het), old_masked=bool(row.masked),
                           target=row.target, game_file=tr['game_file'],
                           observation_before=step['obs_before'],
                           admissible_before=admissible(step['admissible']),
                           prefix=[s['action'] for s in tr['steps'][:turn - 1]],
                           expected_prefix_observations=[s['obs_after'] for s in tr['steps'][:turn - 1]],
                           memory=[dict(text_obs=s['obs_before'], action=s['action'])
                                   for s in tr['steps'][:turn - 1]],
                           peek_path=str(peek_path.relative_to(ROOT)),
                           rollout_path=f'data/rollout/{tid}.json', **pair))
    return table, result


def select(pool, n, n_fork, seed):
    rng = random.Random(seed)
    groups = {value: [p for p in pool if p['old_fork'] == value] for value in (True, False)}
    for group in groups.values():
        rng.shuffle(group)
    quotas = {True: n_fork, False: n - n_fork}
    chosen, tasks = [], set()
    # Draw smaller stratum first; balance task types within each stratum.
    for flag in sorted(quotas, key=lambda v: quotas[v]):
        task_counts = Counter()
        for _ in range(quotas[flag]):
            eligible = [p for p in groups[flag] if p['task_id'] not in tasks]
            if not eligible:
                raise ValueError(f'Not enough distinct tasks for Fork={flag}; reduce quotas')
            p = min(eligible, key=lambda x: task_counts[x['task_type']])
            chosen.append(p)
            tasks.add(p['task_id'])
            task_counts[p['task_type']] += 1
    rng.shuffle(chosen)
    for i, p in enumerate(chosen):
        p['index'] = i
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=int, default=32)
    parser.add_argument('--fork-points', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--out', type=Path, default=HERE / 'design.json')
    parser.add_argument('--verify-env', action='store_true')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not 0 <= args.fork_points <= args.points:
        parser.error('Require 0 <= --fork-points <= --points')
    if args.out.exists():
        parser.error(f'{args.out} already exists; use a new filename to change selection')
    table, pool = candidates()
    points = select(pool, args.points, args.fork_points, args.seed)
    inputs = {str(table.relative_to(ROOT)): file_hash(table)}
    for point in points:
        for key in ('peek_path', 'rollout_path'):
            inputs[point[key]] = file_hash(ROOT / point[key])
    design = dict(protocol=VERSION, status='exploratory; not frozen confirmatory results',
                  selection=__doc__, seed=args.seed,
                  candidate_points=len(pool), candidate_tasks=len({p['task_id'] for p in pool}),
                  counts=dict(points=len(points), tasks=len({p['task_id'] for p in points}),
                              fork=sum(p['old_fork'] for p in points),
                              non_fork=sum(not p['old_fork'] for p in points)),
                  intervention='Append the same truthful location sentence to both arms once, immediately '
                               'after the forced action. It then follows the original two-turn memory rule. '
                               'Do not change physical state, admissible actions, temperature, or step budget.',
                  primary='(q_revealing_native-q_other_native) - '
                          '(q_revealing_completed-q_other_completed), within original Fork stratum; '
                          'also report both arm lifts so harming the revealing arm is not called rescue.',
                  inputs=inputs, points=points)
    design['design_sha256'] = digest(design)
    write(args.out, design)
    lines = ['# Location-completion cases', '', 'Exploratory, enriched 24 Fork / 8 non-Fork by default. '
             'Pair selection does not inspect success rates or Gold.', '',
             '| Point | Original Fork | Task | Revealing action | Other action | Shared clue |',
             '|---|---|---|---|---|---|']
    for p in points:
        lines.append(f"| {p['id']} | {p['old_fork']} | {p['task']} | {p['arms']['revealing']['action']} | "
                     f"{p['arms']['other']['action']} | {p['clue']['text']} |")
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(design['counts'], 'eligible:', len(pool), 'design:', args.out, flush=True)
    if args.verify_env:
        records = verify_environment(points)
        write(args.out.with_suffix('.preflight.json'), dict(design_sha256=design['design_sha256'],
                                                           checked_arms=len(records), records=records))


if __name__ == '__main__':
    main()
