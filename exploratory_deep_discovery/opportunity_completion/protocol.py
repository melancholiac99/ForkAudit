"""Shared, exploratory protocol for one-time truthful location completion."""
from pathlib import Path
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import common

VERSION = 'location-completion-v1'
CONDITIONS = ('native', 'completed')
ARMS = ('revealing', 'other')


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return common.file_sha256(str(path))


def write(path, obj):
    common.write_json(str(path), obj)


def admissible(items):
    return sorted(a for a in items if a != 'help')


def post_facts(peek_file, arm):
    return (set(peek_file['state']['facts']) - set(arm['facts_removed'])) | set(arm['facts_added'])


def relation_fact(obj, location, facts):
    pat = re.compile(r'^inreceptacle\(' + re.escape(obj) + r'\s*:\s*object,\s*'
                     + re.escape(location) + r'\s*:\s*receptacle\)$')
    return next((f for f in sorted(facts) if pat.match(f)), None)


def location_clues(observation, action, target):
    """Extract only location information actually stated in a peek, never a hidden fact.

    Facts later validate the extracted statement; they never generate its contents.
    Resolve 'In it' only for an explicitly named opened receptacle.
    """
    clauses = re.findall(r'(?:On|In) the ([a-z]+ \d+), you see ([^.]+)\.', observation)
    if 'In it, you see ' in observation:
        names = re.findall(r'The ([a-z]+ \d+) is open\.', observation)
        if names:
            clauses += [(names[-1], x) for x in re.findall(r'In it, you see ([^.]+)\.', observation)]
    result = []
    for location, contents in clauses:
        for obj in sorted(set(re.findall(r'\b' + re.escape(target) + r' \d+\b', contents))):
            result.append(dict(object=obj, location=location,
                               text=f'Additional location information: {obj} is at {location}.'))
    return result


def presented_observation(raw, condition, clue):
    if condition == 'native':
        return raw
    if condition == 'completed':
        return raw + '\n\n' + clue
    raise ValueError(condition)


def seed_for(point_index, arm, rep, step):
    """New seed namespace; native/completed intentionally share random numbers."""
    assert 0 <= point_index < 10000 and 0 <= rep < 1000 and 0 <= step < 100
    return 90_000_000_000_000 + point_index * 1_000_000 + ARMS.index(arm) * 100_000 + rep * 100 + step


def load_design(path):
    design = read(path)
    if design['protocol'] != VERSION:
        raise ValueError('Unsupported protocol')
    payload = {k: v for k, v in design.items() if k != 'design_sha256'}
    if digest(payload) != design['design_sha256']:
        raise ValueError('Design changed; regenerate it explicitly into a new file')
    if len({p['id'] for p in design['points']}) != len(design['points']):
        raise ValueError('Duplicate points')
    return design


def verify_inputs(design):
    for name, expected in design['inputs'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Input changed: {name}')


def verify_environment(points):
    """CPU preflight: replay both arms, validate observations and clue truth in each."""
    import alfenv
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    env = None
    current_game = None
    records = []
    try:
        for point in points:
            game = str(Path(alfenv.ALFWORLD_DATA) / point['game_file'])
            if env is None:
                env = textworld.start(game, request_infos=textworld.EnvInfos(
                    won=True, facts=True, admissible_commands=True),
                    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
            elif current_game != game:
                env.load(game)
            current_game = game
            for name in ARMS:
                arm = point['arms'][name]
                state = env.reset()
                for action, expected in zip(point['prefix'], point['expected_prefix_observations']):
                    state, _, _ = env.step(action)
                    if state.feedback != expected:
                        raise ValueError(f"Prefix diverged: {point['id']}")
                if state.feedback != point['observation_before']:
                    raise ValueError(f"Wrong starting observation: {point['id']}")
                if admissible(state['admissible_commands']) != point['admissible_before']:
                    raise ValueError(f"Wrong starting actions: {point['id']}")
                state, _, done = env.step(arm['action'])
                facts = {str(f) for f in state['facts']}
                if state.feedback != arm['observation_after'] or bool(done) or bool(state['won']):
                    raise ValueError(f"Forced arm mismatch or terminal: {point['id']} {name}")
                if admissible(state['admissible_commands']) != arm['admissible_after']:
                    raise ValueError(f"Post-arm actions changed: {point['id']} {name}")
                if digest(sorted(facts)) != arm['post_facts_sha256']:
                    raise ValueError(f"Post-arm facts changed: {point['id']} {name}")
                if point['clue']['fact'] not in facts:
                    raise ValueError(f"Clue is not true: {point['id']} {name}")
                records.append(dict(point=point['id'], arm=name, clue_true=True, prefix_matches=True))
            print(f"preflight {point['id']}: both arms match; clue true", flush=True)
    finally:
        if env is not None:
            env.close()
    return records
