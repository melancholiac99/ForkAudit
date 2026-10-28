"""Post-hoc CPU witness of recoverability, not an evaluated continuation policy.

Among the held-object affordance checks, inspect both pick_and_place_simple
cases. Use task/clue metadata to construct a transparent short legal solution.
This establishes existence within budget only; it does not estimate how often
an LLM would discover the repair or establish a population recovery rate.
"""
from pathlib import Path
import re
from protocol import HERE, ROOT, read, write


def main():
    import alfenv
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    design = read(HERE / 'design.json')
    checks = read(HERE / 'affordance_checks.json')['records']
    selected = [(p, r) for p in design['points'] for r in checks
                if p['id'] == r['point_id'] and p['task_type'] == 'pick_and_place_simple']
    records = []
    env = None
    try:
        for p, r in selected:
            game = str(Path(alfenv.ALFWORLD_DATA) / p['game_file'])
            if env is None:
                env = textworld.start(game, request_infos=textworld.EnvInfos(
                    won=True, facts=True, admissible_commands=True),
                    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
            else:
                env.load(game)
            state = env.reset()
            for action, expected in zip(p['prefix'], p['expected_prefix_observations']):
                state, _, _ = env.step(action)
                assert state.feedback == expected
            state, _, done = env.step(p['arms'][r['arm']]['action'])
            assert state.feedback == p['arms'][r['arm']]['observation_after'] and not done
            destination = re.fullmatch(r'put a [a-z]+ in ([a-z]+)\.', p['task']).group(1) + ' 1'
            clue = p['clue']
            nav = f"go to {clue['location']}"
            plan = ([nav] if nav in state['admissible_commands'] else []) + [
                f"move {r['held']} to {clue['location']}",
                f"take {clue['object']} from {clue['location']}",
                f"go to {destination}",
                f"move {clue['object']} to {destination}"]
            transitions = []
            for action in plan:
                legal = action in state['admissible_commands']
                assert legal, (p['id'], action, state['admissible_commands'])
                state, _, done = env.step(action)
                transitions.append(dict(action=action, was_admissible=legal,
                    observation_after=state.feedback, won=bool(state['won'])))
                if done:
                    break
            final_step = p['turn'] + len(transitions)
            row = dict(point_id=p['id'], old_regime=p['old_regime'], task=p['task'],
                arm=r['arm'], forced_action=p['arms'][r['arm']]['action'],
                scripted_transitions=transitions, repair_steps=len(transitions),
                final_episode_step=final_step, within_budget=final_step <= 50,
                won=bool(state['won']))
            assert row['within_budget']
            records.append(row)
            print(row, flush=True)
    finally:
        if env is not None:
            env.close()
    write(HERE / 'scripted_recoverability.json', dict(
        status='post-hoc CPU existence witnesses', design_sha256=design['design_sha256'],
        selection=__doc__, records=records,
        limitation='Scripted plans use task/clue metadata. No repaired LLM rollout, '
                   'no estimated repair success rate, no claim about other Dead states.'))


if __name__ == '__main__':
    main()
