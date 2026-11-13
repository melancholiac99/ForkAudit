"""Exercise all four real games through the pinned native environment and parser."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ['VERL_AGENT_SOURCE'])
import fixed_tasks


def main():
    from agent_system.environments.env_package.alfworld import envs
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    fixed_tasks.install()
    config_path = Path(os.environ['VERL_AGENT_SOURCE']) / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    config = envs.load_config_file(str(config_path))
    base = envs.get_environment('AlfredTWEnv')(config, train_eval='train')
    records = []
    for index, task in enumerate(fixed_tasks.manifest()):
        worker = envs.AlfworldWorker(config, index, base)
        try:
            obs, info = worker.reset()
            assert info['extra.gamefile'][0] == task['native_path']
            action, valid = alfworld_projection(['<think>Inspect the room.</think><action>look</action>'],
                                               [info['admissible_commands'][0]])
            assert action == ['look'] and bool(valid[0])
            result = worker.step(action[0])
            assert isinstance(result[0][0], str)
            records.append(dict(seed=task['seed'], game_file=task['game_file'], sha256=task['sha256'],
                native_reset_and_step=True, xml_parser=True, initial_observation=obs[0]))
        finally:
            worker.env.close()
    Path(os.environ['FORKAUDIT_VERL_OUTPUT'], 'real_environment.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(real_games_passed=len(records))))


if __name__ == '__main__':
    main()
