"""Task-selection adapter only; upstream environment/reward/PPO remain unchanged."""
import json
import os
from pathlib import Path


def manifest():
    return json.loads(Path(os.environ['FORKAUDIT_VERL_TASKS']).read_text())['tasks']


def install():
    import agent_system.environments.env_package.alfworld.envs as module
    if getattr(module.AlfworldWorker, '_forkaudit_fixed', False):
        return
    original = module.AlfworldWorker

    class FixedTaskWorker(original):
        _forkaudit_fixed = True

        def __init__(self, config, seed, base_env):
            import copy
            tasks = manifest()
            game = tasks[int(seed) % len(tasks)]
            base_env = copy.copy(base_env)
            base_env.game_files = [game['native_path']]
            base_env.num_games = 1
            super().__init__(config, seed, base_env)
            self.forkaudit_task = game

    # Defined in driver, serialized by Ray as an actor class; only selection differs.
    module.AlfworldWorker = FixedTaskWorker

