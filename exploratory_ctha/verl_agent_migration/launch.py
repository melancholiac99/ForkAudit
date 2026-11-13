"""Launch pinned upstream training with explicit task and compatibility adapters."""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, os.environ['VERL_AGENT_SOURCE'])
    sys.path.insert(0, str(HERE))
    setup_worker()
    # Ray imports TaskRunner in a fresh process: a runtime setup hook installs adapters there.
    import hydra
    from verl.trainer.main_ppo import run_ppo
    # The archive's config/ is a data directory, not an importable package.
    # Use its absolute path when launching from this external module.
    entrypoint = hydra.main(config_path=str(Path(os.environ['VERL_AGENT_SOURCE']) / 'verl/trainer/config'),
                           config_name='ppo_trainer', version_base=None)(run_ppo)
    entrypoint()


def setup_worker():
    # GPU rank processes need only the native actor/rollout engine. Importing
    # trainer/env adapters in their pre-start hook can initialize CUDA before
    # Ray finishes applying the per-actor device visibility assignment.
    if os.environ.get('RANK') is not None:
        return
    import fixed_tasks
    import correction
    fixed_tasks.install()
    correction.install()
    import reference_probe
    reference_probe.install()


if __name__ == '__main__':
    main()
