"""Launch the pinned trainer without importing the old ALFWorld routing hooks."""
import os
from pathlib import Path
import sys


def setup_worker():
    if os.environ.get('RANK') is not None:return
    from exploratory_shop_audit.grpo_forced_audit.native import install
    install()
    from exploratory_shop_audit.grpo_forced_audit.parallel import install_collector
    install_collector()


def main():
    from .priority_evaluation import run_pending
    run_pending(os.environ.get('SHOP_GRPO_RUN'))
    setup_worker()
    import hydra
    from verl.trainer.main_ppo import run_ppo
    entry=hydra.main(config_path=str(Path(os.environ['VERL_AGENT_SOURCE'])/'verl/trainer/config'),config_name='ppo_trainer',version_base=None)(run_ppo)
    entry()

if __name__=='__main__':main()
