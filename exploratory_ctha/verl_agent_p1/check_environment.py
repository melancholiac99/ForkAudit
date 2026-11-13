"""Exercise real scheduled ALFWorld reset/step routing on CPU, without policy sampling."""
import json
import os
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent),os.environ['VERL_AGENT_SOURCE']]
from run import select_games
from native_p1 import install_tasks
import agent_system.environments.env_package.alfworld.envs as module


def main():
    games=json.loads((ROOT/'exploratory_E4/games.json').read_text())
    def local(game):return dict(game,native_path=str(ROOT/'.runtime/alfworld'/game['game_file']))
    schedule=dict(train=[[local(g) for g in select_games(games['train'],0,i,16)] for i in range(12)],tune=[local(g) for g in games['tune']])
    with tempfile.TemporaryDirectory(prefix='p1_env_',dir=ROOT/'.runtime') as temporary:
        Path(temporary,'tasks.json').write_text(json.dumps(schedule))
        os.environ['FORKAUDIT_P1_CAMPAIGN']=temporary
        install_tasks()
        cfg=module.load_config_file(str(Path(os.environ['VERL_AGENT_SOURCE'])/'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'))
        base=module.get_environment('AlfredTWEnv')(cfg,train_eval='train')
        checks=[]
        for seed in [0,15,1000,1001,1015]:
            worker=module.AlfworldWorker(cfg,seed,base)
            try:
                for counter in range(2):
                    obs,info=worker.reset()
                    expected=(schedule['train'][counter][seed] if seed<1000 else schedule['tune'][(counter*16+seed-1000)//2])
                    actual=info['extra.gamefile'][0]
                    assert str(actual).endswith(expected['game_file']), (actual,expected)
                    assert 'Your task is to: ' in obs[0] and info['admissible_commands'][0]
                    result=worker.step('look')
                    assert len(result)==4
                    checks.append(dict(seed=seed,reset=counter,game_file=expected['game_file'],passed=True))
            finally:
                worker.env.close()
    (HERE/'environment_checks.json').write_text(json.dumps(dict(passed=True,checks=checks,scope='Real environment routing only, no model sampling'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,resets=len(checks))))


if __name__=='__main__':main()
