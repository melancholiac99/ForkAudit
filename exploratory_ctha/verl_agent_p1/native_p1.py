"""Full P1 task routing and CTHA auxiliary; native PPO/rewards remain intact."""
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def campaign():
    return Path(os.environ['FORKAUDIT_P1_CAMPAIGN'])


def schedule_task(schedule, seed, reset_index):
    if seed >= 1000:
        index = (reset_index * 16 + seed - 1000) % 80
        return schedule['tune'][index // 2], 'tune', index % 2
    assert 0 <= seed < 16 and 0 <= reset_index < 12
    return schedule['train'][reset_index][seed], 'train', None


def install_tasks():
    import agent_system.environments.env_package.alfworld.envs as module
    if getattr(module.AlfworldWorker, '_forkaudit_p1', False):
        return
    original = module.AlfworldWorker

    class ScheduledWorker(original):
        _forkaudit_p1 = True

        def __init__(self, config, seed, base_env):
            self.schedule = read(campaign() / 'tasks.json')
            self.seed, self.reset_index = int(seed), 0
            self.base = copy.copy(base_env)
            self.base.num_games = 1
            self.env = None

        def reset(self):
            game, split, rep = schedule_task(self.schedule, self.seed, self.reset_index)
            path = Path(game['native_path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == game['game_sha256']
            if self.env is not None:
                self.env.close()
            self.base.game_files = [str(path)]
            # Deterministic single-game reset, native environment semantics unchanged.
            self.env = self.base.init_env(batch_size=1)
            self.env.seed(game['seed'] + (100000 + rep if split == 'tune' else 0))
            self.reset_index += 1
            return super().reset()

    module.AlfworldWorker = ScheduledWorker
    from environment_pool import install_pool
    install_pool()


def episodes_from_batch(data, tokenizer, iteration, split):
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    groups = {}
    mapping = []
    for i in range(len(data)):
        meta = json.loads(data.non_tensor_batch['forkaudit_decision'][i])
        uid = str(data.non_tensor_batch['traj_uid'][i])
        response = data.batch['responses'][i]
        mask = data.batch['attention_mask'][i][-len(response):].bool()
        tokens = response[mask].tolist()
        assert tokens
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        action = alfworld_projection([text], [meta['admissible']])[0][0]
        reward = float(data.non_tensor_batch['episode_rewards'][i])
        assert reward in (0., 10.), 'Binary ALFWorld success mapping changed'
        if uid not in groups:
            groups[uid] = dict(episode_id=uid, game_file=meta['game_file'], task_type=meta['task_type'],
                task=meta['task'], seed=meta['seed'], rep=meta['rep'], iteration=iteration,
                won=reward > 0, steps=[], provenance=dict(split=split, iteration=iteration, dry=False,
                    source='Actual native verl-agent interaction', generating_run=os.environ.get('FORKAUDIT_VERL_OUTPUT')))
        ep = groups[uid]
        assert ep['game_file'] == meta['game_file'] and ep['won'] == (reward > 0)
        turn = meta['turn']
        assert turn == len(ep['steps']) + 1, 'Missing/reordered real interaction step'
        prompt_ids = data.batch['prompts'][i]
        prompt_mask = data.batch['attention_mask'][i][:len(prompt_ids)].bool()
        prompt = tokenizer.decode(prompt_ids[prompt_mask].tolist(), skip_special_tokens=False)
        ep['steps'].append(dict(turn=turn, prompt=prompt, response=text,
            response_token_ids=tokens, action=action, admissible=meta['admissible'], obs_before=meta['obs_before']))
        mapping.append((uid, turn))
    return list(groups.values()), mapping


def load_p0():
    from habits import load_episodes
    verified = read(ROOT / '.runtime/imports/import_verification.json')
    assert verified['verified']
    episodes = []
    train = {g['game_file'] for g in read(ROOT / 'exploratory_E4/games.json')['train']}
    for i, folder in enumerate(sorted((ROOT / '.runtime/imports/p0_base').glob('round_*'))):
        for ep in load_episodes(folder / 'rollouts', -1000 + i):
            assert ep['game_file'] in train and ep['provenance']['split'] == 'train'
            assert ep['provenance']['adapter_sha256'] is None
            ep['steps'] = ep['steps'][:8]  # Only prefix features are used for prior fitting.
            episodes.append(ep)
    assert len(episodes) == 768
    return episodes


def auxiliary(episodes, prior, plan, iteration):
    from habits_v2 import make_batch, oof_audit
    from prepare import effective_omega, validate_train
    validate_train(episodes, iteration)
    allowed = {g['game_file'] for g in read(ROOT / 'exploratory_E4/games.json')['train']}
    assert all(e['game_file'] in allowed and e['provenance']['split'] == 'train' for e in prior + episodes)
    cfg = plan['ctha']
    omega = effective_omega(cfg['omega'], iteration, cfg['anneal_start'], cfg['anneal_end'])
    rows, report = make_batch(episodes, prior, mode='ctha', base='gigpo', family='habit',
        feature_window=8, train_horizon=0, aux_horizon=8, alpha=cfg['ridge_alpha'],
        min_mixed=cfg['min_mixed'], min_pairs=cfg['min_pairs'], permutations=cfg['permutations'],
        bootstraps=cfg['bootstraps'], min_delta=cfg['min_delta'], sign_fraction=cfg['sign_fraction'],
        omega=omega, clip=cfg['clip'], baseline='legal', gamma=1., local_weight=1.)
    # NEVER use the custom trainer's episode/anchor advantage. Only the auxiliary
    # is transferred; native GiGPO's base advantage is calculated by upstream.
    values = {(r['episode_id'], r['turn']): omega*r['advantage_habit'] for r in rows}
    report['discarded_custom_base_advantage'] = True
    report['effective_nonzero_aux_rows'] = sum(v != 0 for v in values.values())
    report['current_policy_gate'] = oof_audit(episodes, family='habit', min_mixed=20,
        min_pairs=200, permutations=12, bootstraps=300)
    return values, report


def add_auxiliary(data):
    import torch
    aux = torch.as_tensor(data.non_tensor_batch['forkaudit_auxiliary'],
                          dtype=data.batch['advantages'].dtype, device=data.batch['advantages'].device)
    assert torch.isfinite(aux).all()
    addition = aux[:, None] * data.batch['response_mask']
    data.batch['advantages'] = data.batch['advantages'] + addition
    # Native GiGPO assigns the same tensor to returns; avoid in-place alias edits.
    data.batch['returns'] = data.batch['returns'] + addition
    return data


def record_summary(folder, episodes, split, index, seconds):
    by_game = {}
    for ep in episodes:
        by_game.setdefault(ep['game_file'], []).append(int(ep['won']))
    row = dict(split=split, index=index, episodes=len(episodes), success_rate=float(np.mean([e['won'] for e in episodes])),
        env_steps=sum(len(e['steps']) for e in episodes), response_tokens=sum(len(s['response_token_ids']) for e in episodes for s in e['steps']),
        seconds=seconds, per_game=by_game)
    with (folder/'episode_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    return row


def training_metrics(data, step):
    aux = data.non_tensor_batch['forkaudit_auxiliary']
    mask = data.batch['response_mask'].cpu().numpy()
    _, unique = np.unique(data.non_tensor_batch['forkaudit_row_id'], return_index=True)
    values = dict(effective_rows=int(np.count_nonzero(aux)),
        effective_unique_rows=int(np.count_nonzero(aux[unique])),
        unique_rows=len(unique), padded_rows=len(aux)-len(unique),
        token_fraction=float((mask * (aux != 0)[:, None]).sum()/mask.sum()),
        mean_abs=float(np.mean(np.abs(aux))), max_abs=float(np.max(np.abs(aux))))
    folder = Path(os.environ['FORKAUDIT_VERL_OUTPUT'])
    write(folder / f'applied_auxiliary_step_{step}.json', values)
    metrics = {'p1/aux/' + k: v for k, v in values.items()}
    report_path = folder / f'ctha_{step-1:03d}.json'
    if report_path.exists():
        report = read(report_path)
        metrics.update({'p1/aux/gate_passed': int(report['gate']['passed']),
                        'p1/aux/current_policy_gate_passed': int(report['current_policy_gate']['passed']),
                        'p1/aux/omega': report['omega_effective']})
    return metrics


def install():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector as collector
    from verl.trainer.ppo import ray_trainer
    if getattr(collector, '_forkaudit_full_p1', False):
        return
    plan = read(campaign() / 'PLAN.json')
    schedule = read(campaign() / 'tasks.json')
    game_index = {g['game_file']: g for block in schedule['train'] for g in block}
    game_index.update({g['game_file']: g for g in schedule['tune']})
    old_preprocess = collector.preprocess_single_sample

    @functools.wraps(old_preprocess)
    def preprocess(self, item, gen_batch, obs):
        result = old_preprocess(self, item, gen_batch, obs)
        envs = self._forkaudit_p1_envs
        path = str(envs.gamefile[item])
        game_file = 'json_2.1.1/' + path.split('json_2.1.1/', 1)[1]
        game = game_index[game_file]
        rep = item % 4 if self._forkaudit_p1_train else ((self._forkaudit_eval_chunks * 16 + item) % 80) % 2
        result['forkaudit_decision'] = json.dumps(dict(game_file=game_file, task_type=game['task_type'],
            seed=game['seed'], task=envs.tasks[item], rep=rep, turn=len(envs.memory[item])+1,
            obs_before=str(envs.pre_text_obs[item]), admissible=[a for a in envs.envs.get_admissible_commands[item] if a != 'help']))
        return result

    collector.preprocess_single_sample = preprocess
    old_collect = collector.multi_turn_loop

    @functools.wraps(old_collect)
    def collect(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        self._forkaudit_p1_envs, self._forkaudit_p1_train = envs, is_train
        self._forkaudit_train_round = getattr(self, '_forkaudit_train_round', 0)
        self._forkaudit_eval_chunks = getattr(self, '_forkaudit_eval_chunks', 0)
        started = time.monotonic()
        result = old_collect(self, gen_batch, actor_rollout_wg, envs, is_train=is_train)
        iteration = self._forkaudit_train_round
        split = 'train' if is_train else 'tune'
        index = iteration if is_train else self._forkaudit_eval_chunks
        episodes, mapping = episodes_from_batch(result, self.tokenizer, iteration, split)
        expected = schedule['train'][iteration] if is_train else schedule['tune'][(index % 5)*8:(index % 5+1)*8]
        counts = {g['game_file']: 0 for g in expected}
        for ep in episodes:
            counts[ep['game_file']] += 1
        assert set(counts.values()) == {4 if is_train else 2}, counts
        folder = Path(os.environ['FORKAUDIT_VERL_OUTPUT'])
        write(folder / f'episodes_{split}_{index:03d}.json', episodes)
        record_summary(folder, episodes, split, index, time.monotonic()-started)
        if is_train:
            values = {}
            if os.environ['FORKAUDIT_P1_ARM'] == 'gigpo_ctha':
                if not hasattr(self, '_forkaudit_p0'):
                    self._forkaudit_p0 = load_p0()
                history = []
                for step in range(max(0, iteration-8), iteration):
                    history += read(folder / f'episodes_train_{step:03d}.json')
                values, report = auxiliary(episodes, self._forkaudit_p0 + history, plan, iteration)
                report.update(iteration=iteration, historical_p0_episodes=768, history_episodes=len(history))
                write(folder / f'ctha_{iteration:03d}.json', report)
            result.non_tensor_batch['forkaudit_auxiliary'] = np.array([values.get(key, 0.) for key in mapping], dtype=np.float32)
            self._forkaudit_train_round += 1
        else:
            self._forkaudit_eval_chunks += 1
        return result

    collector.multi_turn_loop = collect
    original_advantage = ray_trainer.compute_advantage

    @functools.wraps(original_advantage)
    def advantage(data, *args, **kwargs):
        data = original_advantage(data, *args, **kwargs)
        if os.environ['FORKAUDIT_P1_ARM'] == 'gigpo_ctha':
            data = add_auxiliary(data)
        else:
            assert not np.any(data.non_tensor_batch['forkaudit_auxiliary'])
        return data

    ray_trainer.compute_advantage = advantage
    # Report episode-weighted Tune success, correcting native response-weighted
    # aggregation across chunks without changing outcomes or training feedback.
    original_validate = ray_trainer.RayPPOTrainer._validate

    @functools.wraps(original_validate)
    def validate(self):
        folder = Path(os.environ['FORKAUDIT_VERL_OUTPUT'])
        before = len(list(folder.glob('episodes_tune_*.json')))
        metrics = original_validate(self)
        files = sorted(folder.glob('episodes_tune_*.json'))[before:]
        assert len(files) == 5
        episodes = [ep for p in files for ep in read(p)]
        assert len(episodes) == 80
        groups = {}
        for ep in episodes:
            groups.setdefault(ep['game_file'], []).append(int(ep['won']))
        assert len(groups) == 40 and all(len(v) == 2 for v in groups.values())
        metrics['p1/tune_success_episode_mean'] = float(np.mean([ep['won'] for ep in episodes]))
        write(folder / f'tune_evaluation_{before//5:02d}.json', dict(global_step=self.global_steps,
            success_rate=metrics['p1/tune_success_episode_mean'], per_game=groups, episodes=80,
            interpretation='Development Tune only; no feedback into training'))
        return metrics

    ray_trainer.RayPPOTrainer._validate = validate
    collector._forkaudit_full_p1 = True
