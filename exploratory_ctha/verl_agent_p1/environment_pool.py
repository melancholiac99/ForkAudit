"""Batch native text environments into fewer Ray processes, preserving slot order."""
import copy


class LocalBatch:
    def __init__(self, worker_class, config, seeds, base_env):
        self.slots = [worker_class(config, seed, copy.copy(base_env)) for seed in seeds]

    def reset(self):
        return [slot.reset() for slot in self.slots]

    def step(self, actions):
        assert len(actions) == len(self.slots)
        return [slot.step(action) for slot, action in zip(self.slots, actions)]

    def close(self):
        for slot in self.slots:
            if slot.env is not None:
                slot.env.close()


def install_pool():
    import numpy as np
    import ray
    import agent_system.environments.env_package.alfworld.envs as module
    original = module.AlfworldEnvs
    if getattr(original, '_forkaudit_pooled', False):
        return

    class PooledEnvs(original):
        _forkaudit_pooled = True

        def __init__(self, alf_config_path, seed, env_num, group_n,
                     resources_per_worker, is_train=True, env_kwargs=None):
            env_kwargs = env_kwargs or {}
            config = module.load_config_file(alf_config_path)
            assert config['env']['type'] == 'AlfredTWEnv'
            base = module.get_environment('AlfredTWEnv')(
                config, train_eval='train' if is_train else env_kwargs.get('eval_dataset','train'))
            self.multi_modal = False
            self.num_processes = env_num * group_n  # Native name means logical slots.
            self.group_n = group_n
            self.prev_admissible_commands = [None] * self.num_processes
            worker = ray.remote(num_cpus=1)(LocalBatch)
            self.workers = []
            self.ranges = []
            for start in range(0, self.num_processes, 16):
                stop = min(start+16, self.num_processes)
                seeds = [seed + i//group_n for i in range(start,stop)]
                self.workers.append(worker.remote(module.AlfworldWorker, config, seeds, base))
                self.ranges.append((start,stop))

        def reset(self):
            groups = ray.get([worker.reset.remote() for worker in self.workers])
            results = [result for group in groups for result in group]
            texts, infos = [], []
            assert len(results) == self.num_processes
            for i, (obs, info) in enumerate(results):
                info = {key: value[0] for key,value in info.items()}
                texts.append(obs[0]);infos.append(info)
                self.prev_admissible_commands[i] = info['admissible_commands']
            return texts, None, infos

        def step(self, actions):
            assert len(actions) == self.num_processes
            groups = ray.get([worker.step.remote(actions[start:stop])
                for worker,(start,stop) in zip(self.workers,self.ranges)])
            results = [result for group in groups for result in group]
            texts, rewards, dones, infos = [], [], [], []
            assert len(results) == self.num_processes
            for i,(obs,scores,done,info) in enumerate(results):
                info = {key:value[0] for key,value in info.items()}
                texts.append(obs[0]);dones.append(done[0]);infos.append(info)
                self.prev_admissible_commands[i] = info['admissible_commands']
                rewards.append(module.compute_reward(info,False))
            return texts, None, rewards, dones, infos

        def close(self):
            ray.get([worker.close.remote() for worker in self.workers])
            for worker in self.workers:
                ray.kill(worker)

    module.AlfworldEnvs = PooledEnvs
