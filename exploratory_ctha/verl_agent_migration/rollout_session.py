"""Explicit TP1 rollout residency. Native sync and RNG transitions stay intact."""
import functools
import os
import time


class Session:
    def __init__(self, enter, leave):
        self.enter_native, self.leave_native = enter, leave
        self.open = self.resident = self.poisoned = False
        self.enters = self.exits = 0
        self.enter_seconds = self.exit_seconds = 0.

    def begin(self):
        assert not self.open and not self.resident and not self.poisoned
        self.open = True

    def enter(self):
        assert not self.poisoned
        if self.resident:
            assert self.open
            return
        start = time.monotonic()
        try:
            self.enter_native()
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.enter_seconds += time.monotonic() - start
        self.enters += 1
        self.resident = True

    def leave(self, exc_type=None, exc=None, tb=None, force=False):
        if not self.resident or (self.open and exc_type is None and not force):
            return
        start = time.monotonic()
        try:
            self.leave_native(exc_type, exc, tb)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.exit_seconds += time.monotonic() - start
            self.resident = False
        self.exits += 1

    def end(self):
        try:
            self.leave(force=True)
        finally:
            self.open = False

    def assert_closed(self):
        assert not (self.open or self.resident or self.poisoned), 'Close healthy rollout session before scoring/update/checkpoint'


def state(worker):
    manager = worker.rollout_sharding_manager
    if not hasattr(manager, '_forkaudit_session'):
        manager._forkaudit_session = Session(lambda: manager._forkaudit_original_enter(),
            lambda *args: manager._forkaudit_original_exit(*args))
    return manager._forkaudit_session


def begin(worker):
    assert worker.world_size == 1 and not worker._is_offload_param
    state(worker).begin()
    return status(worker)


def end(worker):
    state(worker).end()
    return status(worker)


def status(worker):
    import torch
    torch.cuda.synchronize()
    s = state(worker)
    return dict(open=s.open, resident=s.resident, poisoned=s.poisoned,
                enters=s.enters, exits=s.exits, enter_seconds=s.enter_seconds, exit_seconds=s.exit_seconds)


def control(group, fn):
    assert group.world_size == 1, 'Session adapter is admitted only for TP1'
    return group.execute_all_sync('actor_rollout_execute_with_func_generator', fn)[0]


def install_worker():
    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager as manager
    from verl.workers.fsdp_workers import ActorRolloutRefWorker as worker
    if getattr(manager, '_forkaudit_residency', False):
        return
    manager._forkaudit_original_enter = manager.__enter__
    manager._forkaudit_original_exit = manager.__exit__

    def get(m):
        if not hasattr(m, '_forkaudit_session'):
            m._forkaudit_session = Session(lambda: m._forkaudit_original_enter(),
                lambda *args: m._forkaudit_original_exit(*args))
        return m._forkaudit_session

    manager.__enter__ = lambda m: get(m).enter()
    manager.__exit__ = lambda m, *args: get(m).leave(*args)
    for name in ['compute_log_prob', 'compute_ref_log_prob', 'update_actor', 'save_checkpoint', 'load_checkpoint']:
        original = getattr(worker, name, None)
        if original is None:
            continue
        def wrap(fn):
            @functools.wraps(fn)
            def guarded(self, *args, **kwargs):
                state(self).assert_closed()
                return fn(self, *args, **kwargs)
            return guarded
        setattr(worker, name, wrap(original))
    manager._forkaudit_residency = True


def install_collector():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    old = TrajectoryCollector.vanilla_multi_turn_loop
    if getattr(old, '_forkaudit_residency', False):
        return
    @functools.wraps(old)
    def collect(self, gen_batch, actor_rollout_wg, envs):
        import json
        from pathlib import Path
        started = time.monotonic()
        initial = control(actor_rollout_wg, begin)
        completed = False
        try:
            result = old(self, gen_batch, actor_rollout_wg, envs)
            completed = True
            return result
        finally:
            final = control(actor_rollout_wg, end)
            folder = os.environ.get('FORKAUDIT_VERL_OUTPUT')
            if folder:
                record = dict(seconds=time.monotonic()-started, completed=completed,
                    manager_enters=final['enters']-initial['enters'],
                    manager_exits=final['exits']-initial['exits'], final_state=final)
                with (Path(folder)/'rollout_sessions.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
    collect._forkaudit_residency = True
    TrajectoryCollector.vanilla_multi_turn_loop = collect
