"""Mocked failure/repair/shutdown integration tests; never starts an agent or shuts down."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch
import repair_guardian as G


def state(done=False, failure=False, hold=False):
    return dict(states={k: dict(stage='completed_verified' if done else ('failed' if failure else 'running')) for k in ('E4', 'P8')},
                processes=[], reasons=['simulated error'] if failure else [], both_done=done, hold=hold)


def scenario(kind):
    calls, iterations = [], [0]
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        r = Path(tmp); cfg = r/'config.json'
        G.save(cfg, dict(authorization='repair_on_failure_then_archive_shutdown_if_unrecoverable', guardian_sha256=G.COMPLETE.digest(G.__file__), max_attempts_total=2))
        stack.enter_context(patch.object(G, 'R', r)); stack.enter_context(patch.object(G, 'CONFIG', cfg))
        stack.enter_context(patch.object(G.time, 'sleep', lambda _: None))
        stack.enter_context(patch.object(G, 'status', lambda *a, **k: None))
        stack.enter_context(patch.object(G.subprocess, 'check_output', lambda *a, **k: ''))
        stack.enter_context(patch.object(G, 'archive', lambda label, evidence: calls.append(('archive', label)) or r))
        stack.enter_context(patch.object(G, 'shutdown', lambda cfg, reason, **kw: calls.append(('shutdown', reason))))

        def inspect():
            iterations[0] += 1
            assert iterations[0] < 20, 'infinite loop'
            if kind == 'success': return state(done=True)
            if kind == 'held': return state(hold=True) if iterations[0] == 1 else state(done=True)
            if kind == 'recovered': return state(done=any(c[0] == 'repair' for c in calls), failure=not any(c[0] == 'repair' for c in calls))
            if kind == 'bad_final_audit': return state(done=True)
            return state(failure=True)

        def repair(cfg, evidence, attempt):
            calls.append(('repair', attempt))
            return kind == 'recovered', r

        def validate():
            if kind == 'bad_final_audit': raise AssertionError('simulated independent audit mismatch')
            return dict(passed=True)

        stack.enter_context(patch.object(G, 'inspect', inspect))
        stack.enter_context(patch.object(G, 'repair', repair))
        stack.enter_context(patch.object(G.COMPLETE, 'validate_results', validate))
        G.main()
    return calls


def main():
    results = {}
    for kind in ('success', 'held', 'recovered', 'unrecoverable', 'bad_final_audit'):
        calls = scenario(kind)
        shutdowns = [c for c in calls if c[0] == 'shutdown']
        repairs = [c for c in calls if c[0] == 'repair']
        assert len(shutdowns) == 1
        failed = kind in ('unrecoverable', 'bad_final_audit')
        assert shutdowns[0][1] == ('repair_unsuccessful_logs_saved' if failed else 'both_experiments_verified')
        assert len(repairs) == (2 if failed else 1 if kind == 'recovered' else 0)
        if failed:
            assert ('archive', 'unrecoverable_final') in calls
            assert calls.index(('archive', 'unrecoverable_final')) < calls.index(shutdowns[0])
        results[kind] = dict(passed=True, calls=calls)
    G.save(G.R/'repair_selftest.json', dict(passed=True, simulations=results, real_shutdown_called=False, real_agent_called=False))
    print('Five failure/recovery/shutdown scenarios passed; no real shutdown or repair invocation.')


if __name__ == '__main__':
    main()
