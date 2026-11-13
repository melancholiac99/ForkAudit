"""Audit the zero-update ALFWorld session integration evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_run import recompute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    read = lambda name: json.loads((run / name).read_text())
    assert read('exit.json')['returncode'] == 0
    restore = read('restore_rank_0.json')
    assert restore['passed']
    sessions = [json.loads(line) for line in (run / 'rollout_sessions.jsonl').read_text().splitlines()]
    assert len(sessions) == 1
    session = sessions[0]
    assert session['completed'] and session['manager_enters'] == session['manager_exits'] == 1
    assert not any(session['final_state'][key] for key in ['open', 'resident', 'poisoned'])
    path = run / 'validation_alignment_0001.npz'
    with np.load(path) as data:
        actual, extra = recompute(data['actor'], data['rollout'], data['mask'])
    reported = read('validation_alignment_0001.json')
    for key, value in actual.items():
        assert math.isclose(value, reported[key], rel_tol=1e-9, abs_tol=1e-10), key
    assert actual['sampled_behavior_k3'] <= .02 and actual['clip_fraction'] <= .30
    memory = list(csv.DictReader((run / 'gpu_memory.csv').open()))
    result = dict(
        passed=True, scope='Native four-task val-only ALFWorld rollout; not a training throughput comparison',
        session=session, probability_checks=actual, diagnostics=extra,
        interpretation=reported['interpretation'],
        evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        restored_state=restore['actual'],
        peak_nvml_mib=max(float(row['used_mib']) for row in memory),
        Learning='NOT_RUN', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')
    (run / 'independent_session_environment_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
