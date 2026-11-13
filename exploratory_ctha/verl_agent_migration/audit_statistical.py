"""Independent NumPy audit of real-token statistical admission evidence."""
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
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    report = json.loads((run / 'reference_probe.json').read_text())
    checks = []
    files = []
    for row in report['checks']:
        path = run / f"reference_batch_{row['batch_size']}.npz"
        with np.load(path) as data:
            for reference, key in [('reference_actor', 'versus_original_GPU_HF'),
                                   ('rollout', 'versus_original_vllm_raw_scores')]:
                stats, extra = recompute(data['actor'], data[reference], data['mask'])
                for field, actual in stats.items():
                    assert math.isclose(actual, row[key][field], rel_tol=1e-9, abs_tol=1e-10), (path, field)
                assert stats['sampled_behavior_k3'] <= .02 and stats['clip_fraction'] <= .30
                checks.append(dict(batch=row['batch_size'], reference=reference, **stats, **extra))
        files.append(path)
    path = run / 'fresh_eager_alignment.npz'
    with np.load(path) as data:
        stats, extra = recompute(data['actor'], data['rollout'], data['mask'])
        for field, actual in stats.items():
            assert math.isclose(actual, report['fresh_real_context_sampling'][field], rel_tol=1e-9, abs_tol=1e-10)
        assert stats['sampled_behavior_k3'] <= .02 and stats['clip_fraction'] <= .30
        checks.append(dict(batch='fresh_eager', reference='actual_sampler', **stats, **extra))
    files.append(path)
    restore = json.loads((run / 'restore_rank_0.json').read_text())
    unchanged = json.loads((run / 'gradient_benchmark_no_update.json').read_text())
    bench = json.loads((run / 'gradient_benchmark.json').read_text())
    assert restore['passed'] and unchanged['passed'] and unchanged['before'] == unchanged['after']
    assert json.loads((run / 'exit.json').read_text())['returncode'] == 0
    memory = list(csv.DictReader((run / 'gpu_memory.csv').open()))
    during = [r for r in memory if bench['started_unix'] <= float(r['unix_time']) <= bench['finished_unix']]
    result = dict(statistical_admission='PASS', Engineering='PASS',
        scope='Native BF16 statistical-admission protocol only; not original CTHA engineering checklist',
        Learning='NOT_RUN', Generalization='NOT_RUN', CTHA_online_efficacy='NOT_TESTED',
        exact_greedy_trajectory_reproduction='NOT_REPRODUCED; original records retained',
        optimizer_updates=0, actual_checkpoint_restore='PASS', nonupdating_backward_state_hashes='PASS',
        configuration_sha256=json.loads((run / 'registration.json').read_text())['configuration_sha256'],
        observed_nvml_peak_mib=max(float(r['used_mib']) for r in memory),
        observed_backward_peak_mib=max(float(r['used_mib']) for r in during),
        probability_checks=checks,
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (run / 'independent_statistical_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['probability_checks','files_sha256']}, indent=2))


if __name__ == '__main__':
    main()
