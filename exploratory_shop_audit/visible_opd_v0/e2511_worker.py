"""Run independent native actor jobs side by side, one GPU per arm."""
import json
import sys

from . import d_collect, d_eval
from .e25_layout import Layout
from .e25_probe import configure


def main():
    phase, payload = sys.argv[1:]
    rank = configure()
    jobs = json.loads(payload)
    if phase == 'targets':
        from .e2511_targets import score_shard
        score_shard(rank,jobs)
        return
    slots = {'R': 0, 'P': 1, 'R-matched': 2}
    assigned = [job for job in jobs if slots[job[2]] == rank]
    if not assigned:
        return
    assert len(assigned) == 1
    run, step, arm = assigned[0]
    layout = Layout(run)
    if phase == 'collect':
        d_collect.collect(step, rank, arm, layout.folder, layout.state)
    elif phase == 'eval':
        assert run == 'formal'
        d_eval.run(step, rank, layout=layout, arm_override=arm)
    else:
        raise ValueError(phase)


if __name__ == '__main__':
    main()
