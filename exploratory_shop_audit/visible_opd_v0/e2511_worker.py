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
    if rank >= len(jobs):
        return
    run, step, arm = jobs[rank]
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
