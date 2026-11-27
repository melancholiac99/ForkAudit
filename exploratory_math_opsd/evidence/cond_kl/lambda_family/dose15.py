"""Incremental receipts for the registered above-one dose."""
import os
import sys
from endpoint200 import ARMS, HERE, ROOT, collect, probe
from evaluate import LONG_OUT, dynamics, run_cells
from launch import evaluate, read, save, sha

ARM = 'R-lambda15'
MODEL = ARM + '@100'
STAGES = {'new': ((ARM,), 100), 'r': (('R-extension',), 200), 'weak': (tuple(ARMS), 200)}


def main(args):
    action = args[0]
    if action == 'training':
        save(HERE/'dose15_training.json', dict(receipt=read(ROOT/'run'/ARM/'complete.json'), dynamics=dynamics((ARM,))))
    elif action == 'probe':
        arms, step = STAGES[args[1]]
        probe(int(args[2]), arms, step)
    elif action == 'collect':
        arms, step = STAGES[args[1]]
        collect(arms, step, 'dose15_probe.json' if step==100 else 'r200_probe.json')
    elif action == 'aime':
        rec = evaluate(ROOT, os.environ.copy(), MODEL, 'aime24', 1001)
        assert rec['adapter_sha256'] == read(ROOT/'run'/ARM/'complete.json')['adapters']['100']
        agreement = read(HERE/'dose15_probe.json')['metrics'][MODEL]['content_top1_agreement']['mean']
        prediction = 50.20 - .397*(100*agreement - 74.61)
        error = rec['rate'] - prediction
        save(HERE/'dose15_aime24.json', dict(receipt=rec, predicted_pct=prediction, observed_minus_predicted_pp=error,
                                              within_four_pp=abs(error)<=4, below_band=error < -4))
    elif action == 'math':
        run_cells([args[1]], 16384)
    elif action == 'openings':
        sys.path.insert(0,str(HERE.parent/'answer_openings'))
        from openings import measure
        folder = LONG_OUT/('nonthinking-'+MODEL)
        rec = read(folder/'receipt.json')
        openings = measure(MODEL,16384,False)
        assert openings['archive_sha256'] == rec['answers_sha256']
        save(HERE/'dose15_math_openings.json', dict(receipt=rec, openings=openings,
             truncated_above_20_pct=rec['truncated_pct']>20, gibberish_review='pending'))
    else:
        raise ValueError(args)


if __name__ == '__main__':
    main(sys.argv[1:])
