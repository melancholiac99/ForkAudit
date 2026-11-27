"""Eight sequential two-sample MATH-500 evaluations using the archived checkpoints."""
import fcntl
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from pathlib import Path

from four_engines import run
from launch import call, read, save, sha
from math500_trajectory import DATA, DIGEST, ROOT, prepare

CELLS = [(True, m) for m in ('B', 'R@100', 'P@100', 'Rres@100')] + [
    (False, m) for m in ('B', 'R@100', 'T0@100', 'P@100')]
OUT = ROOT / 'run/math500-long-seed1001-n2'
REPORT = Path(__file__).resolve().parent / 'evidence/math500_long/results.json'


def evaluate_cell(thinking, model, nonthinking_budget, rows, hashes, lane_call, start_times=None):
    key = ('thinking-' if thinking else 'nonthinking-') + model
    folder = OUT / key
    folder.mkdir(exist_ok=True)
    budget, temp, top_p, top_k = (32768, 1., .95, -1) if thinking else (nonthinking_budget, .7, .8, 20)
    adapter = None if model == 'B' else str(ROOT/'run'/model.split('@')[0]/('checkpoint-'+model.split('@')[1]))
    config = dict(enable_thinking=thinking, adapter=adapter, temperature=temp, top_p=top_p,
                  top_k=top_k, min_p=0., presence_penalty=0., max_new_tokens=budget, seed=1001, val_n=2)
    identity = dict(config=config, data_sha256=DIGEST, adapter_sha256=hashes.get(model),
                    evaluator_sha256=sha(ROOT/'source/eval/evaluate_math.py'))
    if (folder/'identity.json').exists():
        assert read(folder/'identity.json') == identity
    else:
        save(folder/'identity.json', identity)
    dest = folder/'answers.json'
    receipt = folder/'receipt.json'
    if receipt.exists():
        entry = read(receipt)
        assert sha(dest) == entry['answers_sha256']
    else:
        cmd = [sys.executable, str(Path(__file__).with_name('verified_eval.py')),
               '--base_model', str(ROOT/'models/Qwen3-1.7B'), '--dataset', 'math500',
               '--local_eval_file', str(DATA/'test.parquet'), '--output_file', str(dest),
               '--enable_thinking' if thinking else '--no_thinking', '--temperature', str(temp),
               '--top_p', str(top_p), '--top_k', str(top_k), '--min_p', '0', '--presence_penalty', '0',
               '--max_new_tokens', str(budget), '--val_n', '2', '--seed', '1001', '--tensor_parallel_size', '1']
        if adapter:
            cmd += ['--checkpoint_dir', adapter]
        x, seconds = run(ROOT, os.environ.copy(), cmd, folder/'shards', 'long-math-'+key,
                         lane_call, read, questions=500, samples=2)
        assert all(x[k] == v for k,v in config.items())
        assert [r['problem'] for r in rows] == [r['problem'] for r in x['results']]
        for lane in range(4):
            proof = read(folder/'shards'/f'lane{lane}.pending.adapter.json')
            assert proof['sha256'] == hashes.get(model) and proof['adapter'] == adapter
        assert all(len(r['generations']) == 2 for r in x['results'])
        gs = [g for r in x['results'] for g in r['generations']]
        assert len(gs) == 1000 and all(0 <= g['generated_tokens'] <= budget for g in gs)
        assert sum(g['correct'] for g in gs) == x['average_at_n']
        save(dest, x)
        if start_times:
            seconds = time.monotonic() - min(start_times)
        entry = dict(**identity, answers=len(gs), accuracy_percent=x['average_at_n_pct'],
                     mean_tokens=sum(g['generated_tokens'] for g in gs)/len(gs),
                     truncated_pct=sum(g['finish_reason']=='length' for g in gs)*100/len(gs),
                     wall_seconds=seconds, answers_sha256=sha(dest))
        save(receipt, entry)
    return key, entry


def pipeline_cells(cells, execute):
    """Process the same shard on the same GPU; each lane advances in model order."""
    released = [[Event() for _ in range(4)] for _ in cells]
    failed = Event()

    def worker(index):
        starts = []

        def dispatch(root, cmd, log, env):
            lane = int(env['CUDA_VISIBLE_DEVICES'])
            if index:
                released[index-1][lane].wait()
            if failed.is_set():
                raise RuntimeError('An earlier evaluation failed')
            starts.append(time.monotonic())
            try:
                return call(root, cmd, log, env)
            except BaseException:
                failed.set()
                raise
            finally:
                released[index][lane].set()

        try:
            return execute(cells[index], dispatch, starts)
        except BaseException:
            failed.set()
            raise
        finally:
            # Existing receipts or shard files skip dispatch. Maintain the
            # predecessor dependency even when a whole cached cell returns early.
            for lane, event in enumerate(released[index]):
                if index:
                    released[index-1][lane].wait()
                event.set()

    with ThreadPoolExecutor(max_workers=len(cells)) as pool:
        futures = [pool.submit(worker,i) for i in range(len(cells))]
        for future in as_completed(futures):
            yield future.result()


def main(nonthinking_budget=16384, pipeline=False):
    rows, hashes = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = {}

        def execute(cell, dispatch, starts):
            return evaluate_cell(*cell, nonthinking_budget, rows, hashes, dispatch, starts)

        if pipeline:
            completed = pipeline_cells(CELLS, execute)
        else:
            completed = (execute(cell, call, None) for cell in CELLS)
        for key, entry in completed:
            summary[key] = entry
            save(REPORT, summary)


if __name__ == '__main__':
    main()
