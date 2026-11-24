"""Run one registered evaluation cell over four disjoint, order-preserving shards."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time


def merge(rows, order):
    total = len(order)
    merged = dict(rows[0])
    parameters = ('dataset','adapter','base_model','seed','enable_thinking','temperature',
                  'top_p','top_k','min_p','presence_penalty','max_new_tokens','val_n')
    indexed = {}
    for lane, row in enumerate(rows):
        assert all(row[k] == merged[k] for k in parameters)
        assert [r['problem'] for r in row['results']] == order[lane::4]
        assert row['total_solutions'] == 12 * len(row['results'])
        indexed.update((r['problem'], r) for r in row['results'])
    assert len(indexed) == total == 30
    merged.update(results=[indexed[q] for q in order], num_problems=total, total_solutions=12*total)
    for key, denominator, percent in (
        ('pass_at_n',total,'pass_at_n_pct'), ('majority_vote_at_n',total,'majority_vote_at_n_pct'),
        ('average_at_n',12*total,'average_at_n_pct'), ('formatted_count',12*total,'format_rate')):
        merged[key] = sum(row[key] for row in rows)
        merged[percent] = 100 * merged[key] / denominator
    return merged


def run(root, env, cmd, folder, log_prefix, call, read):
    import pyarrow as pa
    import pyarrow.parquet as pq
    start = time.monotonic()
    source_index = cmd.index('--local_eval_file') + 1
    output_index = cmd.index('--output_file') + 1
    table = pq.read_table(cmd[source_index])
    folder.mkdir(parents=True, exist_ok=True)
    def worker(lane):
        shard = folder / f'lane{lane}.parquet'
        output = folder / f'lane{lane}.json'
        pq.write_table(table.take(pa.array(list(range(lane,len(table),4)))), shard)
        if not output.exists():
            args = cmd.copy()
            pending = output.with_suffix('.pending')
            args[source_index], args[output_index] = str(shard), str(pending)
            call(root,args,f'{log_prefix}-lane{lane}.log',{**env,'CUDA_VISIBLE_DEVICES':str(lane)})
            pending.replace(output)
        return read(output)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(worker,range(4)))
    column = next(k for k in ('problem','question','Problem') if k in table.column_names)
    return merge(rows,table[column].to_pylist()), time.monotonic()-start
