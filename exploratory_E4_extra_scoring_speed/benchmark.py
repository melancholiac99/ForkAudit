"""Compare candidate batching with frozen P4 backend. Outputs are engineering only."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import statistics
import time
import types
import numpy as np
import torch
from batched import score_batched

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = '/root/autodl-tmp/models/Qwen2.5-32B-Instruct'


def reference_module():
    path = ROOT / 'exploratory_P4/p4_e2_backend.py'
    spec = importlib.util.spec_from_file_location('frozen_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probabilities(rows):
    scores = np.array([sum(row) for row in rows]); p = np.exp(scores - scores.max())
    return p / p.sum()


def tv(a, b):
    return float(np.abs(probabilities(a) - probabilities(b)).sum() / 2)


def selftest():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = 'sdpa'
    ref = types.SimpleNamespace()
    # CPU reference follows exactly the original cache algorithm, without hardcoded CUDA.
    ref.t = torch
    ref.model = Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    ref.model.lm_head.to(dtype=torch.float32)
    import copy
    def sequential(job):
        pre = job['prefix_ids']; m = ref.model
        with torch.inference_mode():
            out = m.model(input_ids=torch.tensor([pre]), use_cache=True)
            first = torch.log_softmax(m.lm_head(out.last_hidden_state[0, -1].float()), -1)
            rows = []
            for candidate in job['candidates']:
                cmd = candidate['suffix_ids']; kv = copy.deepcopy(out.past_key_values)
                h = m.model(input_ids=torch.tensor([cmd[:-1]]), past_key_values=kv, use_cache=False).last_hidden_state
                lp = torch.log_softmax(m.lm_head(h[0].float()), -1)
                rows.append([float(first[cmd[0]])] + lp[torch.arange(len(cmd)-1), torch.tensor(cmd[1:])].tolist())
            return rows
    checks = []
    for seed, prefix_len in enumerate([1, 7, 31, 80]):
        rng = np.random.default_rng(seed)
        job = dict(prefix_ids=rng.integers(1, 97, prefix_len).tolist(),
                   candidates=[dict(suffix_ids=rng.integers(1, 97, length).tolist()) for length in [2, 3, 7, 5, 11]])
        baseline = sequential(job)
        for batch in [1, 2, 5]:
            result, actual = score_batched(ref, job, batch)
            distance = tv(baseline, result)
            assert distance < .005, (seed, batch, distance)
            assert [len(v) for v in result] == [len(v) for v in baseline]
            checks.append(dict(prefix_tokens=prefix_len, batch=batch, tv=distance, actual_batches=actual))
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, scope='Tiny random CPU model only; not evidence of 32B equivalence', checks=checks)
    (HERE / 'cpu_selftest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def tokenize(job, tok):
    prefix = tok.apply_chat_template([dict(role='user', content=job['text'])], tokenize=False,
                                     add_generation_prompt=True) + '<action>'
    pre = tok.encode(prefix, add_special_tokens=False)
    full = [tok.encode(prefix+a+'</action>'+tok.eos_token, add_special_tokens=False) for a in job['actions']]
    start = len(pre)
    for ids in full:
        i = 0
        while i < min(len(pre), len(ids)) and pre[i] == ids[i]:
            i += 1
        start = min(start, i)
    assert len(pre)-start <= 3 and max(map(len, full)) <= 32768
    assert all(ids[:start] == full[0][:start] and ids[-1] == tok.eos_token_id for ids in full)
    return dict(**job, prefix_ids=full[0][:start], candidates=[dict(suffix_ids=ids[start:]) for ids in full])


def benchmark():
    from transformers import AutoTokenizer
    formal = ROOT / 'exploratory_E4/data/judge'
    completed = json.loads((formal/'32b_F/run.json').read_text())
    assert completed['complete'], 'Wait for formal F to finish'
    source = ROOT/'exploratory_P4/p4_e2_backend.py'
    original = json.loads((ROOT/'exploratory_P4/e2_32b/run.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original['scorer_sha256'] == completed['backend_sha256']
    raw = gzip.decompress((formal/'requests.jsonl.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == completed['requests_sha256']
    jobs = [json.loads(line) for line in raw.decode().splitlines()]
    selected = []
    for condition in ['F', 'L']:
        group = sorted([j for j in jobs if j['condition'] == condition], key=lambda j: (len(j['text']), j['request_id']))
        for q in [0, .5, .9, 1]:
            selected.append(group[round(q*(len(group)-1))])
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    selected = [tokenize(job, tok) for job in selected]
    module = reference_module(); module.MODEL = MODEL; scorer = module.Scorer()
    results = []
    def timed(function):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return value, time.perf_counter()-start, torch.cuda.max_memory_allocated()
    scorer.score(selected[0])  # warm baseline kernels, excluded from timing
    score_batched(scorer, selected[0], 5)  # warm batched kernels
    for i, job in enumerate(selected):
        record = dict(request_id=job['request_id'], request_sha256=job['request_sha256'],
                      condition=job['condition'], prefix_tokens=len(job['prefix_ids']), trials=[])
        reference = scorer.score(job)
        if i in [0, 3, 4, 7]:
            direct = [scorer.direct(job, k) for k in range(5)]
            record['reference_vs_direct_tv'] = tv(reference, direct)
            assert record['reference_vs_direct_tv'] < .005
        for repeat in range(2):
            # Alternate order to reduce simple warmup/order bias.
            for size in ([0, 2, 5] if repeat == 0 else [5, 2, 0]):
                if size == 0:
                    value, seconds, peak = timed(lambda: (scorer.score(job), [1]*5))
                else:
                    value, seconds, peak = timed(lambda: score_batched(scorer, job, size))
                scores, actual = value
                distance = tv(reference, scores)
                argmax_same = int(np.argmax(probabilities(reference))) == int(np.argmax(probabilities(scores)))
                max_logprob_error = max(abs(a-b) for row_a, row_b in zip(reference, scores) for a, b in zip(row_a, row_b))
                record['trials'].append(dict(batch=size or 1, baseline=size == 0, repeat=repeat,
                                             seconds=seconds, peak_allocated_bytes=peak,
                                             actual_batches=actual, tv=distance, passed=distance < .005,
                                             argmax_same=argmax_same, max_token_logprob_error=max_logprob_error))
                assert distance < .005, (job['request_id'], size, distance)
        results.append(record)
        (HERE/'benchmark_progress.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    summary = {}
    for condition in ['F', 'L']:
        group = [r for r in results if r['condition'] == condition]
        times = {b: sum(statistics.median(t['seconds'] for t in r['trials'] if t['batch']==b) for r in group) for b in [1, 2, 5]}
        summary[condition] = dict(sample_seconds=times, speedup_batch2=times[1]/times[2], speedup_batch5=times[1]/times[5])
    out = dict(passed=True, tv_threshold=.005, selection='text-length quantiles 0, .5, .9, 1 per condition, no outcome-based selection',
               scope='Exploratory benchmark only; formal scores and costs unchanged; small sample not full-workload timing',
               model=MODEL, backend_sha256=completed['backend_sha256'], gpu=torch.cuda.get_device_name(), summary=summary, results=results)
    (HERE/'benchmark.json').write_text(json.dumps(out, indent=2)+'\n')
    lines = ['Exploratory scoring speed benchmark; formal E4 backend remains unchanged.']
    for condition, s in summary.items():
        lines.append(f'{condition}: batch 2 speedup {s["speedup_batch2"]:.3f}x; batch 5 speedup {s["speedup_batch5"]:.3f}x (four length-selected requests).')
    lines += ['All tested TV distances < 0.005. This is not a full-corpus numerical audit or full-workload throughput.',
              f'Argmax changes across measured trials: {sum(not t["argmax_same"] for r in results for t in r["trials"])}.',
              'No optimized scores enter formal E4; no benchmark duration enters cost matching.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    selftest() if args.selftest else benchmark()
