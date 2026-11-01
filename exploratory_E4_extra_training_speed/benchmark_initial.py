"""Isolated forward/backward speed study; never writes formal E4 training artifacts."""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = Path('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
ADAPTER = ROOT/'exploratory_E4/data/adapters/Aprime_s0_lr5e-06'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
VARIANTS = ['original', 'response_only', 'response_only_no_checkpoint']
# Engineering tolerances fixed before real-model measurements, not research decision gates.
TOLERANCES = dict(loss_absolute=1e-4, gradient_relative_l2=1e-3, gradient_cosine_min=.99999)


def save(name, value):
    path = HERE/name; tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def logp(model, pair, variant):
    pi, ri = pair
    device = next(model.parameters()).device
    ids = torch.tensor([pi+ri], device=device)
    kwargs = {}
    if variant != 'original':
        kwargs = dict(logits_to_keep=torch.arange(len(pi)-1, len(pi)+len(ri)-1, device=device), use_cache=False)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = model(input_ids=ids, **kwargs)
        logits = (output.logits[0, len(pi)-1:-1] if variant == 'original' else output.logits[0]).float()
    assert logits.shape[0] == len(ri)
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ri, device=device)[:, None])[:, 0]
    return lp.sum(), lp.mean()


def configure(model, variant):
    if variant == 'response_only_no_checkpoint':
        model.gradient_checkpointing_disable()
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()


def trial(model, job, variant, refs, seed, collect_grad=True):
    configure(model, variant)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if job['kind'] == 'DPO':
        chosen = logp(model, job['c'], variant)[0]
        rejected = logp(model, job['r'], variant)[0]
        margin = (chosen-refs[0])-(rejected-refs[1])
        loss = -torch.nn.functional.logsigmoid(.1*margin)
        _ = int(margin.item() > 0)  # Match the formal per-example metric synchronization.
    else:
        loss = -logp(model, job['s'], variant)[1]
    (loss/16).backward()
    value = float(loss.detach())  # Match formal loss logging after backward.
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
    grad = None
    if collect_grad:
        grad = torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1).float().cpu()
                          for p in model.parameters() if p.requires_grad])
    return dict(seconds=seconds, loss=value, peak_allocated_bytes=peak), grad


def compare(actual, baseline, grad, reference):
    diff = grad-reference
    norm = float(torch.linalg.vector_norm(reference))
    relative = float(torch.linalg.vector_norm(diff))/max(norm, 1e-30)
    other_norm = float(torch.linalg.vector_norm(grad))
    cosine = float(torch.dot(grad, reference))/max(norm*other_norm, 1e-30)
    delta = abs(actual['loss']-baseline['loss'])
    return dict(loss_absolute_difference=delta, gradient_relative_l2=relative,
                gradient_cosine=cosine, max_gradient_absolute_difference=float(diff.abs().max()),
                gradients_bitwise_equal=bool(torch.equal(grad, reference)),
                finite=math.isfinite(actual['loss']) and bool(torch.isfinite(grad).all()),
                passed=delta <= TOLERANCES['loss_absolute'] and relative <= TOLERANCES['gradient_relative_l2']
                and cosine >= TOLERANCES['gradient_cosine_min'])


def references(model, job):
    model.eval()
    if job['kind'] != 'DPO':
        return None
    with torch.no_grad(), model.disable_adapter():
        return [float(logp(model, job[key], 'original')[0]) for key in ['c', 'r']]


def selftest():
    torch.set_num_threads(1); torch.manual_seed(19)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = 'sdpa'
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=.05,
                                           target_modules=TARGETS, bias='none', task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
                if 'lora_B' in name:
                    p.normal_(0, .01)  # Exercise nonzero gradients in both A and B matrices.
    jobs = [dict(kind='DPO', c=[[1, 2, 3, 4], [5, 6, 7]], r=[[1, 2, 3, 4], [8, 9, 10, 11]]),
            dict(kind='SFT', s=[list(range(1, 32)), [32, 33, 34]])]
    checks = []
    for i, job in enumerate(jobs):
        refs = references(model, job)
        base, grad = trial(model, job, 'original', refs, 113+i)
        for variant in VARIANTS[1:]:
            result, other = trial(model, job, variant, refs, 113+i)
            check = compare(result, base, other, grad)
            assert check['passed'] and check['finite'], check
            checks.append(dict(kind=job['kind'], variant=variant, **check))
    assert not torch.cuda.is_initialized()
    report = dict(passed=True, scope='Tiny random CPU model only, not real 7B evidence', tolerances=TOLERANCES, checks=checks)
    save('cpu_selftest.json', report); print(json.dumps(report, indent=2), flush=True)


def prepare():
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    end = tok.convert_tokens_to_ids('<|im_end|>')
    def encode(row, key):
        text = tok.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False, add_generation_prompt=True)
        pi = tok.encode(text, add_special_tokens=False)
        ri = tok.encode(row[key], add_special_tokens=False)+[end]
        assert len(pi)+len(ri) <= 6144
        return [pi, ri]
    dpo, sft, hashes = [], [], {}
    for path in sorted((ROOT/'exploratory_E4/data/pairs').glob('*.jsonl')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for i, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            if path.stem == 'Esft':
                pair = encode(row, 'response')
                sft.append(dict(kind='SFT', s=pair, arm=path.stem, row=i, total_tokens=sum(map(len, pair))))
            else:
                c, r = encode(row, 'chosen'), encode(row, 'rejected')
                dpo.append(dict(kind='DPO', c=c, r=r, arm=path.stem, row=i, total_tokens=sum(map(len, c))+sum(map(len, r))))
    dpo.sort(key=lambda j: (j['total_tokens'], j['arm'], j['row']))
    sft.sort(key=lambda j: (j['total_tokens'], j['row']))
    selected = [dpo[round(q*(len(dpo)-1))] for q in [0, .5, .9, 1]]+[sft[-1]]
    report = dict(selection='DPO total token length quantiles 0/.5/.9/1 across all arms; longest SFT example; no outcome/loss-based selection',
                  source_sha256=hashes, tolerances=TOLERANCES, cases=selected,
                  population_dpo=len(dpo), population_sft=len(sft))
    save('inputs.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}), flush=True)
    print([(j['kind'], j['arm'], j['row'], j['total_tokens']) for j in selected], flush=True)


def benchmark():
    prepared = json.loads((HERE/'inputs.json').read_text())
    assert prepared['tolerances'] == TOLERANCES
    for name, digest in prepared['source_sha256'].items():
        assert hashlib.sha256((ROOT/'exploratory_E4/data/pairs'/name).read_bytes()).hexdigest() == digest
    assert (ADAPTER/'train_log.json').is_file()
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation='sdpa').to('cuda')
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    results = []
    for i, job in enumerate(prepared['cases']):
        refs = references(model, job)
        seed = 82000+i
        baseline, reference_grad = trial(model, job, 'original', refs, seed)
        record = {k:job[k] for k in ['kind', 'arm', 'row', 'total_tokens']}
        record['trials'] = []
        for repeat in range(2):
            order = VARIANTS if repeat == 0 else list(reversed(VARIANTS))
            for variant in order:
                try:
                    actual, grad = trial(model, job, variant, refs, seed)
                    checks = compare(actual, baseline, grad, reference_grad)
                    record['trials'].append(dict(variant=variant, repeat=repeat, **actual, **checks))
                    del grad
                except torch.cuda.OutOfMemoryError:
                    record['trials'].append(dict(variant=variant, repeat=repeat, passed=False, error='CUDA OOM'))
                    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        results.append(record)
        save('benchmark_progress.json', results)
        print(json.dumps(record), flush=True)
        del reference_grad
    summary = {}
    for variant in VARIANTS[1:]:
        checks = [t for r in results for t in r['trials'] if t['variant'] == variant]
        ratios = []
        for r in results:
            a = [t['seconds'] for t in r['trials'] if t['variant']=='original' and 'seconds' in t]
            b = [t['seconds'] for t in r['trials'] if t['variant']==variant and 'seconds' in t]
            if a and b:
                ratios.append(statistics.median(a)/statistics.median(b))
        summary[variant] = dict(all_checks_passed=all(t['passed'] and t.get('finite', False) for t in checks),
                                per_case_speedup=ratios, geometric_mean_speedup=math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                                peak_allocated_GiB=max((t.get('peak_allocated_bytes', 0) for t in checks), default=0)/1024**3,
                                max_gradient_relative_l2=max((t.get('gradient_relative_l2', 0) for t in checks), default=0))
    out = dict(complete=True, scope='Engineering benchmark; no formal weights, examples, hyperparameters or pipeline implementation changed',
               gpu=torch.cuda.get_device_name(), torch_threads=torch.get_num_threads(), tolerances=TOLERANCES,
               versions={name:importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft']},
               benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               formal_training_source_sha256=hashlib.sha256((ROOT/'exploratory_E4/e4_06_train.py').read_bytes()).hexdigest(),
               adapter=str(ADAPTER), adapter_sha256=hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest(),
               selection=prepared['selection'], summary=summary, results=results)
    save('benchmark.json', out)
    lines = ['E4 training speed study: five length-selected cases, two repetitions, fixed dropout RNG per case.',
             'Measures forward/backward for one micro-example, excludes optimizer and reference preprocessing; not full-run speed.']
    for name, s in summary.items():
        lines.append(f'{name}: geometric mean speedup {s["geometric_mean_speedup"]}; checks passed {s["all_checks_passed"]}; peak {s["peak_allocated_GiB"]:.2f} GiB; max relative gradient error {s["max_gradient_relative_l2"]:.6g}.')
    lines += ['No optimization was installed in formal E4. Passing these sampled engineering gates does not imply bitwise-equivalent full training.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['selftest', 'prepare', 'benchmark'])
    args = parser.parse_args()
    {'selftest': selftest, 'prepare': prepare, 'benchmark': benchmark}[args.mode]()
