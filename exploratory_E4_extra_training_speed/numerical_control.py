"""Deterministic gradient checks only; shared-GPU timings deliberately not reported."""
import os
# Must be set before CUDA initialization for deterministic cuBLAS reductions.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import json
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import benchmark as B


def main():
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.5)
    free, total = torch.cuda.mem_get_info()
    assert free > 45*1024**3, 'Insufficient spare GPU memory; do not compete for formal training allocations'
    prepared = json.loads((B.HERE/'inputs.json').read_text())
    model = AutoModelForCausalLM.from_pretrained(B.BASE, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation='sdpa').to('cuda')
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, B.ADAPTER, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    cases = []
    for i, job in enumerate(prepared['cases']):
        refs = B.references(model, job)
        baseline, reference = B.trial(model, job, 'original', refs, 82000+i)
        result = {k:job[k] for k in ['kind', 'arm', 'row', 'total_tokens']}
        result['checks'] = []
        for variant in ['original', 'no_checkpoint', 'response_only', 'response_only_no_checkpoint', 'original']:
            actual, grad = B.trial(model, job, variant, refs, 82000+i)
            checks = B.compare(actual, baseline, grad, reference)
            result['checks'].append(dict(variant=variant, **checks, peak_allocated_bytes=actual['peak_allocated_bytes']))
            del grad
        cases.append(result)
        B.save('numerical_control_progress.json', cases)
        print(json.dumps(result), flush=True)
        del reference
    variants = ['original', 'no_checkpoint', 'response_only', 'response_only_no_checkpoint']
    summary = {}
    for variant in variants:
        checks = [c for r in cases for c in r['checks'] if c['variant'] == variant]
        summary[variant] = dict(all_passed=all(c['passed'] and c['finite'] for c in checks),
                                all_bitwise_equal=all(c['gradients_bitwise_equal'] for c in checks),
                                max_loss_difference=max(c['loss_absolute_difference'] for c in checks),
                                max_gradient_relative_l2=max(c['gradient_relative_l2'] for c in checks),
                                min_gradient_cosine=min(c['gradient_cosine'] for c in checks),
                                peak_allocated_GiB=max(c['peak_allocated_bytes'] for c in checks)/1024**3)
    out = dict(complete=True, scope='Numerical control only; GPU shared with formal training, no speed measurements used',
               deterministic_algorithms=True, cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
               per_process_memory_fraction=.5, metric_reduction_dtype='float64', thresholds=B.TOLERANCES,
               note='Initial benchmark cosine metrics used FP32 reductions and are not reliable; this control uses FP64. Original outputs retained.',
               summary=summary, cases=cases,
               benchmark_source_sha256=hashlib.sha256((B.HERE/'benchmark.py').read_bytes()).hexdigest())
    B.save('numerical_control.json', out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
