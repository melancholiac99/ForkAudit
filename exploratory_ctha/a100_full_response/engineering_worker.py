"""New protocol actions wrapping preserved R2 numerical implementations."""
import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from protocol_common import ENGINEERING, FAST, HERE, MODEL, PARENT, R2, ROOT, configure, load, save, sha, sources, emit


def new_trainer():
    spec = importlib.util.spec_from_file_location('full_response_trainer', HERE / 'trainer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'checkpoint', 'adapter_sample', 'adapter_score'])
    a = p.parse_args()
    assert sources() == load(ENGINEERING / 'registration.json')['source_sha256']
    backend, util = configure(ENGINEERING)
    config = load(ENGINEERING / 'frozen_config.json')
    if a.action == 'train':
        original = backend.legacy
        backend.legacy = lambda name: new_trainer() if name == 'trainer' else original(name)
        from types import SimpleNamespace
        backend.train(SimpleNamespace(out=ENGINEERING / 'direction_probe', config=ENGINEERING / 'frozen_config.json'), util)
        emit(ENGINEERING, 'engineering', {'checkpoint/saved': 1}, 4)
    elif a.action == 'checkpoint':
        runpy.run_path(str(PARENT / 'v3_engineering_fast_20261113/check_checkpoint.py'), run_name='__main__')
        emit(ENGINEERING, 'engineering', {'checkpoint/restored': 1}, 4)
    elif a.action == 'adapter_sample':
        backend.adapter_sample(config)
    else:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        from precision import selected_logprobs
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        out = ENGINEERING / 'adapter_parity'
        rows = load(out / 'replies.json')
        model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True, dtype=torch.float32, attn_implementation='sdpa').to('cuda')
        model = PeftModel.from_pretrained(model, str(ENGINEERING / 'direction_probe')).eval()
        with torch.no_grad():
            unmerged = [selected_logprobs(model, row, .7).cpu() for row in rows]
            model = model.merge_and_unload(safe_merge=True).eval()
            merged = [selected_logprobs(model, row, .7).cpu() for row in rows]
        save(out / 'hf_logprobs.json', [v.tolist() for v in unmerged], immutable=True)
        save(out / 'hf_merged_logprobs.json', [v.tolist() for v in merged], immutable=True)
        pg = load(HERE / 'PLAN.json')['parity_gate']

        def report(left, right):
            errors = np.asarray([v for a, b in zip(left, right) for v in (a - b).abs().tolist()])
            value = dict(mean_abs=float(errors.mean()), p99_abs=float(np.quantile(errors, .99)), max_abs=float(errors.max()), tokens=len(errors))
            value['passed'] = value['mean_abs'] <= pg['mean_absolute_sampled_logprob_error_max'] and value['p99_abs'] <= pg['p99_absolute_sampled_logprob_error_max'] and value['max_abs'] <= pg['max_absolute_sampled_logprob_error_max']
            return value

        parity = report(unmerged, [torch.tensor(r['behavior_logprobs']) for r in rows])
        parity['adapter_sha256'] = sha(ENGINEERING / 'direction_probe/adapter_model.safetensors')
        merge = report(unmerged, merged)
        save(out / 'verification.json', parity, immutable=True)
        save(out / 'merge_verification.json', merge, immutable=True)
        assert parity['passed'] and merge['passed'], 'Updated policy parity or merge mismatch'
        emit(ENGINEERING, 'engineering', {'parity/updated_max': parity['max_abs'], 'parity/merge_max': merge['max_abs']}, 4)


if __name__ == '__main__':
    main()
