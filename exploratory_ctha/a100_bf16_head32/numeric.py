"""Fresh real-prompt BF16/FP32-head candidate, with unchanged parity gates."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'exploratory_ctha/a100_validation'))
from a100_common import MODEL, hardware, load, save, sha

OUT = ROOT / '.runtime/runs/a100_bf16_head32_r1'
PROMPTS = ROOT / '.runtime/runs/ctha_a100_eng_20261113_r2/reference/prompts.json'


def sample():
    from vllm import LLM, SamplingParams
    import vllm
    OUT.mkdir(parents=True, exist_ok=False)
    source = Path(vllm.__file__).parent / 'model_executor/layers/logits_processor.py'
    save(OUT / 'registration.json', dict(time=time.time(), plan=load(HERE / 'PLAN.json'),
        source_sha256=sha(Path(__file__)), plan_sha256=sha(HERE / 'PLAN.json'),
        input_sha256=sha(PROMPTS), hardware=hardware(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        vllm_logits_processor_sha256=sha(source),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'transformers', 'peft', 'triton']}), immutable=True)
    engine = LLM(model=str(MODEL), dtype='bfloat16', hf_overrides={'head_dtype': 'float32'},
        seed=0, max_model_len=6144, gpu_memory_utilization=.55, max_num_seqs=16,
        enforce_eager=True, enable_prefix_caching=True, enable_lora=False,
        generation_config='vllm', logprobs_mode='processed_logprobs')
    try:
        tok = engine.get_tokenizer()
        prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
            add_generation_prompt=True) for r in load(PROMPTS)]
        params = [SamplingParams(temperature=.7, top_p=1., top_k=-1, min_p=0.,
            repetition_penalty=1., presence_penalty=0., frequency_penalty=0.,
            max_tokens=512, logprobs=0, seed=20261113 + i) for i in range(len(prompts))]
        started = time.perf_counter()
        replies = engine.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - started
        rows = []
        for i, r in enumerate(replies):
            c = r.outputs[0]
            rows.append(dict(prompt_ids=list(r.prompt_token_ids), completion_ids=list(c.token_ids),
                behavior_logprobs=[float(p[t].logprob) for p, t in zip(c.logprobs, c.token_ids)],
                seed=20261113 + i))
        save(OUT / 'replies.json', rows, immutable=True)
        save(OUT / 'performance.json', dict(seconds=seconds, tokens=sum(len(r['completion_ids']) for r in rows),
            tokens_per_second=sum(len(r['completion_ids']) for r in rows) / seconds, timed_warmup=False), immutable=True)
    finally:
        engine.llm_engine.engine_core.shutdown()


def score():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    # Keep the backbone BF16; avoid the output-logit BF16 rounding before softmax.
    head = model.get_output_embeddings()
    head.float()
    hook = head.register_forward_pre_hook(lambda module, args: (args[0].float(), *args[1:]))
    scores, errors, drifts, ratios = [], [], [], []
    started = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            for row in load(OUT / 'replies.json'):
                pi, ri = row['prompt_ids'], row['completion_ids']
                logits = model(input_ids=torch.tensor([pi + ri], device='cuda'), use_cache=False).logits
                assert logits.dtype == torch.float32 and model.get_input_embeddings().weight.dtype == torch.bfloat16
                window = logits[0, len(pi)-1:len(pi)+len(ri)-1]
                lp = (window / .7).log_softmax(-1).gather(-1, torch.tensor(ri, device='cuda')[:, None]).squeeze(-1).cpu()
                old = torch.tensor(row['behavior_logprobs'])
                scores.append(lp.tolist())
                errors.extend((lp - old).abs().tolist())
                d = old - lp
                drifts.extend((d.exp() - d - 1).tolist())
                ratios.extend((lp - old).exp().tolist())
    finally:
        hook.remove()
    es, rs = np.asarray(errors), np.asarray(ratios)
    report = dict(mean_abs=float(es.mean()), p99_abs=float(np.quantile(es, .99)), max_abs=float(es.max()),
        sampled_behavior_k3=float(np.mean(drifts)), clip_fraction=float(np.mean((rs < .8) | (rs > 1.2))),
        token_ESS_fraction=float(rs.sum()**2 / (len(rs) * (rs**2).sum())),
        tokens=len(es), hf_seconds=time.perf_counter() - started,
        hf_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
    gate = load(HERE / 'PLAN.json')['gates']
    report['original_probability_gate_passed'] = report['mean_abs'] <= gate['mean_abs'] and report['p99_abs'] <= gate['p99_abs'] and report['max_abs'] <= gate['max_abs']
    report['ratio_stability_passed'] = report['sampled_behavior_k3'] <= gate['sampled_behavior_k3'] and report['clip_fraction'] <= gate['clip_fraction']
    report['passed'] = report['original_probability_gate_passed'] and report['ratio_stability_passed']
    report.update(adopted=False, optimizer_updates=0, Engineering='INCOMPLETE' if report['passed'] else 'FAIL',
        Learning='NOT_RUN', CTHA_online_efficacy='NOT_TESTED')
    save(OUT / 'hf_logprobs.json', scores, immutable=True)
    save(OUT / 'verification.json', report, immutable=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['sample', 'score'])
    a = p.parse_args()
    (sample if a.stage == 'sample' else score)()
