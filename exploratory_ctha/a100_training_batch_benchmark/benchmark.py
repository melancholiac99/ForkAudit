"""Compare true full-response gradients under compute microbatching; no updates."""
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'exploratory_ctha/a100_full_response'))
from protocol_common import ENGINEERING, FAST, MODEL, configure, fingerprint, hardware, load, save, sha


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import save_file
    out = ROOT / '.runtime/runs/a100_training_batch_benchmark_r1'
    out.mkdir(parents=True, exist_ok=False)
    plan = load(HERE / 'PLAN.json')
    save(out / 'registration.json', dict(plan=plan, hardware=hardware(),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256=sha(Path(__file__)), plan_sha256=sha(HERE / 'PLAN.json'),
        adapter_sha256=sha(ENGINEERING / 'direction_probe/adapter_model.safetensors'),
        samples_sha256=sha(ENGINEERING / 'initial_prepared/samples.jsonl'),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft']},
        time=time.time()), immutable=True)
    configure(ENGINEERING)
    from precision import selected_logprobs
    from train import clipped_token_loss
    from trainer import k3
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32,
        local_files_only=True, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(model, str(ENGINEERING / 'direction_probe'), is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rows = [json.loads(line) for line in (ENGINEERING / 'initial_prepared/samples.jsonl').read_text().splitlines()]
    rows = [rows[i] for i in load(ENGINEERING / 'direction_probe/probe_indices.json')]
    data = [dict(prompt_ids=r['prompt_token_ids'], completion_ids=r['response_token_ids'],
        old=torch.tensor(r['behavior_logprobs'], device='cuda'), advantage=r['advantage']) for r in rows]
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for row in data:
            row['ref'] = selected_logprobs(model, row, .7).detach()
    spec = load(FAST / 'PLAN.json')['optimizer']

    def batch_logprobs(group):
        if len(group) == 1:
            return [selected_logprobs(model, group[0], .7)]
        lengths = [len(row['prompt_ids']) + len(row['completion_ids']) for row in group]
        width = max(lengths)
        ids = torch.full((len(group), width), 151643, device='cuda', dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(group):
            ids[i, :lengths[i]] = torch.tensor(row['prompt_ids'] + row['completion_ids'], device='cuda')
            mask[i, :lengths[i]] = 1
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        assert logits.dtype == torch.float32
        outputs = []
        for i, row in enumerate(group):
            start = len(row['prompt_ids']) - 1
            window = logits[i, start:start + len(row['completion_ids'])]
            outputs.append((window / .7).log_softmax(-1).gather(-1, torch.tensor(row['completion_ids'], device='cuda')[:, None]).squeeze(-1))
        return outputs

    def compute(batch_size):
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        scores, losses = [], []
        for off in range(0, len(data), batch_size):
            group = data[off:off + batch_size]
            lps = batch_logprobs(group)
            total = 0
            for row, lp in zip(group, lps):
                loss = clipped_token_loss(lp, row['old'], row['advantage'], [True] * len(lp), spec['clip']) + spec['kl_coef'] * k3(lp, row['ref']).mean()
                total = total + loss / len(data)
                scores.append(lp.detach().cpu())
                losses.append(float(loss.detach()))
            total.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gradients = {n: p.grad.detach().cpu().clone() for n, p in params}
        assert all(torch.isfinite(v).all() for v in gradients.values())
        return dict(seconds=elapsed, responses_per_second=len(data) / elapsed,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3), scores, losses, gradients

    # Warm up the exact single-response implementation before the timed reference.
    compute(1)
    baseline = None
    results = []
    for size in plan['candidates']:
        folder = out / f'microbatch_{size}'
        folder.mkdir()
        perf, scores, losses, grads = compute(size)
        save(folder / 'logprobs.json', [v.tolist() for v in scores], immutable=True)
        save(folder / 'losses.json', losses, immutable=True)
        save_file(grads, str(folder / 'gradients.safetensors'))
        if baseline is None:
            baseline = (perf, scores, losses, grads)
        refperf, refscores, reflosses, refgrads = baseline
        errors = np.asarray([v for x, y in zip(scores, refscores) for v in (x - y).abs().tolist()])
        sqdiff = sum(float((grads[n].double() - refgrads[n].double()).square().sum()) for n in grads)
        sqref = sum(float(v.double().square().sum()) for v in refgrads.values())
        report = dict(microbatch=size, performance=perf, speedup=refperf['seconds'] / perf['seconds'],
            mean_abs=float(errors.mean()), p99_abs=float(np.quantile(errors, .99)), max_abs=float(errors.max()),
            max_loss_difference=max(abs(a - b) for a, b in zip(losses, reflosses)),
            gradient_relative_l2=(sqdiff / max(sqref, 1e-30)) ** .5,
            gradient_max_abs=max(float((grads[n] - refgrads[n]).abs().max()) for n in grads),
            optimizer_updates=0, gradient_sha256=sha(folder / 'gradients.safetensors'))
        report['passed'] = report['mean_abs'] <= .01 and report['p99_abs'] <= .05 and report['max_abs'] <= .25 and report['max_loss_difference'] <= plan['loss_absolute_difference_max'] and report['gradient_relative_l2'] <= plan['gradient_relative_l2_max'] and report['gradient_max_abs'] <= plan['gradient_max_absolute_difference_max']
        save(folder / 'report.json', report, immutable=True)
        print(json.dumps(report), flush=True)
        results.append(report)
        del grads
    save(out / 'verification.json', dict(results=results, optimizer_updates=0,
        adopted=False, limitation='Eight real responses only; requires fresh engineering protocol/fingerprint before any deployment.'), immutable=True)


if __name__ == '__main__':
    main()
