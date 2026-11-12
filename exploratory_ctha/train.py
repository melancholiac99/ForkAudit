#!/usr/bin/env python3
"""Single-GPU, sequential on-policy LoRA GRPO/PPO update for CTHA.

The rollout worker exits before this process starts; no simultaneous vLLM and
trainer memory residency. Per-token PPO clipping on exactly the sampled vLLM
completion token IDs. Actor log probabilities are computed at the SAME
sampling temperature. No KL/reference term in v1; clipped ratios bound updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(args):
    source = json.loads(args.rollout_summary.read_text())
    prov = source["provenance"]
    if prov["split"] != "train" or prov["dry"]:
        raise ValueError("Online training requires real train-split vLLM rollouts")
    if str(Path(prov["model"]).resolve()) != str(args.model.resolve()):
        raise ValueError("Model checkpoint does not match sampling model")
    adapter_hash = sha256(args.adapter / "adapter_model.safetensors") if args.adapter else None
    if prov["adapter_sha256"] != adapter_hash:
        raise ValueError("On-policy violation: sampled LoRA differs from training initialization")
    if abs(float(prov["temperature"]) - args.temperature) > 1e-9:
        raise ValueError("Old/new logprobs must use identical temperature")
    if prov["max_model_len"] != args.max_model_len:
        raise ValueError("Train and rollout max sequence length differ")
    prepared = json.loads(args.advantage_report.read_text())
    data_hash = sha256(args.data)
    if prepared["samples_sha256"] != data_hash:
        raise ValueError("Prepared advantage samples changed after validation")
    rows = []
    with args.data.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not (isinstance(row["advantage"], (int, float))
                        and math.isfinite(row["advantage"])):
                    raise ValueError("Nonfinite advantage")
                if not row["response_token_ids"]:
                    raise ValueError("Rollout completion must have vLLM token IDs")
                rows.append(row)
    if not rows:
        raise ValueError("No training tokens")
    config = dict(model=str(args.model.resolve()), adapter_sha256=adapter_hash,
                  temperature=args.temperature, data_sha256=data_hash,
                  max_model_len=args.max_model_len, lr=args.lr,
                  epsilon=args.epsilon, epochs=args.epochs,
                  accum=args.accum, grad_clip=args.grad_clip,
                  action_only=args.action_only,
                  use_checkpoint=not args.no_grad_checkpoint,
                  seed=args.seed, loss="per_token_clipped_grpo_no_kl")
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.out.exists():
        log_path = args.out / "train_log.json"
        if log_path.exists() and (args.out / "adapter_model.safetensors").is_file():
            existing = json.loads(log_path.read_text())
            if existing.get("fingerprint") == config["fingerprint"]:
                print(f"Verified completed training: {args.out}")
                return rows, config, True
        raise RuntimeError(f"Checkpoint path exists with incompatible/missing metadata: {args.out}")
    return rows, config, False


def locate_action_mask(tokenizer, generated_ids: list[int]) -> list[bool]:
    """Optional masked policy optimization, fail closed on tokenizer mismatch."""
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    matches = list(re.finditer(r"<action>.*?</action>", decoded, flags=re.I | re.S))
    if not matches:
        # Invalid/unparseable output: train the entire response to suppress it.
        return [True] * len(generated_ids)
    match = matches[-1]
    try:
        encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
        tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except Exception as exc:
        raise ValueError("Action-only masking requires a fast tokenizer with offsets") from exc
    if tokens != generated_ids:
        raise ValueError("Decoded completion does not round-trip to sampled IDs; cannot mask actions safely")
    mask = [(start < match.end() and end > match.start()) for start, end in offsets]
    if not any(mask):
        raise ValueError("No token aligns with parsed action span")
    return mask


def prepare_tokens(tokenizer, rows, max_model_len, action_only):
    encoded = []
    vocab = len(tokenizer)
    for row in rows:
        pi = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True, add_generation_prompt=True)
        if not isinstance(pi, list) or not pi:
            raise ValueError("Expected list of prompt token IDs")
        ids = [int(t) for t in row["response_token_ids"]]
        if any(t < 0 or t >= vocab for t in ids):
            raise ValueError("Sampled completion token outside tokenizer vocabulary")
        if len(pi) + len(ids) > max_model_len:
            raise ValueError(f"Prompt+completion longer than maximum: {len(pi)}+{len(ids)} > {max_model_len}")
        mask = locate_action_mask(tokenizer, ids) if action_only else [True] * len(ids)
        encoded.append(dict(prompt_ids=pi, completion_ids=ids, mask=mask,
                            advantage=float(row["advantage"])))
    return encoded


def selected_logprobs(model, example, temperature):
    """Torch implementation matching categorical sampling at temperature T."""
    import torch
    import torch.nn.functional as F
    pi, ri = example["prompt_ids"], example["completion_ids"]
    device = next(model.parameters()).device
    ids = torch.tensor([pi + ri], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        logits = model(input_ids=ids, use_cache=False).logits
    # Only materialize completion logits in fp32; avoids huge fp32 prompt logits.
    window = logits[0, len(pi)-1:len(pi)+len(ri)-1].float()
    targets = torch.tensor(ri, dtype=torch.long, device=device)
    if window.shape[0] != len(ri):
        raise RuntimeError("Shifted causal token alignment mismatch")
    return F.log_softmax(window / temperature, dim=-1).gather(
        -1, targets[:, None]).squeeze(-1)


def clipped_token_loss(logp, old_logp, advantage, mask, epsilon):
    """PPO surrogate, with each generated step weighted equally, not by response length."""
    import torch
    if not math.isfinite(advantage):
        raise ValueError("Nonfinite policy advantage")
    # Restrict exponentials while preserving PPO clipped gradients.
    ratio = torch.exp(torch.clamp(logp - old_logp, min=-20, max=20))
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    a = torch.as_tensor(advantage, device=logp.device, dtype=logp.dtype)
    surrogate = torch.minimum(ratio * a, clipped * a)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logp.device)
    if not torch.any(mask):
        raise ValueError("Empty token mask")
    return -surrogate[mask].mean()


def train(args):
    rows, config, completed = verify_inputs(args)
    if completed:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError("This is a GPU trainer; CPU CI uses unit tests only")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    data = prepare_tokens(tok, rows, args.max_model_len, args.action_only)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    actor = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=dtype,
        attn_implementation="sdpa").to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(actor, str(args.adapter), is_trainable=True)
    else:
        model = get_peft_model(actor, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM"))
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable LoRA parameters")
    for p in params:
        p.data = p.data.float()
    started = time.monotonic()
    model.eval()
    old = []
    with torch.no_grad():
        for n, sample in enumerate(data):
            lp = selected_logprobs(model, sample, args.temperature)
            old.append(lp.detach().cpu())
            if (n + 1) % 100 == 0:
                print(json.dumps(dict(phase="old_logps", processed=n+1,
                                      total=len(data))), flush=True)
    model.train()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    losses, updates, n_token_steps = [], 0, 0
    order = list(range(len(data)))
    for epoch in range(args.epochs):
        random.Random(args.seed + 1000 * epoch).shuffle(order)
        for start in range(0, len(order), args.accum):
            block = order[start:start + args.accum]
            opt.zero_grad(set_to_none=True)
            acc_loss = 0.0
            nonzero = 0
            for index in block:
                example = data[index]
                if abs(example["advantage"]) < 1e-12:
                    continue
                lp = selected_logprobs(model, example, args.temperature)
                old_lp = old[index].to(lp.device)
                loss = clipped_token_loss(lp, old_lp, example["advantage"],
                                          example["mask"], args.epsilon)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite PPO loss at sample {index}")
                (loss / len(block)).backward()
                acc_loss += float(loss.detach().cpu())
                n_token_steps += sum(example["mask"])
                nonzero += 1
            if nonzero:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                updates += 1
            losses.append(dict(epoch=epoch, batch=start // args.accum,
                               nonzero=nonzero, loss=acc_loss/max(1, nonzero),
                               updates=updates))
            print(json.dumps(dict(phase="policy_update", **losses[-1])), flush=True)
    tmp = args.out.with_name(args.out.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        raise RuntimeError(f"Unclean checkpoint temp path: {tmp}")
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp))
    if not (tmp / "adapter_model.safetensors").is_file():
        raise RuntimeError("PEFT failed to write adapter weights")
    config.update(elapsed_seconds=time.monotonic() - started,
                  n_training_steps=len(data), n_selected_token_positions=n_token_steps,
                  optimizer_steps=updates, token_mask="action" if args.action_only else "full_response",
                  trainable_parameters=sum(p.numel() for p in params),
                  mean_abs_advantage=sum(abs(d["advantage"]) for d in data)/len(data),
                  last_losses=losses[-20:], weights_sha256=sha256(tmp / "adapter_model.safetensors"))
    (tmp / "train_log.json").write_text(json.dumps(config, indent=2) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps(dict(done=True, out=str(args.out),
                          updates=updates, n_samples=len(data),
                          weights_sha256=config["weights_sha256"])), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--advantage-report", type=Path, required=True)
    p.add_argument("--rollout-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-only", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    args = p.parse_args()
    if (not 0 < args.temperature <= 2 or not 0 < args.epsilon < 1
            or args.epochs < 1 or args.accum < 1 or args.lr <= 0
            or args.grad_clip <= 0 or args.max_model_len < 256):
        p.error("Invalid policy-update hyperparameters")
    return args


if __name__ == "__main__":
    train(parse_args())
