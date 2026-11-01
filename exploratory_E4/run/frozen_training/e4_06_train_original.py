"""E4 step 6: LoRA training of one arm and one training seed (PREREG_E4 §5).

DPO arms : loss = -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)]), response-token log-probability sums,
           reference = the frozen base model (adapter disabled), precomputed once before training.
Esft     : mean NLL of response tokens.
LoRA r=16, alpha=32, dropout 0.05 on all linear projections; AdamW, 10% linear warm-up then constant; effective batch 16
(micro-batch 1, gradient accumulation); 2 epochs; bf16 base weights, fp32 LoRA weights; gradient checkpointing.
Prompt = the decision-time prompt under the chat template with generation prompt; response = the recorded text + <|im_end|>.

Usage: python e4_06_train.py --arm A --seed 0 --lr 2e-5 [--smoke]
Output: data/adapters/{arm}_s{seed}[_lr]/ (adapter_model.safetensors, adapter_config.json, train_log.json)
"""
import argparse
import json
import math
import os
import random
import time

import e4_common as E

BETA, R, ALPHA, DROPOUT, EFF_BATCH, EPOCHS, MAX_LEN = 0.1, 16, 32, 0.05, 16, 2, 6144
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=E.ARMS_TRAINED)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--model", default=os.environ.get("M0_MODEL", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="8 examples, 4 optimizer steps of batch 2; checks loss and save")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = [json.loads(l) for l in open(os.path.join(E.DATA, "pairs", f"{args.arm}.jsonl"))]
    eff_batch = EFF_BATCH
    if args.smoke:
        rows, eff_batch = rows[:8], 2
    out = args.out or os.path.join(E.DATA, "adapters", f"{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else ""))
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(end_id, int) and end_id >= 0

    def encode(prompt, response):
        p = tok.apply_chat_template([dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True)
        pi = tok.encode(p, add_special_tokens=False)
        ri = tok.encode(response, add_special_tokens=False) + [end_id]
        assert len(pi) + len(ri) <= MAX_LEN, (len(pi), len(ri))
        return pi, ri

    dpo = args.arm != "Esft"
    data = []
    for r in rows:
        if dpo:
            data.append(dict(c=encode(r["prompt"], r["chosen"]), r=encode(r["prompt"], r["rejected"])))
        else:
            data.append(dict(s=encode(r["prompt"], r["response"])))
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=TARGETS,
                                             bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    def seq_logp(pi, ri):
        ids = torch.tensor([pi + ri], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits[0, len(pi) - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ri, device="cuda")[:, None])[:, 0]
        return lp.sum(), lp.mean(), len(ri)

    if dpo:  # reference log-probabilities with the adapter disabled
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            for d in data:
                d["ref_c"] = float(seq_logp(*d["c"])[0])
                d["ref_r"] = float(seq_logp(*d["r"])[0])
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_total = math.ceil(len(data) * EPOCHS / eff_batch)
    warm = max(1, int(0.1 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    log, t0, step = [], time.time(), 0
    order = []
    for ep in range(EPOCHS):
        o = list(range(len(data)))
        random.Random(args.seed * 1000 + ep).shuffle(o)
        order += o
    acc = dict(loss=0.0, n=0, margin_pos=0)
    for k, idx in enumerate(order):
        d = data[idx]
        if dpo:
            pc = seq_logp(*d["c"])[0]
            pr = seq_logp(*d["r"])[0]
            margin = (pc - d["ref_c"]) - (pr - d["ref_r"])
            loss = -F.logsigmoid(BETA * margin)
            acc["margin_pos"] += int(margin.item() > 0)
        else:
            _, mean_lp, _ = seq_logp(*d["s"])
            loss = -mean_lp
        (loss / eff_batch).backward()
        acc["loss"] += loss.item()
        acc["n"] += 1
        if (k + 1) % eff_batch == 0 or k + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            log.append(dict(step=step, loss=acc["loss"] / acc["n"], reward_acc=acc["margin_pos"] / acc["n"] if dpo else None,
                            lr=sched.get_last_lr()[0], seconds=time.time() - t0))
            print(json.dumps(log[-1]), flush=True)
            acc = dict(loss=0.0, n=0, margin_pos=0)
    model.save_pretrained(out)
    E.write_json(os.path.join(out, "train_log.json"), dict(
        arm=args.arm, seed=args.seed, lr=args.lr, beta=BETA, r=R, alpha=ALPHA, dropout=DROPOUT, eff_batch=eff_batch, epochs=EPOCHS,
        n_examples=len(data), steps=step, log=log, gpu_seconds=time.time() - t0, smoke=args.smoke,
        pairs_sha256=E.file_sha256(os.path.join(E.DATA, "pairs", f"{args.arm}.jsonl")), model=args.model))
    if args.smoke:
        assert all(math.isfinite(x["loss"]) for x in log)
        assert os.path.exists(os.path.join(out, "adapter_model.safetensors"))
        print("SMOKE OK", out)


if __name__ == "__main__":
    main()
