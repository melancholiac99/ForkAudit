"""Preparation checks only: synthetic GPU operations, no policy optimization."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hardware():
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    devices = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            x = torch.randn(256, 256, device=f"cuda:{index}", dtype=torch.bfloat16,
                            requires_grad=True)
            loss = (x @ x.T).float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
            props = torch.cuda.get_device_properties(index)
            devices.append(dict(index=index, name=props.name,
                                memory_bytes=props.total_memory,
                                bf16_forward_backward=True))
    assert len(devices) == 2, f"Expected two visible GPUs, got {len(devices)}"
    return dict(devices=devices, torch=torch.__version__, cuda=torch.version.cuda)


def environment():
    sys.path.insert(0, str(ROOT / "scripts"))
    from alfenv import AlfGame, ALFWORLD_DATA
    games = json.loads((ROOT / "exploratory_E4/games.json").read_text())
    results = {}
    for split, rows in games.items():
        if not isinstance(rows, list):
            continue
        missing = [g["game_file"] for g in rows
                   if not (Path(ALFWORLD_DATA) / g["game_file"]).is_file()]
        assert not missing, f"Missing {split} games: {missing[:3]}"
        if not rows:
            continue
        game = AlfGame(str(Path(ALFWORLD_DATA) / rows[0]["game_file"]))
        try:
            obs, actions, won = game.reset()
            assert obs and actions
            step = game.step(actions[0])
            assert len(step) == 4
            results[split] = dict(games=len(rows), reset_and_step=True)
        finally:
            game.env.close()
    return results


def inference(out):
    from vllm import LLM, SamplingParams
    model = os.environ["M0_MODEL"]
    engine = LLM(model=model, dtype="bfloat16", seed=0, max_model_len=6144,
                 enforce_eager=True, enable_prefix_caching=True,
                 gpu_memory_utilization=0.55, generation_config="vllm",
                 enable_lora=True, max_lora_rank=16, max_loras=1,
                 logprobs_mode="processed_logprobs")
    tok = engine.get_tokenizer()
    prompt = tok.apply_chat_template([
        {"role": "user", "content": "Your task is to find an apple. Reply with [action] look [/action]."}
    ], tokenize=False, add_generation_prompt=True)
    params = SamplingParams(temperature=0.7, top_p=1., top_k=-1,
                            max_tokens=24, seed=20261113, logprobs=0)
    try:
        first = engine.generate([prompt], params, use_tqdm=False)[0]
        second = engine.generate([prompt], params, use_tqdm=False)[0]
        a, b = first.outputs[0], second.outputs[0]
        assert a.token_ids and a.token_ids == b.token_ids, "Repeat token IDs differ"
        probs = [float(item[token].logprob) for token, item in zip(a.token_ids, a.logprobs)]
        assert probs == [float(item[token].logprob) for token, item in zip(b.token_ids, b.logprobs)]
        return dict(prompt_ids=list(first.prompt_token_ids), completion_ids=list(a.token_ids),
                    behavior_logprobs=probs, text=a.text, exact_repeat=True,
                    cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"))
    finally:
        engine.llm_engine.engine_core.shutdown()


def hf_check(sample):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(ROOT / "exploratory_ctha"))
    from train import selected_logprobs
    torch.manual_seed(20261113)
    torch.cuda.manual_seed_all(20261113)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads(sample.read_text())
    model = AutoModelForCausalLM.from_pretrained(
        os.environ["M0_MODEL"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    example = dict(prompt_ids=source["prompt_ids"], completion_ids=source["completion_ids"])
    with torch.no_grad():
        actual = selected_logprobs(model, example, .7)
    diff = (actual.cpu() - torch.tensor(source["behavior_logprobs"])).abs()
    plan = json.loads((ROOT / "exploratory_ctha/v3/PLAN.json").read_text())["parity_gate"]
    parity = dict(mean=float(diff.mean()), p99=float(torch.quantile(diff, .99)), max=float(diff.max()))
    parity["passed"] = (parity["mean"] <= plan["mean_absolute_sampled_logprob_error_max"]
                        and parity["p99"] <= plan["p99_absolute_sampled_logprob_error_max"]
                        and parity["max"] <= plan["max_absolute_sampled_logprob_error_max"])
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0., bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    loss = -selected_logprobs(model, example, .7).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)
    return dict(parity=parity, lora_backward=True, optimizer_steps=0,
                cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["hardware", "environment", "vllm", "hf"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a fresh output path to preserve previous checks")
    if a.stage == "hardware":
        result = hardware()
    elif a.stage == "environment":
        result = environment()
    elif a.stage == "vllm":
        result = inference(a.out)
    else:
        if a.sample is None:
            p.error("hf requires --sample from the vllm stage")
        result = hf_check(a.sample)
    result.update(stage=a.stage, check_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  packages={name: importlib.metadata.version(name) for name in
                            ["torch", "vllm", "transformers", "peft", "alfworld", "textworld"]})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if a.stage == "hf" and not result["parity"]["passed"]:
        raise SystemExit("Synthetic parity check failed; do not proceed to training")


if __name__ == "__main__":
    main()
