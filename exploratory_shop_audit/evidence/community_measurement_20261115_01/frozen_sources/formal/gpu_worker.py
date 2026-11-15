"""One resident, frozen Qwen model. JSON RPC with per-request fixed seeds."""
import argparse
import contextlib
import json
import os
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    # Preserve the RPC channel even when vLLM/native libraries write fd 1.
    rpc = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    with contextlib.redirect_stdout(sys.stderr):
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", tensor_parallel_size=1,
                  max_model_len=8704, gpu_memory_utilization=0.75, max_num_seqs=16,
                  enable_prefix_caching=True, seed=20261115, disable_log_stats=True)
    rpc.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("cmd") == "quit":
            return
        try:
            batch = request["requests"]
            params = [SamplingParams(temperature=row.get("temperature", 0.7), top_p=1,
                       max_tokens=row.get("max_tokens", 512), seed=row["seed"])
                      for row in batch]
            with contextlib.redirect_stdout(sys.stderr):
                outputs = llm.generate([{"prompt_token_ids": row["token_ids"]} for row in batch],
                                       params, use_tqdm=False)
            results = []
            for row, output in zip(batch, outputs):
                reply = output.outputs[0]
                if list(output.prompt_token_ids) != row["token_ids"]:
                    raise RuntimeError("Inference input IDs differ from recorded IDs")
                results.append({"text": reply.text, "token_ids": list(reply.token_ids),
                                "finish_reason": reply.finish_reason, "stop_reason": reply.stop_reason})
            result = {"ok": True, "result": results}
        except Exception:
            result = {"ok": False, "error": traceback.format_exc()}
        rpc.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
