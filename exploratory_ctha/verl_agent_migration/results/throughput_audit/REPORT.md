# Throughput code audit

Read-only investigation on pinned verl-agent20bd331b and installed vLLM0.11.0.
No new GPU run, optimizer update or measured speedup is claimed. Exact source
and real packet hashes are in `evidence.json`; reproduce with
`python exploratory_ctha/verl_agent_migration/inspect_throughput.py` in the pinned env.

## 1. Every environment turn offloads the whole inference model

`fsdp_workers.generate_sequences` enters/exits the sharding manager each turn.
`fsdp_vllm.__exit__` invokes `sleep(level=1)`. Installed vLLM
`v1/worker/gpu_worker.py:112` selects the `weights` allocation tag;
`load_model` loads the base under this tag. `device_allocator/cumem.py:215`
allocates pinned CPU backups and calls cudaMemcpy for those allocations;
`wake_up` restores them to GPU. Thus safetensors preload avoids repeated HF-base
synchronization, but does NOT avoid inference-base CPU/GPU roundtrips.

The BF16 base parameter payload is about14.19GiB. Fifty sleep/wake roundtrips
would move about1.39TiB of base payload alone. This is a static estimate, not
measured PCIe traffic, and excludes LoRA/buffers. The per-turn process also
collects about616MiB of FP32 LoRA and repeatedly empties the CUDA allocator cache.
This strengthens the case for one GPU-worker-owned session per rollout batch,
with cooperative/finally-based teardown before HF scoring or updates.

## 2. Each turn registers an adapter with a new ID

`fsdp_vllm.update_params` uses `time.time_ns()` for a fresh LoRA ID, then
`add_lora`. Installed `lora/worker_manager.py:245` loads an adapter for an unseen
ID instead of reusing a cached adapter. `v1/core/kv_cache_utils.py:514` includes
the adapter ID in prefix-cache identity. Prefix caching is explicitly enabled
in the launcher, but a new ID prevents cross-turn same-adapter cache reuse.
Sleep additionally discards non-weight allocations, including KV storage.
Keep one adapter ID for an unchanged policy version within a session; invalidate
on actual policy updates. This does not guarantee prefix hits for changed prompts.

## 3. HF output-head work includes prompt positions discarded later

With remove-padding enabled and fused kernels disabled, `dp_actor.py:145`
forwards all nonpadding positions, computes full-vocabulary logits, logprobs
and entropy, then returns only response positions at lines202-206.
Vocabulary size is152064. Real R9 raw data has462363 total positions but106965
response positions: about76.9% are prompt positions. The processed PPO batch
has the same proportion. One contiguous16-row chunk can materialize a
3.14GiB BF16 logits tensor before other buffers, gradients or entropy work.

Candidate optimization: gather response-predicting hidden positions before
LM-head projection, or use an appropriately verified chunked/fused head.
Transformer prompt attention remains necessary. Preserve the one-token shift,
packed sequence boundaries, full-response loss and entropy coefficient.001;
verify losses/gradients and the existing probability gates. The4.31x position
ratio is NOT a predicted whole-training speedup.

## 4. Diagnostics incur additional full forward passes

Native pre-update scoring computes entropy for logging. Our `after_update`
also calls native `compute_log_prob`, which always computes entropy even though
the post-update observer only reads logprobs. Retain full-batch pre/post gates;
consider removing unused post-score entropy before removing scoring itself.
Actor entropy regularization must stay. R9 has about41s pre-scoring plus a
separate post-update pass; the latter is included in total iteration time.

## 5. Smaller costs and measurement limits

- Completed environments are generated before filtering:29/800 response
  requests in R9,43/800 in R11. Token-cost fractions are unknown.
- `preprocess_batch` is serial despite its parallel-processing comment;
  prompts are tokenized once for input tensors and again for raw_prompt_ids.
- Eight default data-loader workers are excessive for four metadata rows;
  worker/import overhead is a secondary host-memory/startup target.
- R11 eager mode removed CUDA Graph acceleration. R13 bounded graphs passed
  numerical/memory checks, but8-context timing does not establish a full-rollout gain.
- Swap occupancy alone does not establish active swapping; measure rates/PSI.
- R8's30s stack profile attributed43.6% to gather/wake and8.8% to sleep;
  these are sampled host stacks, not synchronized phase or CUDA timings.

Priority: measure phases/call counts and implement session residency first;
then validate response-only output-head work, active-row compaction and smaller
CPU/diagnostic savings independently. Keep old failures and candidate scores
separate. Four-round learning remains incomplete and CTHA remains disabled.
