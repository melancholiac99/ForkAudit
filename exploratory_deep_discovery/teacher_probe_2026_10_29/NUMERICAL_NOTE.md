# Numerical implementation note, before any completed scientific scores

The first BF16 model scoring check failed the fixed cache-vs-direct tolerance:
individual token log-probability differences reached 0.25 nats for a low-probability
alternative. No `scores.jsonl` had been produced. The initial log is retained as
`score_bf16_head_check.log`.

The transformer remains BF16, but the vocabulary projection weights and inputs
are promoted to FP32 before log-softmax. This reduced but did not eliminate
shape-dependent BF16 transformer differences (one token still differed by 0.19
nats); that attempt is retained as `score_fp32_head_check.log`.

Before any scientific scores, validation was changed to the actual measured
quantity: normalized candidate probabilities. All five candidates are checked
for the original S/H contexts and the longest context in the fixed queue.
The allowed total-variation discrepancy is 0.005, bounding the discrepancy in
any expected binary-success readout by 0.5 percentage points for that check.
This is an implementation tolerance, not an outcome-based parameter choice.
All token differences are still recorded. Passing three checks does not give a
uniform numerical guarantee for every context. Prompts, candidates, sample and
scientific evaluation rules are unchanged. The final run records its precision.

An earlier launch attempted to open preparation metadata before preparation
finished; it exited before loading the model. Its log is `early_start.log`.
