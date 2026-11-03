# Historical full-response OPD-signal diagnostic (prospective execution supplement)

User requested this supplement before its GPU scoring. Retain the running five-action
diagnostic and Think localization; do not run H-LN, OPD training, or new rollouts.
No shutdown authorization for this task. Frozen E1/E1c/P4/E4 files are read-only.

## Population and contexts

- P4: all 386 frozen I/U forks, one actually written response per marked decision;
  teachers 7B and 32B. Same original five-arm held-out q3456, q12-selected comparator,
  and Gold check. Primary executed-action advantage: q3456(f) minus mean of four
  alternatives; q3456(f)-q3456(q12-best alternative) is secondary.
- E4: audit all 6,016 original cf-defined points, retain exactly those with nonconstant
  five-arm three-replay means. Initial audit: 1,586 points, 7,930 original arm_responses;
  4,405 all-zero points. Score all five actual responses at each selected point with
  the 32B teacher. Selection uses original replay values only, before new scoring.
  This outcome-selected subset is not representative of all E4 points. Its ranking
  labels are noisy estimates from three continuations, not noise-free ground truth.
  CPU audit found two factual responses with malformed action markup (s1132_t10,
  s1251_t07). Keep their original response and the actual recorded/replayed action
  byte-for-byte; locate the unique literal action span after Think and flag it.
  Do not "repair" their action strings or exclude their groups after seeing outcomes.
- M/L/LN/F follow the current action diagnostic. M is the original decision prompt.
  L/LN mask the entire target-turn response AND executed-action label in the extra log;
  environment observations/outcomes remain. LN removes other turns' written Think.
  F shows aligned original one-step evidence for all five actions. Within an E4 point,
  ALL five response targets get identical teacher evidence; the factual realized
  episode is not replaced by invented counterfactual episodes. Student always gets M.
- Score every response from its beginning with ALL written Think retained. No
  current-Think drop intervention in this supplement. Teacher conditions never
  reveal the target response by copying it into the extra log.

## Token identities, definitions and limits

Historical files contain response TEXT, not original generated token IDs. Freeze
canonical re-encoding by the original shared Qwen tokenizer, require exact text
round-trip and identical 7B/32B vocabulary, chat template and response IDs. Do not
claim recovery of the original generation's BPE choices, sampling probabilities,
temperature/top-p distribution, or a fresh on-policy sample. Do not append invented
EOS tokens. All recorded response text, including format tokens, is scored.

At position t use only response tokens preceding t, never t itself:

- sampled-token proxy delta_t = log T(y_t | teacher context, y_<t)
  minus log S(y_t | M, y_<t), using the frozen 7B student reference;
- exact full-vocabulary reverse KL = KL(S || T), matching exp2_online.py's default;
- exact full-vocabulary forward KL = KL(T || S), reported separately.

Use FP32 log_softmax and sums, no top-k, probability floor, five-arm renormalization,
or clipping. Store both model log-probabilities and both KLs at EVERY response token.
Split into Think content, action content, and format/boundary/other tokens; whitespace
attached to an action-content BPE token belongs to that action token. The three
sets partition all tokens. Report sums and per-token means for each and the full reply.
The sum of deltas is a sequence log-density-ratio proxy, NOT the implemented exact-KL
loss, a measured gradient, or proof that training will reinforce/improve a policy.
The historical sampling and action-dependent future context further preclude that
causal interpretation. More Think tokens alone do not prove most gradient norm is Think.

## Computation and provenance

7B uses the already documented frozen-weight FP32 recovery, 32B the E1c parameter-only
FP16 recovery with original buffers and FP32 head; SDPA MATH, TF32 off, chunk 1024.
Keep each model/cohort on one recorded GPU UUID/configuration. No old probability reuse.
Cache final FP32 student hidden states and sampled logps on disk (not a full-vocabulary
distribution dump). Re-project the identical frozen FP32 student output head in small
blocks alongside teacher logits to compute exact KL. Cache/output hashes and software,
source, tokenizer, model, precision and device fingerprints prevent silent resume mixing.
Teacher prefill is reused across responses within a point/context. A separate full-prefix
chunked path checks cached-prefix scoring on shortest and longest contexts. Require
maximum full-vocabulary per-token TV <0.005 on every tested response position. A short
one-pass check is also required. Failure stops BEFORE any formal output for that stage;
save diagnostics and record any necessary repair before restarting. No threshold changes.
The cache projection must reproduce saved student logps within 1e-4 nat. Require finite
logps and KL >= -1e-5 (small floating-point tolerance; no clipping in reported values).
No truncation beyond a length cap: any >32768 context aborts with its precise ID.

## Analysis fixed before new scores

Use 4,000 cluster bootstrap replicates, seed 20261104; P4 trajectory clusters, E4 game
clusters. Report confidence intervals as exploratory, not a newly invented confirmatory
test or a familywise-corrected discovery. No binary mechanism claim from logp alone.

P4: per teacher/context, report response/segment delta sums and means, both KLs, slope
of sequence proxy vs independent q3456 advantage (and Gold), positive-proxy fraction on
negative-advantage executed actions, and paired L/LN/F minus M proxy changes. Repeat
alignment using length-normalized proxies; distinguish segment length from signal size.

E4: per point/context rank five responses by sum delta, and separately mean delta;
report within-point Spearman correlation with replay value, pairwise concordance on
unequal-Q pairs (score ties get 0.5), and value of the top-proxy response minus uniform
five-arm value (score ties averaged). Compare contexts paired within the same points.
Report factual versus alternative proxy differences and segment contributions without
mistaking the subset-selected ranking result for held-out policy improvement.

Save numerical checks, per-token arrays, aggregate results, sample counts, all exclusions
(none may be silent), and independent completion verification. Any added precision or
context amendments must be recorded explicitly; do not revise science after reading scores.
