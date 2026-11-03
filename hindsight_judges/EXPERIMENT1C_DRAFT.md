# Experiment 1c (protocol v1): Disentangling hindsight evidence from the executed-action pointer

**Freezing rule:** this file becomes the frozen 1c protocol at the commit the researcher passes to `score --approved-commit`. Nothing below may change after the first formal score.

**v1 changes from the review draft (2026-11-03, before any 1c GPU work):**

1. **Execution layer — chunked prefill under SDPA MATH.** E1 32B peaked at 68.6 GB on the 85 GB H800 with ≤ 1.2k-token contexts; ~67 GB is weights. MATH materialises a heads × N × N score tensor per layer (FP32 for BF16 inputs). The original L contexts on these 386 points are ~11–13k tokens (one score tensor alone ≈ 40 × 12k² × 4 B ≈ 23 GB), so one-pass scoring cannot fit. `exp1c_backend.py` wraps the frozen P4 scorer's model and feeds the prefix in 1,024-token chunks through the same cached-attention path the frozen scorer already uses for candidate suffixes. New gates: chunked shared-prefix vs chunked direct (longest context first, then one per condition) and chunked vs frozen one-pass on the shortest context of each condition that is ≤ 4,096 tokens (≥ 3 required); all TV < 0.005. A unit test compares chunked and one-pass logic token by token on a tiny random Qwen2 model wherever torch + transformers are installed.
2. **Two single-branch conditions `D1F` / `D1B`** (M + only f's or only b's anonymous replay). They test the display-only pointer directly: does the judge move toward the one action whose continuation it was shown, even when that continuation failed? This is the controlled version of the original observation that failed factual trajectories still raise the executed action. 386 × 9 = 3,474 requests.
3. **Endpoints frozen** (section 4): H1 identity label under symmetric two-branch evidence; H2 pull toward a displayed failed branch; H3 value harm of the true label under symmetric evidence. D-vs-M/L value contrasts are descriptive only (oracle competitor, section 3).
4. **Label wording** uses the verified E1 verb: "the original agent executed the first action of branch X" (a replay branch is not the historical trajectory). `LT` gets the same trailing newline as the frozen L/M templates.

**2026-11-03 · No 1c GPU scores or training exist yet.** This is a new, prospective *proposal* informed by P4 E2 and the completed hindsight experiment 1. Its choices must be approved and frozen **before** using the new judge scores. It does not modify the original P4/E4/E4b/E4c or the earlier hindsight E1 protocol/results.

## 1. Scientific question and existing evidence

The original P4 E2 supplied a judge with an **entire realized factual episode** (`L`), including the Agent's generated response text (often containing visible `<think>` blocks), executed actions, observations and terminal outcome. E2-M supplied a full **decision-preceding history** (`M`) without any future. These inputs changed *both* hindsight world evidence and whether the marked decision could be identified as previously executed. P4 post-hoc analysis found an L-associated factual-action preference shift and, in some value-defined subsets, lower independent counterfactual value; however, `L−M` does not identify the effect of factual identity by itself.

The independent E1 label swap established that an explicit identity cue (or even an arbitrary marker) can causally shift a judge's preference, and that this can hurt held-out q3456 value when the marked action is worse. **E1 did not include the complete factual post-decision trajectory.**

**E1c's core questions:** Does this kind of cue matter when the judge sees full, symmetric counterfactual continuations? Can an **anonymous, evidence-bearing** representation preserve useful future observations while avoiding a preference for the historically executed action?

## 2. Frozen upstream cohort, arms, and environment

- Reuse exactly the **386 P4 I/U Live Fork decisions** (184 I + 202 U), identified by the audited `hindsight_judges/results/exp1_p4/pointmeta.json`; fixed source hash from experiment 1.
- Each point has five original candidate actions; **f = arm 0**, and **b = the same q12-optimal alternative arm** chosen before E1 scoring (smallest arm number on ties). The new experiment must not reselect the competitor using new outputs or q3456.
- Use **P4 replay bank 1, replica 0** for f and b. Bank-1 outcome is *not* used to select which continuation to display. This is one realized continuation per displayed branch, not an exact action value. **Disclosed overlap:** b itself was chosen by q12 (banks 1–2), which contains this displayed sample, so b's displayed branch is slightly optimistic: on these 386 points the bank-1 success rate is 0.788 for b versus held-out q3456 0.742 (f: 0.383 versus 0.364). All values are evaluated on q3456 / Gold, which share no samples with the display.
- P4 bank-1 `data/replay/{traj}/t{turn}.json` stores the complete *action sequence* and success outcome but not observations. The **original E2 L already contains every factual prefix action and observation**, and `data/task_list.json` maps the original 100 task seeds to game paths and SHA256. Rebuild the factual prefix directly from these tracked sources, then restart the original ALFWorld game, refeed the prefix, execute each stored bank-1 continuation action, and record observations. Validate the game's SHA256, refeed divergences = 0, marked-state observation, first-step observation against P4's aligned F evidence, terminal outcome and 50-step episode budget. Any original `data/rollout/*.json` on disk is an *optional additional consistency check* rather than a hard dependency; GitHub currently contains only the first 50 despite the cohort including up to s099.
- The 50-step episode cap is counted from the *start of the task*, not as 50 extra steps after the marked decision. Both branches have identical initial state and remaining budget but may end at different times due to their actions.
- **Source-data constraint:** `data/replay/` is excluded from Git and must be supplied from the original P4 server data. Never substitute E4's different task population or newly sampled policy trajectories. If absent, the CPU preflight stops; it does not silently construct fabricated branches.

## 3. Judge inputs (nine conditions, exactly 386 × 9 = 3,474 requests for 32B)

All conditions use the original P4 base judge instruction, the original five candidate actions in their original randomized display order, and the same 32B judge checkpoint and scoring semantics (five full candidate sequences, log-prob sums, temperature-1 softmax). **Re-score every condition in one new machine/configuration; no subtracting old RTX6000D/H800 probabilities.**

| Condition | Contents | What it isolates |
|---|---|---|
| `M` | Original P4 E2-M decision-preceding full history, byte-identical | History-only reference; no future |
| `L` | Original P4 E2 full factual episode, byte-identical | Existing complete hindsight treatment, re-scored |
| `LN` | Same L with **visible generated `<think>…</think>` spans removed** only from episode `steps[].response`; original executed actions, observations and terminal reward kept | Contribution of *written* reasoning/narration, not hidden internal CoT |
| `LT` | Same original factual episode **through turn t inclusive**, with executed action and resulting observation, but no subsequent steps or final reward; header says truncated | Effect of future trajectory beyond the marked action's immediate feedback |
| `D0` | Original M plus **two equally formatted complete P4 bank-1 replays**, f and b, randomly ordered and anonymously named branches A/B; environment observations and outcomes retained | Hindsight evidence in a symmetric presentation; **not identical world evidence to L** |
| `DE` | D0 **plus** one provenance line saying which of A/B was the historical action at t | Identity cue with full symmetric evidence held fixed |
| `DX` | Same as DE except the provenance line names the *other* branch; all traces, steps, outcomes and order identical | Swapped identity cue under exactly matched evidence |
| `D1F` | Original M plus **only f's** bank-1 rep-0 replay, anonymous, no provenance statement | Display-only pointer toward f |
| `D1B` | Original M plus **only b's** bank-1 rep-0 replay, same layout | Display-only pointer toward b (mirror of D1F) |

Provenance line (DE/DX), identical except the letter: *"Execution record for the marked decision: the original agent executed the first action of branch X."* The single-branch intro says nothing about who chose the displayed action (in E1 an "evaluation marker" was itself a strong pointer).

**Oracle competitor.** Only f and the q12-best b receive branches. Showing b's branch therefore points at an arm the experimenter chose with q12: on these points q3456(b) = 0.742 versus 0.425 for the average alternative. Within DE/DX/D0 this is held fixed, so H1 and H3 are clean; any value comparison of a D condition against M or L includes this selection and is reported as descriptive only.

`D0` deliberately does **not** identify which branch is historically factual, but the judge can still see which original action is the first action of each branch. Branch A/B order is deterministically randomized by point ID and fixed independently of rewards; metadata is not allowed to disclose internal arm index, q values or branch source role. The other three candidates **do not receive full continuations**; this is a **two-arm evidence intervention** within an unchanged five-action judgment problem. Report this limitation. `DE/DX` differ by one branch letter, never by the evidence.

For LN, P4 rollouts often include plain-text `<think>` content from the Agent's generated *responses*. The judge itself is **not** required to generate chain-of-thought. LN is not ablation of hidden model reasoning. If the CPU audit finds zero text `<think>` spans, LN is non-operative and the fact must be reported, not invented.

## 4. Endpoints (frozen at the approved commit)

Values use held-out q3456 (Gold as an independent check); q12 only fixes b and the predefined lower-value subset `q12(b) − q12(f) ≥ 1/3`. Statistics are per decision point with a trajectory-cluster bootstrap, 4,000 resamples, seed `20261103`. Log-odds compare the same candidate strings within a point (`ℓ_c(a)` = sequence log-prob).

**H1 — identity label under symmetric full-trajectory evidence (primary).**
\[ D_{id}=\tfrac12\{[\ell_{DE}(f)-\ell_{DE}(b)]-[\ell_{DX}(f)-\ell_{DX}(b)]\} \]
on all 386 points; supported if the 95% CI lower bound > 0. Secondary: the same restricted to points where both arms are below the log-prob ceiling in D0 (`ℓ_D0 < −1`), and by displayed outcome (f won / lost × b won / lost).

**H2 — display-only pointer on failed evidence (primary).** For each displayed branch that **failed** (f's in D1F, b's in D1B), the pull toward the displayed arm relative to M and to the three never-displayed arms:
\[ \text{pull}=[\ell_{D1}(a)-\ell_{M}(a)]-\operatorname{mean}_{o\notin\{f,b\}}[\ell_{D1}(o)-\ell_{M}(o)] \]
pooled over failed instances; supported if the CI lower bound > 0. An information-only judge should not move toward an action after seeing its continuation fail. Secondary: headroom-restricted version (`ℓ_M(a) < −1`), the same pull for succeeded branches, and `ℓ_D1F(f)−ℓ_D1F(b)` minus the M value on f-failed points.

**H3 — value cost of the true label under symmetric evidence (primary).** On the q12-worse subset: `V(DE) − V(D0)` with `V(c) = Σ_a p_c(a) q3456(a)`; supported if the CI upper bound < 0. Secondary: `V(DE) − V(DX)`, Gold.

**Descriptive (no decision attached):** `V(L)−V(M)`, `V(LN)−V(L)`, `V(LT)−V(L)` and the matching f:b log-odds shifts; `D0−M`, `D0−L`, `D1F−M` values (oracle caveat); argmax = f by condition; slopes of `L−M`, `D0−M`, `D1F−M` log-odds shifts on held-out `q3456(f)−q3456(b)` with trajectory-bootstrap CIs. Groups: all, I, U, q12-worse, Gold, Gold q12-worse.

**Reading rules.**
- H1 and H2 supported: the executed-action pointer survives full symmetric trajectories, and showing one branch is enough to create it even when that branch failed.
- H2 supported, H1 not: explicit provenance stops mattering once full symmetric evidence is shown, but one-sided display creates the bias; the fix is symmetric display.
- H2 not supported: the failed-trajectory anchoring in L is not reproduced by display alone; look to L-specific content (LN, LT).
- H3 supported: the label costs decision value even with both full branches on screen.
- Non-significance is never described as absence or equivalence.

## 5. Code & data gates before the first real GPU score

1. Run `python hindsight_judges/test_exp1c.py` plus built-in `selftest` on CPU. Synthetic fixture covers complete `preflight → prepare → analyze`, exact equality of `DE/DX` evidence aside from provenance letter, fixed bank-1 rep-0 selection (regardless of observed success), deterministic refeed, and outcome consistency.
2. Run `preflight` against the original P4 bank-1 directory and original ALFWorld game files. Missing sources are a **hard stop**. No GPU and no policy model needed.
3. Run `prepare`, which validates original P4 request hashes, fixed pointmeta, `task_list` and individual game SHA256s, per-point P4 replay hashes, reconstructed factual-prefix state, first-step feedback and terminal outcomes. Original rollout JSON is cross-checked only if present. Persist audit JSON and reproducible request hash. **CPU-only but requires installed ALFWorld/TextWorld**.
4. Run `lengths` with the original 32B tokenizer and all 3,474 rendered requests. Anything exceeding **32,768** tokens blocks scoring. **No automatic truncation, shortening of only one branch or silent point exclusion.** Review any failure and amend/freeze before GPU.
5. **Approval gate:** the GPU scoring command requires an explicit `--approved-commit` matching the clean current HEAD for the script and protocol. If not approved, the CLI refuses to start. If approved: score the 32B model only, one model load with SDPA MATH throughout (same numerical mode as verified E1 32B), prefix fed in 1,024-token chunks (`exp1c_backend.py`, change 1). Preserve frozen 32B checkpoint hashes and original P4 scoring backend hash; record the chunked backend hash and chunk size. Before any formal score: chunked shared-prefix vs chunked direct TV **< 0.005** on the longest context (first, so an out-of-memory error surfaces immediately) and one example per condition; chunked vs frozen one-pass TV **< 0.005** on the shortest context of each condition that is ≤ 4,096 tokens, at least 3 such checks. An out-of-memory error stops the run with no score file. Keep every condition on the same GPU configuration; record source/weights/code/hardware hashes. No extra policy rollout and no training.
6. Analyze only once all 3,474 requests are complete. Never replace the completed experiment-one results or edit its E1 files.

## 6. What this experiment cannot establish

- Equal-format paired trajectories are a counterfactual *presentation*, not a magically bias-free truth. The two shown branches are sampled continuations of the same ongoing policy, with outcome variance and no evidence for the other three candidates. Their random seeds and natural lengths can differ.
- `L−D0` changes not only identity cues but the number and type of observed continuations and the visible reasoning text, and D0 shows an oracle-chosen competitor. The **only direct identity manipulations** in 1c are `DE−DX` and (to a lesser extent, due to new words) `D0−DE`; the **display-only pointer** is H2 (D1F/D1B).
- One environment, one 32B judge, one fixed P4 cohort. There are no training runs; this cannot show that any bias forms a self-confirming reinforcement loop.
- For technical feasibility, the *original P4 bank-1 files and ALFWorld games* must be accessible to the server. A passing synthetic test only confirms code plumbing. Complete real-data preflight and 32B context-length measurements are still required before GPU use.
