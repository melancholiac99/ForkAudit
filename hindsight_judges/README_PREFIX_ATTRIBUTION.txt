Prefix attribution v2: eight conditions, protocol a2f1226 (2026-11-05)
=================================================================
Protocol: PROTOCOL_PREFIX_ATTRIBUTION_20261105.md, unchanged.
Branch: research/hindsight-judges. All output uses a new dedicated directory:
  hindsight_judges/results/prefix_attribution_v2_20261105/
No original P4/E1/E1c/E4, action/token diagnostic or v1 artifacts are overwritten.

Conditions and rendering
------------------------
AS/AO: original BASE plus exact self/other current-step source cue, then active R+TAG.
KEEP/DROP: exact original prompt, prefix, action lead/tail, EOS and token-boundary scoring.
QS/QO/QN: first user = BASE + source cue + verbatim quoted R;
          second user = 'Output the next legal action.'; active assistant = TAG.
AP: first user = BASE + same self cue as QS; completed assistant = exact R;
    second user = identical request; active new assistant = TAG.

All messages are real structured chat messages. AP's R ends with the tokenizer's
assistant <|im_end|> before the new user request. No fake role tokens are injected
inside user text. QS/QO/QN use identical quote delimiters. Cues exactly match the
revised protocol examples (no 'earlier attempt'). R's bytes and all original action
formatting are preserved. The s071_t01 outside-Think action marker is deliberately
retained and reported, not silently cleaned. No factual action is added to a quote.

CPU preparation and historical reproduction
------------------------------------------
Use the existing /root/miniconda3/envs/m0/bin/python (numpy, torch, transformers).
From repository root:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m unittest discover -s hindsight_judges -p test_prefix_attribution.py -v
python hindsight_judges/exp4_prefix_attribution.py prepare \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --arms /root/autodl-tmp/idea/exploratory_deep_discovery/exploratory_arms.parquet

Preparation checks original source hashes, 386 points / 60 trajectories, all 6,176
requests per model, tokenized roles/boundaries and all 1,544 historical anchor inputs.
Historical raw score checksums and softmax probabilities are independently checked;
historical four M/F × KEEP/DROP headline values and centered slopes are recomputed
from the existing score files and frozen q3456, without any GPU or rewritten values.
The old worktree's parquet path is required because this worktree has no local copy.
Missing input paths, invalid parsing or checksum mismatches stop execution.

Outputs: requests.jsonl.gz, prepare.json, condition_matrix_audit.json,
exact_prompt_examples.json (all 16 prompts), historical_cpu_audit.json.
The two model tokenizers must match exactly; each request token hash is checked again
before scoring. No historical score is imported as a new v2 score.

Pilot before full run
---------------------
Pin a single recorded GPU UUID (not numeric index). Run a separate first32 through
first64 pilot for EACH model. CPU tests run automatically before model allocation.
CUDA_VISIBLE_DEVICES=GPU-actual-uuid python hindsight_judges/exp4_prefix_attribution.py score \
  --name 7b --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --limit 32
python hindsight_judges/exp4_prefix_attribution.py analyze --name 7b --limit 32
python hindsight_judges/exp4_prefix_attribution.py verify --name 7b --limit 32
Repeat with --name 32b --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct.

The direct-versus-cache gate covers global shortest/longest plus shortest F request
of all eight conditions. Every candidate TV must be strictly below 0.005. Precision
is unchanged: frozen 7B FP32; frozen 32B FP16 transformer + FP32 head, original buffers,
SDPA MATH, chunk 1024, TF32 off. No truncation, clipping or outcome-selected conditions.
Pilot/full and resumption require identical model/code/source/token/hardware/runtime
fingerprints. Full execution requires a completed independently verified pilot and
historical anchor compatibility. Measure pilot seconds/context and VRAM before ETA.

After both pilots pass, omit --limit for separate full 386-point runs. --pilot-size 64
selects a first64 pilot (default first32). Each full model has 6,176 contexts; both
models total 12,352. Neither pilot scores nor historical scores enter the full result.

Scoring/analysis artifacts under <7b|32b>/<first32|full>/:
  scores.jsonl: five raw logp and probabilities, exact request/Q, hashes, time per row
  run.json, numerical_gate.json, progress.json: fingerprints, gate, resource usage
  historical_anchor_comparison.json: every matched historical/new raw score/probability,
    delta, TV; mean/max TV, all nonidentical/review IDs and both headline values
  point_metrics.jsonl.gz: all raw E/Q gaps and centered five-arm sufficient statistics
  analysis.json, REPORT.txt: absolute values, all contrasts, CI and limitations
  verification.json: independent raw-score primary contrast/CI and slope/ratio audit

Anchor compatibility review uses the existing 0.005 TV scale conservatively as a
NUMERICAL REVIEW trigger, not as a scientific outcome threshold. Any historical
candidate TV >= 0.005 blocks verification and full progression pending investigation.
All raw logp differences (including low-probability tails) are still reported. A
bitwise difference alone is not called a scientific change. Old files remain read-only.

Statistics and interpretation
-----------------------------
Co-primary F contrasts: SLOT_F = QS−AP; P_Q_F = QO−QS. Report their M companions and
F−M interactions I_SLOT/I_Q. Report AP−AS, QN−QS, QN−QO, AO−AS, QS−AS, DROP−KEEP,
legacy extra contrasts, F−M in each condition, factual argmax/p_f and bad-factual
correction rates. Value uses percentage points; rates use 0–1. Paired point estimates,
original trajectory clusters, 4000 draws, seed 20261105, 95% CI, no multiplicity correction.

Unclipped E_j = (ell_F,j−ell_F,0)−(ell_M,j−ell_M,0) is compared with Q_j−Q_0;
report within-point sign concordance (zero E = .5), aligned nat and through-origin
slope. All-Q-tied points stay in V but have undefined pairwise alignment.
Additionally, center raw delta=(ell_F−ell_M) and Q over ALL FIVE arms within each point.
Pooled slope = sum_i,j centered_Q*centered_delta / sum_i,j centered_Q^2.
Paired slope contrasts and slope(c)/slope(DROP) use the SAME trajectory draws.
If the DROP slope bootstrap CI includes zero or any draw has zero denominator, no
bounded ratio CI is claimed; absolute slopes and undefined-draw counts remain visible.
Historical 0.55/0.36 are not assumed comparable: the historical formula is recomputed
explicitly. Bad factual means Q_f < max(Q); top-score ties average correction rates.

QS−AP matches restart/request but also changes message/history/quotation structure.
QO−QS tests source wording, not internal identity. AP−AS includes restart/request.
Favorable action values or rankings never establish useful OPD gradients or training
benefit. No new trajectories, HLN or training are part of this task.
