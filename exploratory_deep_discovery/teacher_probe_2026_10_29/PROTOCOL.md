# Actual hindsight teacher probe — 2026-10-29

Exploratory, frozen before model scoring. This is a diagnostic experiment, not
training and not an official reproduction of an existing Hindsight OPSD method.
No new environment continuations are required.

## Question

Do hindsight-conditioned action preferences become less useful for the original
student, and can correctly aligned one-step branch evidence improve them?
The earlier native/completed outcome-derived rules were not actual teachers.

## Fixed sample and four contexts

Reuse all 144 points in the previously prepared bundle: 24 mechanism-development
points and 120 historical-main points. Tasks are disjoint across these cohorts.
No prompt tuning on development teacher results is planned in this experiment.

- S: original student prompt only.
- H: original prompt plus complete factual episode and its terminal reward.
- A: H plus all five correctly aligned one-step action/observation records.
- X: same as A, with a fixed derangement of the four alternative observations;
  the factual action/observation remains fixed. A and X share display order,
  feedback multiset, instructions and candidate commands.

The full episode includes initial observation, every actual response (including
thought), executed action, subsequent observation, and final reward. It is
explicitly retrospective evidence. All H/A/X conditions use the same decision
instruction and information-boundary statement. No replay rates, cue-completed
outcomes, mechanism labels, or hidden environment facts enter model input.

## Scoring

Qwen2.5-7B-Instruct, frozen base checkpoint, BF16 transformer with FP32 vocabulary
log-softmax, raw probabilities (temperature 1). Preserve the actual student's
thinking text and actual action opening delimiter for each point. Score all five
complete commands with the same closing delimiter convention and EOS. Normalize
summed sequence log probabilities over the five candidates. No mean-token scoring.

Tokenization is performed on complete text and split only at a shared token
prefix, avoiding artificial boundary tokenization. Shared-prefix KV scoring is
checked against direct full-sequence scoring on fixed examples before the main run.
The probabilities are conditional on the realized thought and the fixed candidate
set; they are not free-generation action marginals. This probe cannot rule out
useful supervision of earlier thought tokens by Hindsight OPSD.

## Evaluation and interpretation

Main: the existing 80 Live points with banks 5+6 (6 trials/arm), also report the
63 points where all original candidates are legal. Gold-covered points are a
secondary outcome source. The 40 old Dead points lack independent banks 5+6 and
are reported only as teacher-behavior diagnostics; no zeros are imputed.

Development: all 24 points have native/completed 16-trial five-arm outcomes.
Report every old stratum, explicitly noting 8 tasks per stratum and cue enrichment.

Readout J(p,q)=sum_a p(a) q(a), conditional on candidate choice, followed by the
original student continuation. Report H-S, A-H, A-X, candidate probability mass,
chosen mass, legal-restricted sensitivities, and representative cases.
Task-cluster bootstrap 4000 draws, preserving paired conditions. No statistical
threshold triggers more replay or prompt search. Historical selection used some
evaluation banks; independent outcome seeds do not make this confirmatory.

Evidence for the strong mismatch hypothesis would require actual H preferences
to be worse for native continuation while comparably useful under completion,
with corresponding branch-preference changes. A improvement alone does not show
that hindsight harmed the student. X may be semantically inconsistent; it tests
sensitivity to action alignment, not a pure causal mediation effect.

If H is already useful and A adds nothing, report that and do not claim that the
earlier proxy analysis demonstrated a real teacher failure. No training is
authorized by this protocol; the user authorized this one-card diagnostic run.

## Budget and outputs

576 shared-prefix contexts / 2880 complete candidate sequences, plus a small
numerical scoring check. One 96GB GPU. Measure startup, scoring time and peak
memory; do not extrapolate from environment-continuation throughput. Stop and
release the model after scoring. All new files remain in this directory or a new
report; frozen data and earlier prepared bundles are read-only.
