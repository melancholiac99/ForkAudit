# Post-main diagnostic: ensure actual Teacher sees the true location clue

This diagnostic is specified AFTER inspecting the complete main probe. It is
not a preregistered extension and does not replace its results. No prompt search.

Reason: only 4/8 mechanism-Fork factual trajectories mention the clue object in
their observations. Ordinary Hindsight therefore does not reliably instantiate
the intended condition of a teacher explicitly informed of the target location.

Score exactly the same 24 development points once, adding the previously
validated true location clue to Hindsight. Label it evidence for the teacher,
explicitly preserving the original student continuation information. Keep the
same actual thought prefix, candidate commands, syntax, and numerical scorer.
Do not generate outcomes, modify the native continuation, or train a student.

Compare this H+clue condition (C) to the already saved S/H/A/X probabilities using
the existing native/completed 16-trial outcomes. Report all 24 cases and all old
strata. If C improves native recommendations, the assertion that merely making
this teacher more informed erases useful action preferences is not supported.
If C harms native but not completed utility, that supports the proposed risk in
this controlled prompt setting. No additional diagnostic follows automatically.

Budget: 24 shared-prefix contexts / 120 candidate sequences on the same single
GPU, plus numerical checks. The earlier main scoring code is archived before
adding a directory option; the likelihood implementation is unchanged.
