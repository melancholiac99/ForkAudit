# Phase 0.1 — CWGT archive search log

Date: 2026-10-22

## (a) arXiv TeX source — DONE
- `curl https://arxiv.org/src/2608.19760` → `downloads/cwgt_src/src.tar` (gzip, 234,537 B,
  sha256 739b27da897cd3d6ae86ad6fcb2d823faf572c54ab243f35aaa51b003f8a255c), extracted to `downloads/cwgt_src/x/`.
- Title confirmed: "Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay". ICLR 2027 style file.
- `grep -iE "http|anonymous|4open|osf|zenodo|github|huggingface"` over all .tex/.bib/.bbl/appendix_sources:
  the only URL in the entire source is `https://arxiv.org/pdf/2509.14480` (TARL citation).
- The paper states (Sec. 9 reproducibility, App. "Release inventory") that the replay archive (29,402 files, 4.7 G),
  per-turn `summary.json` (17,682), logprob dumps `.pol/.hind` (4,601), generator and ledger are released
  "through an anonymized repository" — **no link is given anywhere in the arXiv source.**
- Useful bundled material (not the archive): verbatim HCAPO ALFWorld template, judge prompts, all pre-registrations
  (`KILLSHOT_PREREG.md` contains the metric cards incl. exact T3 definition).

## (b)–(e) — NOT YET DONE
Paused at the Appendix Q stop condition (see DECISIONS.md D-001). Remaining: arXiv Code/Data tab, alphaXiv,
Papers with Code, Semantic Scholar; OpenReview (ICLR 2027) title search; GitHub search; author email.
