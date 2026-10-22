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

## (b) arXiv page / aggregators — nothing
- arxiv abs v1 (2026-08-20, "49 pages … Under review") and v2 (2026-08-25, "52 pages"): no repository link in static HTML.
- HF arXiv-linked repos API: empty; HF papers API: 404. alphaXiv resources/implementations: empty.
- Papers with Code: defunct (redirects to HF). Semantic Scholar CorpusId 291257089: arXiv ID only.
- awesomepapers.io page labelled "Code & Paper": HTTP 429, unverified.

## (c) OpenReview — revised version found, supplementary not accessible
- **Me3WCeeMWW**, ICLR 2027 submission (created 2026-09-18): "Credit Without Ground Truth: Claim-Matched Evaluation of
  Step-Level Credit with Finite Executed Replay". Redesigned study: two independently seeded replay banks, 27,520
  continuations over 172 fixed turns, 16 continuations per arm per bank; cross-bank replay-contrast Pearson 0.736,
  pooled Spearman 0.330. Has `supplementary_material` zip (/attachment/e88de196c3782e2fcd9111a18ed6eac16ca829ce.zip) —
  blocked by OpenReview's browser challenge (403 for curl/API). **Needs a human browser download.**
- Other hits (checked, not the archive): ac6BotAkVF RMPO (ICLR 2027; code anonymous.4open.science/r/rmpo-CD48),
  HbNTIXwoeU / plrHvqYzhK ACRE audit (ARR Aug/Oct 2026), sAYkoo4ufN ACRE rubric (ICLR 2027). See DECISIONS D-011.

## (d) GitHub / HF / Zenodo / OSF / 4open — nothing
- GitHub repo search (title, "credit-validity-paradox", "collect_v2 alfworld", "replay archive alfworld", …): nothing.
  github.com/gadisr/replay-credit-audit: README-only, 1 commit 2026-09-15, no link to the author — not the archive.
- HF datasets/models, Zenodo: nothing. OSF API 404. anonymous.4open.science/r/credit-validity-paradox: 401.

## (e) Author contact
- haiyuez@usc.edu (Haiyue Zhang, USC) — from the author block of https://arxiv.org/html/2608.19760v2.
- Email NOT sent (outward-facing; left to the user). Draft in DECISIONS D-011.

## Status: archive NOT obtained → Branch B (own replay), per §7 0.1(e).
Raw search artifacts: downloads/archive_search/ (fetched by a search sub-agent 2026-10-22).
