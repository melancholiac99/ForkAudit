# Repository Guidelines

## Project Structure & Module Organization

ForkAudit audits step-level credit in ALFWorld agents. `scripts/` contains numbered rollout, replay, scoring, and evaluation stages plus shared Python modules. `data/`, `prompts/`, `eval/`, and `logs/` hold inputs, exact prompts, audit outputs, and execution records. Each `exploratory_*/` directory contains a separate experiment, implementation, tests, and reports. `papers/`, `proposals/`, and `review_sources/` contain writing and supporting decisions; `env/` records environment verification artifacts.

## Build, Test, and Development Commands

There is no package build or unified dependency manifest. Use Python 3.10+; CPU CI uses Python 3.11. Install dependencies appropriate to the experiment; model execution requires additional GPU libraries.

- `python -m py_compile exploratory_ctha/*.py`: check CTHA syntax.
- `python -m unittest discover -s exploratory_ctha -p 'test_ctha*.py' -v`: run CTHA CPU tests.
- `python -m unittest discover -s exploratory_credit_projection -p 'test_audit.py' -v`: test credit projection auditing.
- `python exploratory_E4c/test_e4c.py`: run standalone E4c CPU checks.
- `bash scripts/run.sh 12_audit_replay.py`: audit stored Phase 1 records. The wrapper changes into `scripts/`; inspect its hardcoded Conda, data, and cache paths before use.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, and descriptive module names. Preserve numbered pipeline filenames and experiment-local conventions. Prefer `pathlib.Path` for new path handling. No repository-wide formatter or linter configuration is present; follow surrounding code.

## Testing Guidelines

Tests use `unittest` and standalone assertion-based runners. Keep `test_*.py` beside the relevant experiment and run its suite before proposing changes. Cover changed scientific logic, isolation guards, and artifact validation. No coverage threshold is configured. Synthetic CPU checks do not establish benchmark performance.

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive subjects such as “Validate square-bracket action spans before CTHA v3 training.” Follow that pattern. PRs should explain the affected experiment, behavioral change, validation commands, and artifact or protocol implications; link relevant decisions or proposals.

## Research Integrity & Configuration

Read `CLAUDE.md`, applicable rulings, `FEATURES_FROZEN.md`, and experiment preregistrations before changing research behavior. Record unspecified research decisions and rationale in `DECISIONS.md`. Preserve frozen definitions and verdict rules; report exploratory additions separately. Record seeds, generating scripts, commit hashes, and input checksums. Keep model weights and excluded raw replay files out of Git.
