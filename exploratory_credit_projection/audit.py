#!/usr/bin/env python3
"""CPU-only, game-held-out audit of E4/M0 five-arm rollout credit.

Important: output measures out-of-sample predictability of *noisy* replay
values, not a proven 'realizable fraction', causal training utility, or EVPI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Point:
    point_id: str
    group: str
    prompt: str
    actions: tuple[str, ...]
    outcomes: tuple[tuple[int, ...], ...]


class JoinedParts:
    """Read a split .tar.gz without assembling a giant intermediate archive."""

    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, size=-1):
        chunks, left = [], size
        while size < 0 or left > 0:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = open(path, "rb")
            data = self.current.read(-1 if size < 0 else left)
            if not data:
                self.current.close()
                self.current = None
                continue
            chunks.append(data)
            if size >= 0:
                left -= len(data)
        return b"".join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()


def _load_archive(root: Path, check_hashes=True):
    base = root / "exploratory_E4" / "results_archive"
    manifest = json.loads((base / "archive_manifest.json").read_text())
    files = [base / p["path"] for p in manifest["parts"]]
    for meta, file in zip(manifest["parts"], files):
        if not file.is_file() or file.stat().st_size != meta["bytes"]:
            raise FileNotFoundError(f"Missing or incomplete archive part: {file}")
        if check_hashes:
            h = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != meta["sha256"]:
                raise ValueError(f"Archive SHA256 mismatch: {file}")
    points, replays = {}, {}
    stream = JoinedParts(files)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                name = member.name.removeprefix("./").lstrip("/")
                is_point = re.search(r"(?:^|/)data/points/[^/]+\.json$", name)
                is_replay = re.search(r"(?:^|/)data/replay/s\d+/t\d+\.json$", name)
                if not is_point and not is_replay:
                    continue
                content = tar.extractfile(member)
                if content is None:
                    continue
                rec = json.load(content)
                if is_point and rec.get("cf_defined"):
                    points[rec["point_id"]] = rec
                if is_replay and rec.get("cf_defined"):
                    replays[f"s{rec['seed']}_t{int(rec['turn']):02d}"] = rec
    finally:
        stream.close()
    return points, replays, [dict(path=str(p.relative_to(root)), sha256=m["sha256"]) for p, m in zip(files, manifest["parts"])]


def _load_e4_dirs(root: Path):
    base = root / "exploratory_E4" / "data"
    points = {r["point_id"]: r for file in sorted((base / "points").glob("*.json"))
              if (r := json.loads(file.read_text())).get("cf_defined")}
    replays = {f"s{r['seed']}_t{int(r['turn']):02d}": r
               for file in sorted((base / "replay").glob("s*/t*.json"))
               if (r := json.loads(file.read_text())).get("cf_defined")}
    return points, replays, []


def _make_point(rec, replay, source, max_reps):
    actions = tuple([rec["factual"]] + list(rec["alternatives"]))
    if len(actions) != 5 or len(set(actions)) != 5:
        raise ValueError(f"{rec['point_id']}: expected 5 distinct arms")
    if replay["factual_action"] != actions[0] or tuple(replay["alternatives"]) != actions[1:]:
        raise ValueError(f"{rec['point_id']}: point/replay arm order mismatch")
    outcomes = [replay["factual_outcomes"]] + list(replay["alt_outcomes"])
    if len(outcomes) != 5 or min(map(len, outcomes)) < max_reps:
        raise ValueError(f"{rec['point_id']}: need >= {max_reps} outcomes per arm")
    selected = tuple(tuple(int(x) for x in a[:max_reps]) for a in outcomes)
    if any(x not in (0, 1) for a in selected for x in a):
        raise ValueError(f"{rec['point_id']}: nonbinary success value")
    return Point(rec["point_id"], f"{source}:{rec.get('game_file') or rec['seed']}",
                 rec["prompt"], actions, selected)


def load_e4(root: Path, max_reps=3, archive="auto", check_hashes=True):
    direct = root / "exploratory_E4" / "data" / "points"
    if archive == "yes" or (archive == "auto" and not any(direct.glob("*.json"))):
        points, replays, source_files = _load_archive(root, check_hashes)
    else:
        points, replays, source_files = _load_e4_dirs(root)
    missing = sorted(set(points) - set(replays))
    if missing:
        raise ValueError(f"Missing replay for {len(missing)} E4 points, e.g. {missing[:3]}")
    return [_make_point(rec, replays[pid], "e4", max_reps)
            for pid, rec in sorted(points.items())], source_files


def load_m0(root: Path, max_reps=3):
    """Optional: requires unpublished M0 data/replay/ and tracked prompt files."""
    points = []
    for traj_path in sorted((root / "data" / "rollout").glob("s*.json")):
        traj = json.loads(traj_path.read_text())
        for step in traj["steps"]:
            replay_file = root / "data" / "replay" / traj["traj_id"] / f"t{int(step['turn']):02d}.json"
            if not replay_file.is_file():
                continue
            replay = json.loads(replay_file.read_text())
            if not replay.get("cf_defined"):
                continue
            prompt_file = root / step["prompt_path"]
            if not prompt_file.is_file():
                raise FileNotFoundError(f"Missing M0 student prompt: {prompt_file}")
            rec = dict(point_id=f"{traj['traj_id']}_t{int(step['turn']):02d}",
                       seed=traj["seed"], game_file=traj.get("game_file", traj["traj_id"]),
                       prompt=prompt_file.read_text(), factual=step["action"],
                       alternatives=replay["alternatives"])
            points.append(_make_point(rec, replay, "m0", max_reps))
    if not points:
        raise FileNotFoundError("M0 raw data/replay/ absent (not tracked in credit-grok); supply local original files")
    return points, []


def feature_text(prompt: str, action: str, mode="prompt_action"):
    """Only student-visible prompt + *one* legal candidate; no peeks, s, game id or outcomes."""
    if mode == "action_only":
        return "candidate_action " + action.lower()
    # Explicit type x visible-context terms let a linear regressor learn some
    # state-dependent action preferences, not just a common prompt intercept.
    words = list(dict.fromkeys(re.findall(r"[a-z][a-z0-9_]*", prompt.lower())))[:80]
    verb = re.findall(r"[a-z][a-z0-9_]*", action.lower())[:1] or ["unknown"]
    cross = " ".join(f"ctx_{verb[0]}_{w}" for w in words)
    return "student_prompt " + prompt + " candidate_action " + action + " " + cross


def crossfit(points, *, folds=5, alpha=10.0, mode="prompt_action"):
    if not points:
        raise ValueError("No replayed points")
    groups = np.array([p.group for p in points])
    n_groups = len(set(groups))
    if n_groups < folds:
        raise ValueError(f"Need at least {folds} distinct games, got {n_groups}")
    y = np.asarray([p.outcomes for p in points], dtype=np.float64).mean(axis=2)
    text_rows = [feature_text(p.prompt, a, mode) for p in points for a in p.actions]
    prediction = np.full((len(points), 5), np.nan)
    baseline = np.full((len(points), 5), np.nan)
    splits = GroupKFold(n_splits=folds)
    diagnostics = []
    for k, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(points)), groups=groups)):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError("Game leakage")
        train_rows = [int(i) * 5 + j for i in train_idx for j in range(5)]
        test_rows = [int(i) * 5 + j for i in test_idx for j in range(5)]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                     sublinear_tf=True, dtype=np.float32)
        Xtrain = vectorizer.fit_transform([text_rows[i] for i in train_rows])
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
        model.fit(Xtrain, y[train_idx].ravel())
        Xtest = vectorizer.transform([text_rows[i] for i in test_rows])
        prediction[test_idx] = np.clip(model.predict(Xtest).reshape(-1, 5), 0.0, 1.0)
        baseline[test_idx] = y[train_idx].mean()
        diagnostics.append(dict(fold=k, train_games=len(set(groups[train_idx])),
                                heldout_games=len(set(groups[test_idx])),
                                train_points=len(train_idx), heldout_points=len(test_idx),
                                vocab_size=len(vectorizer.vocabulary_)))
    if not np.all(np.isfinite(prediction)):
        raise AssertionError("Incomplete out-of-fold predictions")
    return y, prediction, baseline, diagnostics


def tied_bins(values, requested=10):
    """Never break identical measured gaps into arbitrary 'deciles'."""
    unique, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) <= requested:
        return inv, len(unique)
    midranks = (np.cumsum(counts) - counts / 2) / counts.sum()
    indices = np.minimum((midranks * requested).astype(int), requested - 1)
    _, compact = np.unique(indices, return_inverse=True)
    return compact[inv], len(set(compact))


def by_bin(rows, key, requested=10):
    if not rows:
        return []
    labels, _ = tied_bins(np.array([row[key] for row in rows]), requested)
    result = []
    for label in sorted(set(labels)):
        subset = [r for r, b in zip(rows, labels) if b == label]
        result.append(dict(bin=int(label + 1), n=len(subset), gap_min=float(min(r[key] for r in subset)),
                           gap_max=float(max(r[key] for r in subset)),
                           gap_mean=float(np.mean([r[key] for r in subset])),
                           projected_diff_mean=float(np.mean([r["projected_diff"] for r in subset])),
                           signed_ratio_mean=float(np.mean([r["signed_ratio"] for r in subset])),
                           clipped_ratio_mean=float(np.mean([r["clipped_ratio"] for r in subset])),
                           **({"heldout_diff_mean": float(np.mean([r["heldout_diff"] for r in subset]))}
                              if "heldout_diff" in subset[0] else {})))
    return result


def _rows(points, q, predictions):
    primary, holdout = [], []
    for i, p in enumerate(points):
        q_i, phat = q[i], predictions[i]
        best, worst = int(np.argmax(q_i)), int(np.argmin(q_i))
        gap = float(q_i[best] - q_i[worst])
        if gap > 0:
            pred_diff = float(phat[best] - phat[worst])
            primary.append(dict(point_id=p.point_id, game=p.group, gap=gap,
                                projected_diff=pred_diff,
                                signed_ratio=pred_diff / gap,
                                clipped_ratio=float(np.clip(pred_diff / gap, 0, 1)),
                                correct_sign=int(pred_diff > 0)))
        outcomes = np.asarray(p.outcomes, dtype=float)
        reps = outcomes.shape[1]
        for r in range(reps):
            sel = (outcomes.sum(axis=1) - outcomes[:, r]) / (reps - 1)
            b, w = int(np.argmax(sel)), int(np.argmin(sel))
            g = float(sel[b] - sel[w])
            if g <= 0:
                continue
            predicted = float(phat[b] - phat[w])
            holdout.append(dict(point_id=p.point_id, game=p.group, heldout_rep=r,
                                gap=g, projected_diff=predicted,
                                heldout_diff=float(outcomes[b, r] - outcomes[w, r]),
                                signed_ratio=predicted / g,
                                clipped_ratio=float(np.clip(predicted / g, 0, 1)),
                                correct_sign=int(predicted > 0)))
    return primary, holdout


def grouped_spearman(rows, bootstraps=500, seed=20261112):
    """Game-cluster CI; descriptive only, ratio has gap in denominator."""
    if len(rows) < 3 or len({r["gap"] for r in rows}) < 2:
        return dict(rho=None, ci95=None, reason="Insufficient nonzero gap levels")
    x = np.asarray([r["gap"] for r in rows])
    y = np.asarray([r["clipped_ratio"] for r in rows])
    if len(set(y)) < 2:
        return dict(rho=None, ci95=None, reason="Constant predictable ratio")
    observed = float(spearmanr(x, y).statistic)
    if bootstraps <= 0:
        return dict(rho=observed, ci95=None)
    games = np.asarray([r["game"] for r in rows])
    unique = np.unique(games)
    idx_map = [np.flatnonzero(games == g) for g in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstraps):
        sample_games = rng.integers(len(unique), size=len(unique))
        ids = np.concatenate([idx_map[j] for j in sample_games])
        if len(np.unique(x[ids])) > 1 and len(np.unique(y[ids])) > 1:
            val = float(spearmanr(x[ids], y[ids]).statistic)
            if np.isfinite(val):
                values.append(val)
    return dict(rho=observed, ci95=[float(z) for z in np.percentile(values, [2.5, 97.5])] if values else None,
                n_games=len(unique))


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root = args.repo_root.resolve()
    if args.source == "e4":
        points, sources = load_e4(root, args.reps, args.archive, not args.skip_hash_check)
    else:
        points, sources = load_m0(root, args.reps)
    y, phat, baseline, diag = crossfit(points, folds=args.folds, alpha=args.alpha)
    _, action_hat, action_baseline, _ = crossfit(points, folds=args.folds, alpha=args.alpha, mode="action_only")
    primary, holdout = _rows(points, y, phat)
    n = len(points)
    mse = lambda z: float(np.mean((y - z) ** 2))
    denom = mse(baseline)
    result = dict(source=args.source, n_points=n, n_games=len({p.group for p in points}),
                  n_arms=5, n_replays_per_arm=args.reps,
                  zero_gap_count=n - len(primary), zero_gap_share=(n - len(primary)) / n,
                  features="student prompt + candidate action only; no hidden state or peeks",
                  oof=dict(mse=mse(phat), baseline_mse=denom,
                           r2_relative_to_game_heldout_mean=(1 - mse(phat) / denom) if denom > 0 else None,
                           action_only_mse=mse(action_hat),
                           action_only_baseline_mse=mse(action_baseline)),
                  gap_bins=by_bin(primary, "gap", args.bins),
                  noise_holdout_bins=by_bin(holdout, "gap", args.bins),
                  rank_association=grouped_spearman(primary, args.bootstrap),
                  folds=diag, archive_sources=sources)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "points.csv", primary)
    _write_csv(args.out / "replicate_holdout.csv", holdout)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# E4/M0 counterfactual credit predictability audit", "",
             f"Source: {args.source}; {n} points, {result['n_games']} game groups; {args.reps} replays/arm.",
             f"Nonzero observed gaps: {len(primary)}; zero gap: {result['zero_gap_share']:.1%}.",
             f"Out-of-fold MSE: {mse(phat):.5f}; fold-mean baseline: {denom:.5f}; "
             f"action-only MSE: {mse(action_hat):.5f}.", "",
             "## Full-data gap bins (ties preserved)", "",
             "| Bin | N | Observed gap | Predicted difference | Positive clipped ratio |",
             "|---:|---:|---:|---:|---:|"]
    for row in result["gap_bins"]:
        lines.append(f"| {row['bin']} | {row['n']} | {row['gap_mean']:.3f} | "
                     f"{row['projected_diff_mean']:.3f} | {row['clipped_ratio_mean']:.3f} |")
    lines += ["", "## Independent-replicate check (select on n−1, evaluate on held-out 1)", "",
              "| Bin | N | Selection gap | Held-out realized gap | Predicted difference |",
              "|---:|---:|---:|---:|---:|"]
    for row in result["noise_holdout_bins"]:
        lines.append(f"| {row['bin']} | {row['n']} | {row['gap_mean']:.3f} | "
                     f"{row['heldout_diff_mean']:.3f} | {row['projected_diff_mean']:.3f} |")
    lines += ["", "## Interpretation guardrails", "",
              "- A clipped predicted/observed gap ratio is **not** a proven realizability fraction. Its denominator "
              "also induces mechanical downward trends as observed gaps grow.",
              "- E4 uses only three binary continuations per arm, so ties are common and ten distinct deciles "
              "may be impossible. Never split equal gap values to manufacture deciles.",
              "- Top-minus-bottom on the same rollouts is biased upward (winner's curse). Use the independent "
              "replicate check before attributing a trend to hidden-state luck.",
              "- Out-of-fold model failure may reflect limited model capacity, sparse successes or noisy returns; "
              "it is not proof that student-visible information is insufficient.",
              "- This audit does not establish EVPI, causal training benefit, or the claimed negative trend. "
              "No validation trajectories are used as training data.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:7]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["e4", "m0"], default="e4")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("exploratory_credit_projection/results_local"))
    args = parser.parse_args()
    if args.reps < 3 or args.folds < 2 or args.bins < 2 or args.alpha <= 0 or args.bootstrap < 0:
        parser.error("Require reps>=3, folds>=2, bins>=2, alpha>0, bootstrap>=0")
    run(args)


if __name__ == "__main__":
    main()
