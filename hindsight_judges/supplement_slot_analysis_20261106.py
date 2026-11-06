"""Read-only post-run supplement for token-slot and MATH slot experiments.

Does not modify frozen scorer/analysis/test/protocol files, and does not rescore.
Requires completed original verification. Outputs ONLY to a separate directory.
See AMENDMENT_TOKEN_MATH_20261106.md. Run in the original server environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import exp2_common as C
import token_slot_analysis as A

HERE = Path(__file__).resolve().parent
TOKEN_ROOT = HERE / "results/token_slot_20261106"
MATH_ROOT = HERE / "results/math_slot_20261106"
OUTPUT = HERE / "results/next_slot_supplement_20261106"
T1 = "T1_b1"
T2 = "T2_b1_clip5_full"
SEED = 20261106
PHASES = ("all", "pre_k", "at_or_after_k", "no_k")
RESTART = re.compile(
    r"^\s*(?:<think>\s*)?(?:let['’]?s\b|let us\b|first(?:ly)?\b|"
    r"we need\b|to solve\b|given\b|step\s*(?:1|one)\b|"
    r"we (?:start|begin)\b|i (?:will|shall)\b)",
    re.IGNORECASE,
)


def require_verified(directory: Path):
    run = directory / "run.json"
    if run.exists():
        assert C.read_json(run)["complete"], f"Incomplete original run: {run}"
    vfile = directory / "verification.json"
    assert vfile.exists() and C.read_json(vfile)["passed"], (
        f"Wait for original verified run: {vfile}"
    )
    v = C.read_json(vfile)
    expected = v.get("analysis_sha256", v.get("analysis_sha"))
    assert expected == C.sha_file(directory / "analysis.json"), "Analysis changed after verification"
    if "manifest_sha256" in v:
        assert v["manifest_sha256"] == C.sha_file(directory / "manifest.json")
    for key, filename in (("annotations_sha", "cut_annotations.json"), ("review_sha", "outcome_review.json")):
        if key in v:
            assert v[key] == C.sha_file(directory / filename)
    return C.sha_file(vfile)


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def softmax(values):
    x = np.asarray(values, float)
    assert x.shape == (5,) and np.isfinite(x).all()
    p = np.exp(x - x.max())
    return p / p.sum()


def ordinary(p):
    reasons = set(p["reasons"])
    return (
        p["kind"] == "think"
        and bool(reasons & {"ordinary_pre", "ordinary_post", "ordinary_no_k"})
        and not bool(reasons & {
            "literal_action_mention", "action_first", "action_discriminating"
        })
    )


def bounds(arms):
    seqs = [a["ids"] for a in arms]
    assert len(seqs) == 5 and all(
        len(a["tokens"]) == len(a["ids"]) for a in arms
    )
    start = 0
    while all(len(s) > start for s in seqs) and len({
        s[start] for s in seqs
    }) == 1:
        start += 1
    end = 0
    while all(len(s) > start + end for s in seqs) and len({
        s[len(s) - 1 - end] for s in seqs
    }) == 1:
        end += 1
    return start, end


def selected_tokens(probe, trim):
    start, end = bounds(probe["arms"]) if trim else (0, 0)
    rows = [a["tokens"][start:len(a["ids"])-end if end else len(a["ids"])]
            for a in probe["arms"]]
    # Empty discrimination is not assigned a made-up neutral score.
    return None if trim and any(not row for row in rows) else rows


def probe_value(probe, q, target, trim=False):
    rows = selected_tokens(probe, trim)
    if rows is None:
        return None
    scores = [sum(t["target_logp"][target] for t in toks) for toks in rows]
    return float(100 * softmax(scores) @ np.asarray(q, float))


def external_value_delta(probe, q, trim=False):
    rows = selected_tokens(probe, trim)
    if rows is None:
        return None
    values = [100 * softmax([sum(t["raw_logp"][c] for t in ts) for ts in rows]) @ np.asarray(q)
              for c in ("EF", "EM")]
    return float(values[0] - values[1])


def probe_slope(probe, q, trim=False):
    rows = selected_tokens(probe, trim)
    if rows is None:
        return None
    delta = [sum(t["raw_logp"]["EF"] - t["raw_logp"]["EM"] for t in toks) for toks in rows]
    x = np.array(q, dtype=float, copy=True)
    x -= x.mean()
    y = np.asarray(delta, float)
    y -= y.mean()
    den = x @ x
    return float(x @ y / den) if den > 0 else None


def paired(values, clusters):
    return A.ci(values, clusters)


def analyze_token(args):
    root = Path(args.token_root).resolve()
    original = root / args.model / args.cohort
    original_verification = require_verified(original)
    ns = SimpleNamespace(out=str(root), name=args.model,
                         limit=32 if args.cohort == "first32" else 0)
    _, records, directory, _, run, _ = A.load(ns)
    original_analysis = C.read_json(directory / "analysis.json")
    assert original_analysis["points"] == len(records)
    clusters = [r["cluster"] for r in records]

    fields = ("kl_target_AF", "kl_AF_target")
    regular = {}
    pointwise = {}
    for r in records:
        positions = [p for p in r["native"] if ordinary(p)]
        pointwise[r["id"]] = positions
    for kind in ("all", "pre_k", "at_or_after_k", "no_k"):
        chosen = [
            [p for p in pointwise[r["id"]]
             if kind == "all" or p["phase"] == kind]
            for r in records
        ]
        section = {
            "points_with_positions": sum(bool(x) for x in chosen),
            "position_count": sum(map(len, chosen)),
            "trajectory_count": len({c for c, ps in zip(clusters, chosen) if ps}),
            "support_status": "available_counts_reported" if any(chosen) else "inconclusive_insufficient_support",
        }
        for field in fields:
            t1_values = [
                mean_or_none([p["targets"][T1][field] for p in rows])
                for rows in chosen
            ]
            differences = [
                mean_or_none([
                    p["targets"][T2][field] - p["targets"][T1][field]
                    for p in rows
                ])
                for rows in chosen
            ]
            noninferior = [
                mean_or_none([
                    p["targets"][T2][field] - 1.10 * p["targets"][T1][field]
                    for p in rows
                ])
                for rows in chosen
            ]
            stat = paired(noninferior, clusters)
            section[field] = {
                "T1_baseline": paired(t1_values, clusters),
                "T2_minus_T1": paired(differences, clusters),
                "noninferiority_margin": "10% of paired T1 KL, pointwise",
                "T2_minus_1p1_T1": stat,
                "passes": bool(stat["ci95"] is not None and
                               stat["ci95"][1] <= 0),
            }
        difflogp = [
            mean_or_none([
                p["targets"][T2]["logp"] - p["targets"][T1]["logp"]
                for p in rows
            ])
            for rows in chosen
        ]
        lpstat = paired(difflogp, clusters)
        section["T2_minus_T1_student_next_logp"] = lpstat
        section["logp_guard_passes"] = bool(
            lpstat["ci95"] is not None and lpstat["ci95"][0] >= -0.1
        )
        regular[kind] = section

    probes = {}
    for phase in PHASES:
        section = {}
        for trim in (False, True):
            label = "discriminating" if trim else "full_sequence"
            rows = [
                [p for p in r["probes"]
                 if (phase == "all" or p["phase"] == phase)
                 and (not trim or selected_tokens(p, True) is not None)]
                for r in records
            ]
            contrasts = []
            for r, ps in zip(records, rows):
                contrasts.append(mean_or_none([
                    probe_value(p, r["q"], T2, trim)
                    - probe_value(p, r["q"], T1, trim)
                    for p in ps
                ]))
            section[label] = {
                "T2_minus_T1_value_pp": paired(contrasts, clusters),
                "T3_value_pp": paired([
                    mean_or_none([probe_value(p, r["q"], "T3", trim)
                                  for p in ps])
                    for r, ps in zip(records, rows)
                ], clusters),
                "external_F_minus_M_Q_slope": paired([
                    mean_or_none([
                        v for p in ps
                        if (v := probe_slope(p, r["q"], trim)) is not None
                    ])
                    for r, ps in zip(records, rows)
                ], clusters),
                "external_F_minus_M_value_pp": paired([
                    mean_or_none([external_value_delta(p, r["q"], trim) for p in ps])
                    for r, ps in zip(records, rows)
                ], clusters),
            }
        span_stats = [
            (bounds(p["arms"]), [len(a["ids"]) for a in p["arms"]])
            for r in records for p in r["probes"]
            if phase == "all" or p["phase"] == phase
        ]
        section["diagnostic_span"] = {
            "n_probes": len(span_stats),
            "average_LCP_tokens": mean_or_none([x[0][0] for x in span_stats]),
            "average_LCS_tokens": mean_or_none([x[0][1] for x in span_stats]),
            "empty_span_arms": sum(n == b[0] + b[1] for b, lengths in span_stats for n in lengths),
            "empty_span_arm_fraction": mean_or_none([float(n == b[0] + b[1]) for b, lengths in span_stats for n in lengths]),
            "probes_unavailable_due_to_empty_span": sum(any(n == b[0] + b[1] for n in lengths) for b, lengths in span_stats),
            "note": "Sensitivity only: external slot still restarts at every candidate token.",
        }
        section["discriminating_minus_full"] = {}
        for metric in ("T2_minus_T1_value_pp", "T3_value_pp", "external_F_minus_M_Q_slope", "external_F_minus_M_value_pp"):
            full = section["full_sequence"][metric]["estimate"]
            trimmed = section["discriminating"][metric]["estimate"]
            section["discriminating_minus_full"][metric] = {
                "estimate_difference": trimmed-full if full is not None and trimmed is not None else None,
                "sign_changed": bool(full*trimmed < 0) if full is not None and trimmed is not None else None,
                "note": "Descriptive difference; paired estimates retain their individual cluster CIs. Check unavailable-span counts.",
            }
        probes[phase] = section

    gain = original_analysis["paired"]["pre_k_T2_minus_T1_b1_value_pp"]
    enough = gain["n"] >= 30 and gain["n_trajectories"] >= 10
    guards = regular["all"]
    new_gate = bool(
        enough and gain["estimate"] >= 3 and
        gain["ci95"] is not None and gain["ci95"][0] > 0 and
        all(guards[f]["passes"] for f in fields) and
        guards["logp_guard_passes"]
    )
    result = {
        "status": "supplement only; no original gate overwritten",
        "model": args.model, "cohort": args.cohort,
        "pilot_only": args.cohort == "first32",
        "points": len(records),
        "legacy_gate": original_analysis["decision"],
        "supplement_gate": {
            "eligible_to_consider_training": new_gate if args.cohort == "full" and guards["position_count"] else None,
            "support_status": guards["support_status"],
            "pre_k_probe_gain_from_original": gain,
            "ordinary_position_guards": guards,
            "rule": "Pre-k +3pp and CI>0, 30 points/10 clusters; ordinary Think both KL upper(T2-1.1*T1)<=0; ordinary logp lower(T2-T1)>=-0.1 nat/token",
        },
        "ordinary_native": regular,
        "probe_sensitivity": probes,
        "position_and_span_audit": [{
            "id": r["id"], "cluster": r["cluster"],
            "native_positions": [{"t": p["t"], "kind": p["kind"], "phase": p["phase"],
                                  "reasons": p["reasons"], "included_as_ordinary": ordinary(p)} for p in r["native"]],
            "probes": [{"char": p["char"], "phase": p["phase"], "lcp": bounds(p["arms"])[0],
                        "lcs": bounds(p["arms"])[1], "arm_lengths": [len(a["ids"]) for a in p["arms"]],
                        "discriminating_available": selected_tokens(p, True) is not None} for p in r["probes"]],
        } for r in records],
        "original_verification_sha256": original_verification,
        "original_analysis_sha256": C.sha_file(directory / "analysis.json"),
        "original_manifest_sha256": C.sha_file(directory / "manifest.json"),
    }
    return result


def clean(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def continuity_features(prefix, continuation, continuation_ids):
    norm_prefix = clean(prefix)
    norm_reply = clean(continuation)
    opening = continuation[:320]
    nums = set(re.findall(r"(?<![\w.])-?\d{2,}(?:\.\d+)?", prefix[-250:]))
    seen = nums & set(re.findall(
        r"(?<![\w.])-?\d{2,}(?:\.\d+)?", continuation[:400]
    ))
    return {
        "restart_opener_heuristic": bool(RESTART.search(continuation)),
        "prefix_repetition_heuristic": bool(
            len(norm_prefix) >= 80 and norm_prefix[:80] in norm_reply[:400]
        ),
        "carryover_hint": bool(seen),
        "carryover_literals": sorted(seen),
        "opening_preview": opening,
        "continuation_chars": len(continuation),
        "continuation_token_count": len(continuation_ids),
    }


def analyze_math(args):
    root = Path(args.math_root).resolve()
    stage = root / args.stage
    verified = require_verified(stage)
    request = C.read_json(stage / "cut_review_request.json")
    annotations = C.read_json(stage / "cut_annotations.json")["questions"]
    aggregate = C.read_json(stage / "aggregate_rows.json")
    original_analysis = C.read_json(stage / "analysis.json")
    assert hashlib.sha256(C.json_bytes(aggregate)).hexdigest() == original_analysis["records_sha"]
    assert set(annotations) == {q["id"] for q in request["questions"]} == {q["id"] for q in aggregate}
    grades = {r["id"]: r for r in aggregate}
    rows = []
    blinded = []
    sample = continuity_sample(request["questions"])
    sample_ids = {r["id"] for r in sample}
    for q in request["questions"]:
        qid = q["id"]
        assert qid in grades and qid in annotations
        source = q["student"]["generation"]["text"]
        prefix = source[:annotations[qid]["prefix_end"]]
        acts = {
            arm: C.read_json(stage / "responses" / qid / (arm + ".json"))
            for arm in ("ACTIVE", "QUOTE-CONTINUE", "QUOTE-REWRITE", "INDEPENDENT")
        }
        for arm in acts:
            path = stage / "responses" / qid / (arm + ".json")
            assert C.sha_file(path) == original_analysis["raw_results_hashes"][str(path)]
        active = acts["ACTIVE"]["generation"]
        quote = acts["QUOTE-CONTINUE"]["generation"]
        features = continuity_features(prefix, quote["text"], quote["ids"])
        features["length_ratio_vs_ACTIVE"] = (
            len(quote["ids"]) / len(active["ids"])
            if active["ids"] else None
        )
        features["length_ratio_unavailable_zero_denominator"] = not bool(active["ids"])
        rows.append({
            "id": qid, "cohort": annotations[qid]["cohort"],
            "features": features,
            "correctness": {
                arm: grades[qid]["grades"][arm]["correct"] for arm in acts
            },
            "retains_student_answer": {
                arm: grades[qid]["grades"][arm]["retains_student_answer"] for arm in acts
            },
            "raw_response_sha256": {
                arm: C.sha_file(stage / "responses" / qid / (arm + ".json"))
                for arm in acts
            },
        })
        if qid in sample_ids:
            swap = int(hashlib.sha256(("blind:" + qid).encode()).hexdigest()[-1], 16) % 2
            left, right = (
                (active["text"], quote["text"]) if swap == 0
                else (quote["text"], active["text"])
            )
            blinded.append({
                "id": qid, "student_prefix_tail": prefix[-380:],
                "continuation_X_opening": left[:450],
                "continuation_Y_opening": right[:450],
                "question_for_reviewer": "For X and Y, label genuine_continuation / restart_or_rewrite / ambiguous; no answer labels shown.",
            })
    strata = {}
    for cohort in ("error_observed", "error_unknown", "all_descriptive"):
        subset = [
            r for r in rows
            if cohort == "all_descriptive" or r["cohort"] == cohort
        ]
        clusters = [r["id"] for r in subset]
        stats = {"n": len(subset), "zero_ACTIVE_token_denominators": sum(
            r["features"]["length_ratio_unavailable_zero_denominator"] for r in subset)}
        for key in ("restart_opener_heuristic", "prefix_repetition_heuristic",
                    "carryover_hint", "length_ratio_vs_ACTIVE"):
            stats[key] = paired([
                r["features"][key] for r in subset
            ], clusters)
        stats["QUOTE_CONTINUE_correctness"] = paired([
            100 * r["correctness"]["QUOTE-CONTINUE"] for r in subset
        ], clusters)
        stats["ACTIVE_correctness"] = paired([
            100 * r["correctness"]["ACTIVE"] for r in subset
        ], clusters)
        strata[cohort] = stats
    review = blind_review_summary(getattr(args, "blind_labels", None), blinded)
    return {
        "status": "automated lexical indicators; blind continuity review " + review["status"],
        "stage": args.stage, "pilot_only": args.stage == "first32",
        "original_verification_sha256": verified,
        "original_analysis_sha256": C.sha_file(stage / "analysis.json"),
        "n": len(rows), "strata": strata,
        "per_question": rows, "blinded_review_sample": blinded,
        "blind_review": review,
        "original_aggregate_sha256": C.sha_file(stage / "aggregate_rows.json"),
        "original_annotations_sha256": C.sha_file(stage / "cut_annotations.json"),
        "original_cut_request_sha256": C.sha_file(stage / "cut_review_request.json"),
        "limitations": [
            "Restart/repetition/numeric carryover are weak heuristics, not verified continuation.",
            "Conditioning on generated behavior is post-treatment; stratified correctness descriptive only.",
        ],
    }


def blind_payload(samples):
    return {"instructions": "Review X/Y without opening correctness or source labels.", "samples": samples}


def continuity_sample(questions):
    return sorted(questions, key=lambda r: hashlib.sha256(
        ("20261106-continuity:" + r["id"]).encode()
    ).hexdigest())[:24]


def blind_review_summary(labels_path, samples):
    allowed = {"genuine_continuation", "restart_or_rewrite", "ambiguous"}
    total = len(samples)
    result = {"sample_size": total, "reviewed_questions": 0, "coverage": 0.0 if total else None,
              "status": "pending", "sample_ids": [s["id"] for s in samples]}
    if not labels_path:
        return result
    doc = C.read_json(Path(labels_path))
    expected = hashlib.sha256(C.json_bytes(blind_payload(samples))).hexdigest()
    assert doc["blind_payload_sha256"] == expected, "Blind review belongs to another sample"
    ids = set(result["sample_ids"])
    seen = set()
    labels = {a: {label: 0 for label in sorted(allowed)} for a in ("ACTIVE", "QUOTE-CONTINUE")}
    for r in doc["samples"]:
        qid = r["id"]
        assert qid in ids and qid not in seen and r["reviewer"] and r["reason"]
        assert r["X"] in allowed and r["Y"] in allowed
        seen.add(qid)
        swap = int(hashlib.sha256(("blind:"+qid).encode()).hexdigest()[-1], 16) % 2
        mapping = ("ACTIVE", "QUOTE-CONTINUE") if swap == 0 else ("QUOTE-CONTINUE", "ACTIVE")
        for arm, label in zip(mapping, (r["X"], r["Y"])):
            labels[arm][label] += 1
    result.update(reviewed_questions=len(seen), coverage=len(seen)/total if total else None,
                  status="complete" if len(seen)==total else "partial", counts_by_arm=labels,
                  labels_sha256=C.sha_file(Path(labels_path)))
    return result


def output_directory(args):
    dest = Path(args.output).resolve()
    for original in (TOKEN_ROOT.resolve(), MATH_ROOT.resolve(), Path(args.token_root).resolve(), Path(args.math_root).resolve()):
        assert dest != original and original not in dest.parents, "Supplement may not write into original score directories"
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=("token", "math"))
    p.add_argument("--model", choices=("7b", "32b"), default="32b")
    p.add_argument("--cohort", choices=("first32", "full"), default="full")
    p.add_argument("--stage", choices=("first32", "full"), default="full")
    p.add_argument("--token-root", default=str(TOKEN_ROOT))
    p.add_argument("--math-root", default=str(MATH_ROOT))
    p.add_argument("--output", default=str(OUTPUT))
    p.add_argument("--blind-labels", help="Optional independently recorded X/Y continuity annotations")
    args = p.parse_args()
    dest = output_directory(args)
    result = analyze_token(args) if args.experiment == "token" else analyze_math(args)
    dest.mkdir(parents=True, exist_ok=True)
    result["supplement_script_sha256"] = C.sha_file(Path(__file__))
    result["amendment_sha256"] = C.sha_file(HERE / "AMENDMENT_TOKEN_MATH_20261106.md")
    name = (
        f"token_{args.model}_{args.cohort}.json"
        if args.experiment == "token" else f"math_{args.stage}.json"
    )
    C.save_json(dest / name, result)
    if args.experiment == "math":
        C.save_json(dest / f"math_{args.stage}_blind_review.json", blind_payload(result["blinded_review_sample"]))
    print("SUPPLEMENT COMPLETE", dest / name, "no original data changed")


if __name__ == "__main__":
    main()
