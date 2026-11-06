"""Freeze exhaustive fixed-pool budget using throughput and cut quality only.

No condition accuracy, effect size, confidence interval or significance is read.
An earlier outcome audit exists; its labels do not enter this deterministic rule.
"""
from pathlib import Path
import sys,time,math
from collections import Counter
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import exp2_common as C,exp6_math_slot as E

def main():
    root=H/'results/math_slot_20261106';pilot=root/'first32'
    assert not (root/'full_budget.json').exists(), 'Do not overwrite a frozen budget'
    assert not list((root/'full/responses').glob('*/*.json')), 'Freeze before full outcomes'
    prep,queue=E.prepared()
    v=C.read_json(pilot/'verification.json');assert v['passed']
    # Hash check only: do not inspect effect estimates in the pilot analysis.
    assert v['analysis_sha']==C.sha_file(pilot/'analysis.json')
    assert v['annotations_sha']==C.sha_file(pilot/'cut_annotations.json')
    assert v['review_sha']==C.sha_file(pilot/'outcome_review.json')
    annotations=C.read_json(pilot/'cut_annotations.json')['questions']
    assert len(annotations)==32
    students={q['id']:C.read_json(root/'student'/(q['id']+'.json')) for q in queue}
    eligible=[q['id'] for q in queue if students[q['id']]['eligible']]
    screened=[qid for qid in eligible if (root/'screen'/(qid+'.json')).exists()]
    qualified=[qid for qid in screened if C.read_json(root/'screen'/(qid+'.json'))['qualified']]
    pending=[qid for qid in eligible if qid not in screened]
    assert (len(queue),len(eligible),len(screened),len(qualified),len(pending))==(512,155,71,32,84)
    assert set(qualified)==set(annotations)
    screen_paths=sorted((root/'screen_generations').glob('*/*.json'))
    eval_paths=sorted((pilot/'responses').glob('*/*.json'))
    # Pool timings across the four conditions; ignore grades and condition effects.
    screen_g=[C.read_json(p)['generation'] for p in screen_paths]
    eval_g=[C.read_json(p)['generation'] for p in eval_paths]
    assert len(screen_g)==4*len(screened) and len(eval_g)==128
    screen_mean=sum(g['seconds'] for g in screen_g)/len(screen_g)
    eval_per_question=sum(g['seconds'] for g in eval_g)/32
    truncations=sum(g['stop']=='length' for g in eval_g)
    assert truncations==0
    counts=dict(Counter(a['cohort'] for a in annotations.values()))
    assert counts=={'error_observed':16,'error_unknown':16}
    assert all(a['original_proposed_boundary_retained'] for a in annotations.values())
    # Exact eligibility ceiling after already rejected questions. Reaching it
    # requires every remaining eligible question to qualify, so the runner's
    # existing target-or-exhaustion loop necessarily screens the full fixed pool.
    target=len(qualified)+len(pending)
    assert target==116 and target<=200
    expected=len(qualified)+len(pending)*len(qualified)/len(screened)
    screening_hours=len(pending)*4*screen_mean/3600
    expected_eval_hours=expected*eval_per_question/3600
    ceiling_eval_hours=target*eval_per_question/3600
    frozen_cuts={}
    for qid in eligible:
        s=students[qid];proposal=s['cut_proposal'];assert proposal
        frozen_cuts[qid]=dict(student_sha256=C.sha_file(root/'student'/(qid+'.json')),
            prefix_end=proposal['prefix_end'],prefix_sha256=E.sha(proposal['prefix']),
            cut_proposal_sha256=E.sha(proposal))
    evidence=dict(fixed_candidate_count=512,student_eligible_count=155,
        screened_count=71,qualified_count=32,pending_count=84,
        eligible_ids_in_frozen_order=eligible,pending_ids_in_frozen_order=pending,
        screened_ids_in_frozen_order=screened,qualified_ids_in_frozen_order=qualified,
        frozen_cut_proposals=frozen_cuts,pilot_cut_cohorts=counts,
        pilot_all_original_cuts_retained=True,pilot_eval_responses=128,
        pilot_eval_truncations=truncations,pilot_eval_max_tokens=max(len(g['ids']) for g in eval_g),
        screen_mean_seconds=screen_mean,eval_seconds_per_question=eval_per_question,
        screen_generation_hashes={str(p.relative_to(H)):C.sha_file(p) for p in screen_paths},
        pilot_generation_hashes={str(p.relative_to(H)):C.sha_file(p) for p in eval_paths},
        candidate_sha=prep['candidate_sha'],prepare_sha256=C.sha_file(root/'prepare.json'),
        pilot_cut_annotations_sha256=C.sha_file(pilot/'cut_annotations.json'))
    C.save_json(root/'full_budget_inputs.json',evidence)
    budget=dict(target=target,basis='timing_and_cut_quality',
        reviewer='Codex agent prospective budget review, 2026-11-06',
        pilot_verification_sha=C.sha_file(pilot/'verification.json'),
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        policy='Exhaust all remaining Student-eligible questions in the original fixed 512-question order, then evaluate all Teacher-qualified questions. No additional candidates or reordering.',
        target_is_eligibility_upper_bound=True,protocol_maximum_target=200,
        target_derivation='32 already qualified + 84 unscreened eligible = 116 maximum possible; reaching 116 also requires exhausting all 84.',
        expected_qualified_descriptive_only=expected,
        expected_count_is_not_a_stopping_rule=True,
        shortfall_policy='If the fixed queue is exhausted below 116, use every available qualified question and report requested ceiling, actual cohort size and screening rates. Do not resample, relax criteria or replace questions.',
        hours=dict(remaining_screening_at_measured_rate=screening_hours,
            full_four_arm_generation_expected=expected_eval_hours,
            expected_total_gpu=screening_hours+expected_eval_hours,
            full_four_arm_generation_if_all_remaining_qualify=ceiling_eval_hours,
            total_gpu_at_maximum_cohort_and_measured_rate=screening_hours+ceiling_eval_hours,
            planning_elapsed_hours_range=[8,11],
            caveat='Extrapolations, not hard runtime caps; longer solutions, qualification variability, model loading, CPU analysis and agent review may change elapsed time.'),
        cut_policy=dict(reuse_pilot_annotations_exactly=True,
            pilot_annotations_sha256=C.sha_file(pilot/'cut_annotations.json'),
            retain_original_proposed_boundaries_for_new_questions=True,
            review_new_prefixes_before_full_four_arm_generation=True,
            keep_error_observed_and_error_unknown=True,
            no_enrichment_or_exclusion_based_on_error_cohort=True),
        reuse_policy='Reuse frozen Student responses, all screening generations and pilot cut annotations. Regenerate all four evaluation conditions with the already declared eval_full seed namespace, including pilot-overlap questions.',
        unchanged='Dataset revision/order, Student/Teacher weights, SDPA MATH, precision, chunk1024, sampling configuration, 3-of-4 screening threshold, TV<0.005, four arms, bootstrap and statistical thresholds.',
        outcome_independence='Previous outcome audit performed, but no condition accuracy, effect, confidence interval or significance enters this budget. Sample rule is exhaustive inclusion based on eligibility only; pooled timing and prospective cut quality justify feasibility.',
        inputs_sha256=C.sha_file(root/'full_budget_inputs.json'),
        source_sha256=C.sha_file(Path(__file__)),math_source=E.fingerprint(),
        no_full_outcomes_at_freeze=True,no_shutdown=True)
    draft=root/'full_budget.validated_draft.json';C.save_json(draft,budget)
    assert budget['basis']=='timing_and_cut_quality' and 1<=budget['target']<=200
    assert E.fingerprint()==prep['source']
    assert not list((root/'full/responses').glob('*/*.json'))
    C.save_json(root/'full_budget_verification.json',dict(passed=True,
        budget_sha256=C.sha_file(draft),inputs_sha256=budget['inputs_sha256'],
        pilot_verified=True,target_within_protocol_cap=True,
        target_equals_known_eligibility_ceiling=True,entire_fixed_pool_will_be_screened=True,
        sample_size_independent_of_condition_effects=True,
        original_math_source_unchanged=True,no_full_outcomes_at_freeze=True))
    draft.replace(root/'full_budget.json')
    print('FROZEN target ceiling',target,'expected cohort',round(expected,1),
        'expected remaining GPU hours',round(screening_hours+expected_eval_hours,2),flush=True)

if __name__=='__main__':main()
