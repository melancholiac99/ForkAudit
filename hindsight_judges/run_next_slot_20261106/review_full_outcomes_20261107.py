"""Freeze the full-run answer audit after the separately frozen continuity review."""
from pathlib import Path
import sys, time, shutil
from collections import Counter
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
import exp2_common as C
import math_slot_core as M
import supplement_slot_analysis_20261106 as S

# Explicit final-answer decisions made by reading each requested response,
# problem and reference; not copied from the automatic grade or pilot labels.
# Correctness here retains the frozen benchmark-answer definition. The
# mathematically ambiguous item is explicitly flagged below, not silently fixed.
REVIEWS = [
 ('intermediate_algebra:568','INDEPENDENT',False,False,'Final 60 differs from reference 150 and Student 0. If the third root is t, t(10-t)=25 gives t=5 and the constant coefficient is 30t=150.'),
 ('intermediate_algebra:434','INDEPENDENT',True,False,'With z=log_2 x>0 the equation gives log_2 z=(3/2)log_2 3, hence z^2=27. Final 27 differs from the Student logarithmic expression.'),
 ('precalculus:97','INDEPENDENT',False,True,'Final 6 repeats Student 6. Law of cosines with a=(b+c)/2 gives a:b:c=6:7:5, whose minimum integral triangle has area 6sqrt(6), not 6.'),
 ('number_theory:451','QUOTE-CONTINUE',False,False,'The polynomial product plus b^3+1 is b^5+b^2, giving 100100 in base b. Final 101100_b is wrong and also differs from Student 10101_b.'),
 ('intermediate_algebra:571','ACTIVE',True,False,'Substitution yields x^2-10x-39=0, roots 13 and -3 with nonzero denominators. Positive root 13 is correct and differs from Student 367/27.'),
 ('precalculus:429','QUOTE-REWRITE',True,False,'Anticommutativity gives -3(4,7,2)+2(1,-7,18)=(-10,-35,30), the final boxed vector. It differs from Student (-14,-7,-42).'),
 ('intermediate_algebra:445','QUOTE-REWRITE',True,False,'The rational terms are k=0,6,12 and sum to 64+924*72+81=66673. Final 66673 differs from Student 532369.'),
 ('counting_and_probability:408','QUOTE-REWRITE',True,False,'20*19*18-3*2*18=6732 assignments avoid having both Alex and Bob. Final 6732 differs from Student 5922.'),
 ('precalculus:64','ACTIVE',True,False,'Deduplicated solutions are 0,pi/3,pi,5pi/3,2pi, with sum 5pi. Final 5pi is correct and differs from Student 4pi.'),
 ('algebra:117','ACTIVE',False,True,'Original expression factors as (1001001-1001)*(1010101+989899)=2*10^12, hence (2,12). Final (8,0) is wrong and repeats the complete Student pair.'),
 ('number_theory:445','QUOTE-REWRITE',True,False,'Exact 20!=2432902008176640000 has leading digit 2. Final 2 is correct and differs from Student 4, although the response has inconsistent magnitude reasoning. Final-answer grading does not certify that reasoning.'),
 ('intermediate_algebra:881','QUOTE-REWRITE',False,True,'Original graph has a hole at 0, a vertical asymptote at 3/2, no horizontal asymptote and one oblique asymptote, so the weighted count is 7. Final 6 repeats Student 6.'),
 ('number_theory:510','ACTIVE',True,False,'Product is 151200, whose hundreds digit is 2. Final 2 differs from Student 1.'),
 ('intermediate_algebra:888','QUOTE-CONTINUE',True,False,'First modulus fixes real part 6; second gives b^2-25b+136=0, so b=8 or 17. Final solution set is correct and differs from the Student radical pair.'),
 ('intermediate_algebra:415','QUOTE-CONTINUE',False,False,'Correct division is r^2/(r+1)=r-1+1/(r+1). Root sum is -1 and reciprocal sum is 49, yielding -50. Final -99 is wrong and differs from Student -101.'),
 ('precalculus:416','ACTIVE',True,False,'Pairing supplementary angles gives 2(sin15+sin45+sin75)=sqrt(6)+sqrt(2). Final expression equals the reference despite reversed term order, and differs from Student sqrt(2).'),
 ('number_theory:182','ACTIVE',True,False,'Among bases 5 through 9 only 8 is a cube and 9 is a square; 512+729=1241. Final 1241 differs from Student 1925.'),
 ('number_theory:182','QUOTE-CONTINUE',False,False,'Final 1709 incorrectly includes cubes of bases 5 and 7. Required sum is 512+729=1241; 1709 also differs from Student 1925.'),
 ('precalculus:399','INDEPENDENT',True,False,'cos(gamma)=(2/sqrt(3)-1)cos(alpha)cos(beta), so the ratio is 3+2sqrt(3). Final radical is correct and differs from Student 2.'),
 ('number_theory:49','INDEPENDENT',True,False,'200 mod 7 is 4; four days before Wednesday is Saturday. Final Saturday differs from Student Sunday.'),
 ('intermediate_algebra:125','ACTIVE',True,False,'Center (5,7) and squared radius 13 give x^2+y^2-10x-14y+61=0. Final (10,14,61) is correct and differs from Student (10,14,48).'),
 ('intermediate_algebra:125','INDEPENDENT',True,False,'Midpoint of the diagonal is (5,7), squared radius 13, so the final triple (10,14,61) is correct and differs from Student (10,14,48).'),
 ('precalculus:405','QUOTE-CONTINUE',True,False,'Product-to-sum gives cos10*cos50*cos70=sqrt(3)/8; multiplying by cos30 yields 3/16. Final 3/16 differs from Student 1/8.'),
 ('geometry:469','QUOTE-CONTINUE',True,False,'Angle ABC is 110 degrees; opposite angle CDA in the cyclic quadrilateral is 70 degrees. Final 70 degrees matches the requested unit and differs from Student 145.'),
 ('geometry:469','QUOTE-REWRITE',False,False,'Final 35 degrees is neither reference 70 nor Student 145. CDA is opposite ABC in the cyclic quadrilateral, rather than equal to either base angle.'),
 ('intermediate_algebra:253','QUOTE-CONTINUE',False,True,'Frozen benchmark key is 1/4; final -3/4 repeats Student -3/4 and does not match that key. IMPORTANT DATA ISSUE: -3/4 also gives a common point (-1/2,-1/2) with both slopes -1. This is a valid tangential contact under the ordinary differential definition, although another intersection exists. Preserve benchmark scoring for the preregistered primary analysis; flag ambiguous reference/eligibility and report a separate exclusion sensitivity. Do not call this unambiguously mathematically wrong.'),
 ('counting_and_probability:329','QUOTE-REWRITE',True,False,'Independent within-office permutations give 6!*4!*5!*3!=12441600. Final 12441600 differs from Student 18!.'),
 ('geometry:95','INDEPENDENT',True,False,'Opposite parallel sides are two apothems apart, 2*(sqrt(3)/2)*6=6sqrt(3). Final answer is correct and differs from Student 3sqrt(3).'),
 ('prealgebra:753','ACTIVE',True,False,'2400=2^5*3*5^2 has eight prime factors counted with multiplicity. Final 8 differs from Student 7.'),
 ('prealgebra:753','QUOTE-CONTINUE',True,False,'Prime-factor multiplicities are 5+1+2=8. Final 8 differs from Student 7.'),
 ('prealgebra:753','QUOTE-REWRITE',True,False,'Prime factorization 2^5*3*5^2 gives eight factors. Final 8 differs from Student 7.'),
 ('prealgebra:753','INDEPENDENT',True,False,'Five factors of 2, one of 3 and two of 5 total eight. Final 8 differs from Student 7.'),
]

def main():
    p=H/'results/math_slot_20261106/full'
    supplement=H/'results/next_slot_supplement_20261106'
    labels=supplement/'math_full_blind_labels.json'
    assert C.sha_file(labels)=='5b7d3d1b7e0e4b6bd5c564774d0618d83485d0d2338da6e5efe6a2c603fbd112'
    blind=S.blind_review_summary(labels,C.read_json(supplement/'math_full_blind_review.json')['samples'])
    assert blind['status']=='complete' and blind['sample_size']==24
    C.save_json(supplement/'math_full_blind_labels_verification.json',dict(passed=True,**blind,
        payload_sha256=C.sha_file(supplement/'math_full_blind_review.json'),
        order='Blind labels saved and hash-checked before reading outcome request, condition mapping, or correctness in this review session.'))
    reqpath=p/'outcome_review_request.json';req=C.read_json(reqpath)
    qs={q['id']:q for q in C.read_json(p/'cut_review_request.json')['questions']}
    assert not (p/'outcome_review.json').exists()
    assert [(x['id'],x['arm']) for x in req['required']]==[(q,a) for q,a,*_ in REVIEWS]
    assert Counter(a for _,a,*_ in REVIEWS)=={a:8 for a in M.ARMS}
    before={r['path']:r['sha'] for r in req['records']}
    assert len(before)==240 and len(qs)==60
    assert all(C.sha_file(Path(f))==s for f,s in before.items())
    raw=[C.read_json(Path(f)) for f in before]
    assert all(g['grade']['format']=='boxed' and g['grade']['parseable'] for g in raw)
    assert all(g['generation']['stop']=='eos' for g in raw)
    for name in ('analysis.json','aggregate_rows.json','REPORT.txt'):
        dst=p/'pre_review_automatic'/name;dst.parent.mkdir(exist_ok=True)
        assert not dst.exists();shutil.copyfile(p/name,dst)
    items=[];changes=[]
    for qid,arm,correct,retains,reason in REVIEWS:
        record=next(x for x in req['required'] if x['id']==qid and x['arm']==arm)
        g=C.read_json(p/'responses'/qid/(arm+'.json'));q=qs[qid]
        items.append(dict(**record,correct=correct,retains_student_answer=retains,
            reviewer='Codex agent full-run final-answer audit',reason=reason,
            original_automatic_grade=g['grade'],final_answer_quote=M.boxed(g['graded_text']),
            student_answer_quote=M.boxed(q['student']['generation']['text']),
            reference_answer_quote=M.boxed(q['gold_for_offline_review_only']),stop_reason=g['generation']['stop']))
        if correct!=g['grade']['correct'] or retains!=g['grade']['retains_student_answer']:
            changes.append(dict(id=qid,arm=arm,original=g['grade'],reviewed=dict(correct=correct,retains_student_answer=retains)))
    assert not changes, 'Inspect disagreement before publishing the review'
    doc=dict(request_sha=C.sha_file(reqpath),items=items,stage='full',
        reviewed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        required_count=32,reviewed_count=len(items),format_anomalies=0,
        automatic_label_changes=changes,review_script_sha256=C.sha_file(Path(__file__)),
        policy='Frozen benchmark final-answer match and whole-answer Student retention; no threshold, response or truncation-policy changes.',
        limitations='Single-assistant audit of required final answers, not independent human adjudication or all-response proof verification. Reference ambiguity for intermediate_algebra:253 is separately documented; primary retains frozen key.',
        data_issue_ids=['intermediate_algebra:253'],
        blind_continuity_labels_frozen_before_answer_audit=True,blind_labels_sha256=C.sha_file(labels))
    draft=p/'outcome_review.reviewed_draft.json';C.save_json(draft,doc)
    assert all(C.sha_file(Path(f))==s for f,s in before.items())
    C.save_json(p/'outcome_review_verification.json',dict(passed=True,raw_responses_unchanged=240,
        required_covered=32,per_arm=dict(Counter(x['arm'] for x in items)),automatic_label_changes=0,
        format_anomalies=0,truncated_responses=0,review_sha256=C.sha_file(draft),
        request_sha256=doc['request_sha'],raw_response_hashes=before,
        preserved_automatic_analysis_sha256=C.sha_file(p/'pre_review_automatic/analysis.json'),
        flagged_reference_ambiguity=['intermediate_algebra:253']))
    C.save_json(supplement/'math_full_reference_ambiguity.json',dict(
        id='intermediate_algebra:253',frozen_reference='1/4',student_and_reviewed_answer='-3/4',
        checked_equations=['At x=y=-1/2,k=-3/4: y=x^2+k and x=y^2+k both hold.',
            'dy/dx=2x=-1 and implicit dy/dx=1/(2y)=-1: same tangent line x+y=-1.',
            'There is also an intersection (3/2,3/2). Requiring unique intersection would exclude this contact, but that is not explicit in the problem.',
            'Subtracting the two parabola equations gives (y-x)(1+x+y)=0. Common tangent condition is 4xy=1. These admit x=y=1/2,k=1/4 and x=y=-1/2,k=-3/4.'],
        implication='The frozen benchmark key omits a valid tangential contact under the shared-tangent definition. Student-wrong eligibility is consequently ambiguous for this item.',
        handling='No change to original responses, grade, key, cuts, selection or primary thresholds. Report an explicitly post-hoc leave-one-item-out sensitivity alongside primary.',
        audit_note='The previously annotated prefix still contains an unsupported necessity of a unique intersection; this note concerns the final-answer key, not a retrospective change of the cut annotation.',
        raw_response_hashes={f:s for f,s in before.items() if '/intermediate_algebra:253/' in f}))
    draft.replace(p/'outcome_review.json')
    print('PUBLISHED: 32/32 required reviews, 240 raw responses unchanged, 0 grading changes; one flagged reference ambiguity',flush=True)

if __name__=='__main__':main()
