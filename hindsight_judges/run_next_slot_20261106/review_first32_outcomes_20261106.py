"""Record individually reviewed final answers without changing raw responses/grades."""
from pathlib import Path
import sys,time,shutil
from collections import Counter
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import exp2_common as C,exp6_math_slot as E,math_slot_core as M

# Explicit decisions from reading the problem, reference answer, Student answer,
# and sampled Teacher answer/end. These are not inferred from the automatic grade.
REVIEWS=[
 ('number_theory:51','QUOTE-CONTINUE',True,False,'12!=479001600 gives a=9. The final boxed 9 is correct and differs from Student 7.'),
 ('intermediate_algebra:607','QUOTE-REWRITE',True,False,'Final boxed 32 is correct and differs from Student 1895064. Independently, writing the five GP roots symmetrically around m gives (sum roots)/(sum reciprocals)=m^2=4, so the absolute product is |m|^5=32. The response still drops r^4 in the reciprocal sum and unjustifiably removes r^10; correct final answer is not a valid-proof certification.'),
 ('precalculus:264','QUOTE-CONTINUE',True,False,'Substitution of (1+t,3t,1-t) into x+y+cz=d gives 1+c+(4-c)t=d, hence (c,d)=(4,5). The final boxed pair matches and differs from Student (-3,-3).'),
 ('intermediate_algebra:439','INDEPENDENT',False,False,'The final 1/2 is neither reference 1/4 nor Student 1. For t=x+1/x, real roots require t<=-2 or t>=2. With a,b in the unit square the feasible region is 2a+b>=2, a triangle of area 1/4. The response confuses a boundary segment with the region and even states zero area before boxing 1/2; the final answer is still unambiguously wrong.'),
 ('number_theory:183','ACTIVE',False,False,'Only 16 and 36 are two-digit squares ending in 6, so the count is 2. The final 3 wrongly includes 64 and is distinct from Student 6.'),
 ('number_theory:183','QUOTE-CONTINUE',False,False,'The final 3 wrongly counts 64 as ending in 6; the correct count is 2 and the Student final answer is 6, so neither correctness nor retention holds.'),
 ('intermediate_algebra:568','ACTIVE',False,False,'If the third root is z, Vieta gives 30+z(10-z)=55, hence z=5 and c=30z=150. Final boxed 30 is wrong and differs from Student 0. Earlier c=0 mentions are intermediate reasoning, not the final retained answer.'),
 ('intermediate_algebra:568','INDEPENDENT',True,False,'The response derives the third root 5 and coefficient c=30*5=150. Final 150 matches the reference and differs from Student 0.'),
 ('intermediate_algebra:434','ACTIVE',True,False,'With y=log_2 x>0, log_2(y/3)=log_2(y)/3 gives y=3^(3/2), so y^2=27. The final 27 is correct and is not the Student value 9(log_2 3)^2/4.'),
 ('intermediate_algebra:434','QUOTE-CONTINUE',True,False,'The transformed logarithm equation yields y^2=27. Final boxed 27 matches the reference and differs from the Student logarithmic expression.'),
 ('intermediate_algebra:434','INDEPENDENT',True,False,'The final boxed 27 follows from log_2 x=3^(3/2), and differs from the Student 9(log_2 3)^2/4.'),
 ('precalculus:97','INDEPENDENT',True,False,'Integer sides have ratios 6:7:5; the minimal sides 6,7,5 give Heron area sqrt(9*3*2*4)=6sqrt(6). The boxed radical is correct and not Student 6.'),
 ('number_theory:451','QUOTE-CONTINUE',False,False,'Correct base polynomial is (b^4+b^3+b+1)(b-1)+(b^3+1)=b^5+b^2, giving digits 100100. Final 101100_b contains an extra b^3 term and differs from Student 10101_b; the base subscript does not make these digit strings equivalent.'),
 ('intermediate_algebra:571','ACTIVE',True,False,'Using y=x^2-10x-69, the rational numerator is -64y-1920, giving y=-30 and x=13 or -3. Denominators are nonzero. Positive answer 13 is correct and differs from Student 367/27.'),
 ('intermediate_algebra:571','INDEPENDENT',True,False,'The quadratic x^2-10x-39=0 has roots 13 and -3. Final 13 is the required positive root and is not Student 367/27.'),
 ('prealgebra:855','INDEPENDENT',True,False,'Largest and smallest fractions are 11/8 and 3/8, whose difference is 1. Final 1 is correct and differs from Student 23/24.'),
 ('intermediate_algebra:573','ACTIVE',False,False,'For this recurrence the first ten terms sum to 11*f7=913. Even the displayed sequence 15,1,16,17,33,50,83,133,216,349 sums to 913, not the final 903. Student final 961 is also distinct.'),
 ('precalculus:429','QUOTE-REWRITE',True,False,'Anticommutativity gives -3(4,7,2)+2(1,-7,18)=(-10,-35,30). Final boxed column vector is correct and differs componentwise from Student (-14,-7,-42).'),
 ('precalculus:429','INDEPENDENT',True,False,'Final column vector (-10,-35,30) equals -3(a cross c)+2(b cross c), and is not Student (-14,-7,-42).'),
 ('intermediate_algebra:445','QUOTE-REWRITE',True,False,'Only k=0,6,12 give rational terms. Their sum is 64+924*8*9+81=66673. Final 66673 is correct and differs from Student 532369.'),
 ('counting_and_probability:408','QUOTE-REWRITE',True,False,'Total ordered assignments are 20*19*18=6840; those containing both Alex and Bob number 3*2*18=108, leaving 6732. Final 6732 is correct and differs from Student 5922.'),
 ('precalculus:64','ACTIVE',True,False,'Distinct solutions in [0,2pi] are 0,pi/3,pi,5pi/3,2pi, with sum 5pi. Final 5pi is correct and differs from Student 4pi; the duplicated cos(x)=1 branch is not counted twice.'),
 ('precalculus:64','QUOTE-REWRITE',True,False,'Combining the sin(x)=0 branch with cos(x)=1/2 and deduplicating endpoints gives sum 5pi. Final 5pi matches the reference and differs from Student 4pi.'),
 ('algebra:117','ACTIVE',False,False,'The original expression factors as (1001001-1001)*(1010101+989899)=1000000*2000000=2*10^12, so (a,b)=(2,12). Final (8,6) is wrong and is not the complete Student pair (8,0); sharing just its first component is not answer retention.'),
 ('algebra:117','QUOTE-CONTINUE',False,False,'The original expression is 2*10^12, so the required pair is (2,12). Final (8,2) differs from both that pair and Student (8,0). Retention requires equality of the whole ordered pair.'),
 ('number_theory:445','QUOTE-REWRITE',True,False,'Exact 20!=2432902008176640000 has leading digit 2. Final 2 is correct and differs from Student 4. The response uses an inaccurate logarithmic approximation, so this final-answer match does not certify all calculations.'),
 ('intermediate_algebra:569','QUOTE-CONTINUE',True,False,'Final boxed 401 matches the reference and differs from Student 192. Independently f composed with itself is injective, hence f is injective; f(f(n))=f(n) forces f(n)=n, so 5n=2005. The response contains false rationalization/integrality statements; retain the final-answer grade without claiming valid reasoning.'),
 ('intermediate_algebra:569','INDEPENDENT',True,False,'Final 401 is correct and differs from Student 192. Injectivity of f follows from f(f(x))=6x-2005; applying it to f(f(n))=f(n) supplies the missing justification that f(n)=n and n=401.'),
 ('intermediate_algebra:881','QUOTE-CONTINUE',False,True,'There is one hole at x=0, one vertical asymptote, no horizontal asymptote and one oblique asymptote, so a+2b+3c+4d=1+2+0+4=7. Final 6 is wrong and exactly equals Student 6.'),
 ('intermediate_algebra:881','QUOTE-REWRITE',False,True,'Cancellation preserves a hole in the original graph at x=0, so the correct weighted count is 7. Final 6 repeats the Student answer exactly; retention is true while correctness is false.'),
 ('algebra:1149','ACTIVE',True,False,'The ratio condition gives (a-b)(a+b-4)=0. Since a differs from b, their sum is 4. Final 4 is correct and is not Student -4.'),
 ('algebra:1149','QUOTE-REWRITE',True,False,'Cross multiplication and a!=b give a+b=4. Final boxed 4 is correct and differs from Student -4.'),
]

def main():
    p=H/'results/math_slot_20261106/first32';reqpath=p/'outcome_review_request.json'
    req=C.read_json(reqpath);cuts=C.read_json(p/'cut_review_request.json')
    qs={q['id']:q for q in cuts['questions']}
    assert not (p/'outcome_review.json').exists(), 'Do not overwrite a completed review'
    E.prepared()
    before={r['path']:r['sha'] for r in req['records']}
    assert len(before)==128
    assert all(C.sha_file(Path(f))==sha for f,sha in before.items())
    assert [(x['id'],x['arm']) for x in req['required']]==[(q,a) for q,a,*_ in REVIEWS]
    assert Counter(a for _,a,*_ in REVIEWS)=={a:8 for a in M.ARMS}
    for name in ('analysis.json','aggregate_rows.json','REPORT.txt'):
        target=p/'pre_review_automatic'/name;target.parent.mkdir(exist_ok=True)
        assert not target.exists();shutil.copyfile(p/name,target)
    raw=[C.read_json(Path(f)) for f in before]
    anomalies=[r for r in raw if r['grade']['format']!='boxed' or not r['grade']['parseable']]
    assert not anomalies, 'This signed review covers the frozen request with no extra format anomalies'
    items=[];changes=[]
    for qid,arm,correct,retains,reason in REVIEWS:
        record=next(x for x in req['required'] if x['id']==qid and x['arm']==arm)
        g=C.read_json(p/'responses'/qid/(arm+'.json'));q=qs[qid]
        assert g['generation']['stop']=='eos', 'Do not override truncation policy'
        entry=dict(**record,correct=correct,retains_student_answer=retains,
            reviewer='Codex agent final-answer audit, 2026-11-06',reason=reason,
            original_automatic_grade=g['grade'],final_answer_quote=M.boxed(g['graded_text']),
            student_answer_quote=M.boxed(q['student']['generation']['text']),
            reference_answer_quote=M.boxed(q['gold_for_offline_review_only']),
            stop_reason=g['generation']['stop'])
        if correct!=g['grade']['correct'] or retains!=g['grade']['retains_student_answer']:
            changes.append(dict(id=qid,arm=arm))
        items.append(entry)
    assert not changes, 'Unexpected disagreement requires inspection before publishing'
    doc=dict(request_sha=C.sha_file(reqpath),items=items,stage='first32',
        reviewed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        required_count=len(req['required']),reviewed_count=len(items),format_anomalies=0,
        automatic_label_changes=changes,review_script_sha256=C.sha_file(Path(__file__)),
        policy='Grade the final answer, not the validity of every reasoning step. Compare the entire answer for Student-answer retention. Preserve the truncation and parsing policies.',
        limitations='Single-agent answer audit of all required samples, not independent human adjudication or full proof verification. Pilot outcomes are descriptive only.',
        blind_continuity_labels_frozen_before_answer_audit=True)
    draft=p/'outcome_review.reviewed_draft.json';C.save_json(draft,doc)
    assert all(C.sha_file(Path(f))==sha for f,sha in before.items())
    C.save_json(p/'outcome_review_verification.json',dict(passed=True,raw_responses_unchanged=128,
        required_covered=len(items),per_arm=dict(Counter(x['arm'] for x in items)),
        automatic_label_changes=0,format_anomalies=0,review_sha256=C.sha_file(draft),
        request_sha256=doc['request_sha'],raw_response_hashes=before,
        preserved_automatic_analysis_sha256=C.sha_file(p/'pre_review_automatic/analysis.json')))
    draft.replace(p/'outcome_review.json')
    print('PUBLISHED: 32/32 required reviews; 0 label changes; 128 raw responses unchanged',flush=True)

if __name__=='__main__':main()
