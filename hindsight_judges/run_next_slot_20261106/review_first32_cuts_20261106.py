"""Prospective agent review of frozen Student prefixes; no evaluation outcomes read.

All proposed boundaries are retained. Reasons use the problem and prefix only,
not the Student suffix, final grade, gold solution, or Teacher screening answers.
This file records reviewer decisions, not an automated error classifier.
"""
from pathlib import Path
import sys,time,hashlib
from collections import Counter
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import exp2_common as C
import exp6_math_slot as E
import math_slot_core as M

# index, ID, exact erroneous quotation (None means error_unknown), justification
REVIEWS=[
 (0,'number_theory:51',None,
  'The displayed 12! = 479001600 is correct. The prefix has not made an incorrect assertion about the missing digit; the eventual wrong answer is not evidence of an error at this cut.'),
 (1,'intermediate_algebra:607',r'\frac{1 + r + r^2 + r^3 + r^4}{a} = 10',
  'For roots a, ar, ar^2, ar^3, ar^4, the reciprocal sum is (1+r+r^2+r^3+r^4)/(a*r^4), not the displayed expression. The missing r^4 propagates into a^2=4. Identity counterexample: a=1,r=2 gives 31/16 versus 31.'),
 (2,'precalculus:264','The normal vector to the plane can be found by taking the cross product of the direction vector of the line and any vector that lies in the plane.',
  'The next sentences explicitly choose the displacement along the same line, (1,3,-1), as the second vector. Its cross product with the direction (1,3,-1) is zero, which cannot define a plane normal. A second nonparallel in-plane direction is required. This invalid construction is already specified in the prefix, although the cross product has not yet been evaluated.'),
 (3,'counting_and_probability:380',None,
  'The nth row has n+1 binomial coefficients with sum 2^n, so the displayed mean 2^n/(n+1) is correct. No incorrect row number or subsequent calculation is present at this cut.'),
 (4,'intermediate_algebra:439',r'\omega + a - b\omega^2 + a\omega + 1 = 0 \implies (1 + a)\omega - b\omega^2 + 1 = 0',
  'The implication drops the constant a. Combining terms gives (1+a)*omega-b*omega^2+a+1, not the displayed expression. The difference is exactly a. This is an algebra error independently of the questionable use of a nonreal candidate for a real-root problem.'),
 (5,'number_theory:183',None,
  'The two-digit-square search range 4 through 9 is correct: 3^2<10, 4^2>=10, 9^2<=99, and 10^2>99. No erroneous enumeration of squares ending in 6 has yet occurred.'),
 (6,'counting_and_probability:241',r'\frac{1}{n} \sum_{d | n} \phi(d) k^{n/d}',
  'The surrounding sentence explicitly presents this as counting necklaces with both rotations and reflections identified. It is instead a rotation-only formula for k-color words allowing reuse; it does not count arrangements of five selected distinct beads modulo the dihedral group. Five distinct selected beads have 5!/10=12 arrangements, whereas substituting n=k=5 in the stated formula gives 629. Formula-level error observed; the prefix subsequently mentions the correct dihedral-group framework but does not retract the formula or apply it numerically. This label does not assert that the formula determines the eventual wrong answer.'),
 (7,'prealgebra:171','From July 27 to September 3 is 50 days in a non-leap year.',
  'Inclusive count is 5 July days + 31 August days + 3 September days = 39. Leap day lies outside the interval, so it is also 39 in a leap year, not the following claimed 51. The nine-year count is otherwise correct.'),
 (8,'intermediate_algebra:568',r'30c = c.',
  'The derivation identifies the third root with the constant parameter because both were named c. With third root z and parameter C, Vieta gives 30*z=C, not 30*z=z. In fact 30+z*(10-z)=55 implies z=5 and C=150, contradicting the displayed c=0 deduction.'),
 (9,'intermediate_algebra:434',None,
  'The substitution y=log_2(x), log_8(x)=y/3, and log_8(y)=log_2(y)/3 are valid on the domain of the original logarithms. The prefix ends at the correct transformed equation; no later algebra is used for this label.'),
 (10,'intermediate_algebra:417',r'27 - 6c_2 + c_1 = 1',
  'The coefficient of x on the original left side x^3+x^2-5 is 0, not 1. The correctly expanded right side has x coefficient 27-6*c2+c1, which must therefore equal 0.'),
 (11,'precalculus:97',None,
  'The law-of-cosines derivation gives 5b^2-2bc-7c^2=0 and the positive ratio b=7c/5. The parenthetical "not an integer" is overbroad, but it is immediately corrected in this same prefix by requiring 5 to divide c; the retained side lengths a=6k,b=7k,c=5k are consistent. Conservatively no unresolved mathematical error established at the cut; do not count an explicitly corrected aside as a recovery case.'),
 (12,'number_theory:451',r'b^5 + b^2 - 1',
  'Expanding (b^4+b^3+b+1)*(b-1) gives b^5-b^3+b^2-1. The displayed simplification drops -b^3. For b=2 the product is 27, whereas the stated expression gives 35.'),
 (13,'intermediate_algebra:571',None,
  'Only the substitution y=x^2-10x-69 has been stated. It is valid; the transformed denominators or later solution have not yet been written.'),
 (14,'prealgebra:567',None,
  'The prime factorizations of 100 and 150 and gcd=50 are correct. The prefix has not yet enumerated or counted the positive two-digit divisors.'),
 (15,'algebra:912',r'\left( \frac{729}{16} \right) b^6 = \frac{32}{27}',
  'After substituting a=(27/4)b^3, a^2*b^3 equals (729/16)*b^9, not (729/16)*b^6: an extra factor b^3 was lost. Later root-taking errors are also visible, but this first explicit exponent error is sufficient.'),
 (16,'prealgebra:855',None,
  'All four displayed decimal/fraction conversions are correct. No incorrect choice of largest/smallest or subtraction appears before the frozen cut.'),
 (17,'precalculus:515',r'z = \frac{s}{4}',
  'Setting 6x=-y=-4z=s implies z=-s/4, not +s/4. The wrong sign is then included in the second direction vector and its dot product.'),
 (18,'intermediate_algebra:573',None,
  'The prefix only states an intention to determine f1 and f2. Those individual values need not be uniquely identifiable from f7 alone, but no values, false recurrence calculation, or uniqueness claim has yet been derived. An unnecessary strategy is not enough to label a concrete prefix error.'),
 (19,'precalculus:429',None,
  'Distributivity and c cross a = -(a cross c) are correctly used. Although b cross c is recalled, the prefix ends before substituting it for c cross b; do not infer a later sign error at this earlier cut.'),
 (20,'intermediate_algebra:58',r'(f(x) - r_1)(f(x) - r_2)(f(x) - r_3)(f(x) - r_4) = x^4 - 14x^3 + 62x^2 - 91x + 42',
  'For roots r_i of the quartic H, the product shown is H(f(x)), not H(x). The original identity is (f(x)-5)*(f(x)-6)=H(x). Since g is quadratic and H has degree 4, f has degree 2; the erroneously displayed product then has degree 8, contradicting its degree-4 right side.'),
 (21,'intermediate_algebra:445',None,
  'The binomial expansion and conditions that 12-k be even and k be divisible by 3 are correct for the distinct prime radicals. No enumeration or sum has been carried out at the cut.'),
 (22,'counting_and_probability:408','Therefore, the number of valid ways in this case is:\n  \\[\n  342 - 306 = 36\n  \\]',
  'With Bob fixed as President, excluding Alex leaves 18 choices for Vice President and 17 for Treasurer, hence 306 valid assignments. The displayed 342-306=36 counts assignments including Alex, which are invalid. The arithmetic is correct but its interpretation as the valid count is reversed.'),
 (23,'intermediate_algebra:262',None,
  'The symbol c is ambiguously reused for a root and the polynomial constant. From the sum-of-roots condition the distinguished third root really is -4. The product statement abc=-c is notation-ambiguous, and the prefix has not unambiguously used that root value as the original constant in a subsequent computation. Conservatively label unknown rather than treating the correct third-root deduction as an already-proven wrong constant answer.'),
 (24,'precalculus:64',None,
  'The double-angle substitution, factorization sin(x)*(2*sin(x)^2-3+3*cos(x))=0, and the listed sin(x)=0 solutions 0,pi,2pi are all correct. The other factor has not been solved yet.'),
 (25,'number_theory:231',r'1399 = 8 \cdot 160 + 159',
  'The remainder is 1399-8*160=119, not 159. The displayed right side equals 1439, so the Euclidean algorithm is already based on a false equation.'),
 (26,'algebra:117',r'1010101 - 1001 = 1000100',
  'The subtraction equals 1009100, not 1000100. The preceding proposed factorization also changes the original expression, but this explicit arithmetic error alone establishes an erroneous prefix.'),
 (27,'number_theory:445',None,
  'The prefix only gives the correct factorial-product definition of 20!. It has not yet computed or inferred its first digit.'),
 (28,'intermediate_algebra:580',r'(4 + 4\sqrt{3} + 3)y^2 + y^2 = (7 + 4\sqrt{3})y^2',
  'The additional y^2 was dropped: the coefficient is 8+4*sqrt(3), not 7+4*sqrt(3). The earlier omission of an absolute value inside a plus/minus root does not change the solution set and is not used to justify this label.'),
 (29,'intermediate_algebra:569',None,
  'The fixed-point implication is correct. Linearity is explicitly introduced as a hypothesis, and the displayed coefficients and rationalization are correct within that hypothesis. General linearity has not been proved, but the prefix has not yet asserted that all solutions must be linear or drawn a wrong n from it; conservatively unknown.'),
 (30,'intermediate_algebra:881','Therefore, there are no holes.',
  'The original rational function excludes x=0, where its common factor x was canceled. The simplified expression has finite limit -1/9 there, so the original graph has one removable hole. Checking only the simplified numerator/denominator loses this domain exclusion; a=0 is incorrect.'),
 (31,'algebra:1149',None,
  'The polynomial notation and discriminants 16-4a and 16-4b are correct. The ratio condition has not yet been transformed or solved at the frozen boundary.'),
]

def main():
    dest=H/'results/math_slot_20261106/first32'
    request_path=dest/'cut_review_request.json'
    request=C.read_json(request_path)
    assert len(request['questions'])==32 and request['created_before_evaluation']
    assert not (dest/'cut_annotations.json').exists(), 'Never overwrite a frozen annotation'
    assert not list((dest/'responses').glob('*/*.json')), 'Review must precede four-arm outcomes'
    E.prepared()  # verifies unchanged math source, dependencies and candidate pool
    annotations={}
    for (index,qid,quote,reason),q in zip(REVIEWS,request['questions']):
        assert index==len(annotations) and qid==q['id']
        path=H/'results/math_slot_20261106/student'/(qid+'.json')
        assert C.sha_file(path)==q['student_sha']
        proposal=q['student']['cut_proposal'];text=q['student']['generation']['text']
        prefix=text[:proposal['prefix_end']]
        assert prefix==proposal['prefix'] and proposal['prefix_end'] in M.cuts(text)
        a=dict(prefix_end=proposal['prefix_end'],cohort='error_observed' if quote else 'error_unknown',
            reviewer='Codex agent mathematical prefix review, 2026-11-06',reason=reason,
            student_sha256=q['student_sha'],prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
            original_proposed_boundary_retained=True)
        if quote:
            assert quote in prefix,(qid,'quote mismatch')
            a['error_quote']=quote
        assert M.validate_annotation(text,a)==prefix
        annotations[qid]=a
    counts=dict(Counter(a['cohort'] for a in annotations.values()))
    assert counts=={'error_unknown':16,'error_observed':16}
    obj=dict(request_sha=C.sha_file(request_path),stage='first32',questions=annotations,
        reviewed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        review_method='Agent reads each original problem and frozen proposed prefix; checks mathematical claims. No evaluation outcomes, gold solution, Student suffix or final wrong-answer inference used to assign cohorts.',
        boundary_policy='Retain every original proposed paragraph boundary; no moving cuts to find an error.',
        unknown_policy='No definite error yet, insufficient derivation, notation ambiguity, or an explicitly self-corrected aside are conservatively separate. Unknown does not assert the entire prefix is correct.',
        counts=counts,review_script_sha256=C.sha_file(Path(__file__)),math_source=E.fingerprint(),
        limitations='Single agent review, not a second independent human adjudication. Labels locate an error already written, not its causal responsibility for the eventual wrong answer. Pilot is not a confirmatory outcome analysis.')
    assert set(annotations)==set(request['selected_ids'])
    draft=dest/'cut_annotations.reviewed_draft.json';C.save_json(draft,obj)
    C.save_json(dest/'cut_review_verification.json',dict(passed=True,questions=32,counts=counts,
        all_proposed_boundaries_retained=True,all_quotes_within_prefix=True,
        all_student_hashes_match=True,no_four_arm_results_at_review=True,
        request_sha256=obj['request_sha'],annotations_sha256=C.sha_file(draft),
        review_script_sha256=obj['review_script_sha256'],math_source_unchanged=True))
    # Publish only after validating all 32 annotations. The existing runner resumes on this file.
    draft.replace(dest/'cut_annotations.json')
    assert E.cut_annotations('first32',32)==obj
    print('PUBLISHED',counts,'all 32 proposed cuts unchanged',flush=True)

if __name__=='__main__':main()
