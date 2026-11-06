"""Full-cohort prospective prefix review; reuse pilot annotations verbatim."""
from pathlib import Path
from collections import Counter
from fractions import Fraction as Q
import hashlib,sys,time
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import exp2_common as C,exp6_math_slot as E,math_slot_core as M

# Decisions made from the original problem and frozen prefix only.
# None is deliberately error_unknown, not an assertion that the proof is complete.
REVIEWS=[
('algebra:932',None,'The interval between 9 a.m. and 6 p.m. is indeed nine hours. The prefix has not counted tour start times or omitted an endpoint; no off-by-one error is present yet.'),
('algebra:748',None,'The first and last multiples are correctly identified as 105 and 196, and the arithmetic progression has common difference seven. Neither the number of terms nor its sum has been incorrectly computed at this cut.'),
('number_theory:510',None,'Every displayed multiplication is correct, including the product 151200. The prefix has not selected the hundreds digit; the later wrong answer cannot establish an error here.'),
('intermediate_algebra:141',None,'This prefix only restates the sum and announces a step-by-step approach. It contains no transformed sum or mathematical inference to classify as erroneous.'),
('intermediate_algebra:888',None,'The first locus is Re(z)=6. Substitution z=6+bi gives the correctly displayed ratio sqrt(36+b^2)/sqrt(36+(b-8)^2)=5/3. No subsequent squaring or root error is available at this cut.'),
('intermediate_algebra:415',r'r_k - 1 + \frac{2}{r_k + 1}',
 'Polynomial division gives r^2/(r+1)=r-1+1/(r+1), not r-1+2/(r+1). The displayed identity is too large by 1/(r+1), nonzero for these roots. This explicit algebra error suffices; the separate cyclotomic-polynomial misnaming is not needed for the label.'),
('precalculus:416',None,'The identity sin(pi-x)=sin(x) and the three pairings are correct, as is doubling the sum at angles pi/12,3pi/12,5pi/12. No faulty value evaluation has yet appeared.'),
('number_theory:182',None,'The root range 5 through 9 and the five cubes 125,216,343,512,729 are correct. The additional requirement that the cube root be a square or cube has not yet been applied; listing the candidate pool is not itself an error.'),
('number_theory:167',r'5^7 \cdot 2^8 = 5^7 \cdot 2^7 \cdot 2 = 128 \cdot 78125 = 10000000',
 'The equality chain drops the final factor 2 when replacing 5^7*2^7*2 with 128*78125. The original denominator is 20,000,000. The resulting claimed fraction 4321/10,000,000 is twice the given fraction; the standalone product 128*78125 itself is correct.'),
('number_theory:92',r'the largest possible value for \(k\) is 503.',
 '37*503=18611 exceeds 18609, so k<18609/37 implies k<=502. In particular 37*503+23=18634 is not less than 18632. The approximate division has already produced an invalid integer bound.'),
('precalculus:399',None,'The tangent-of-sum identity, csc(pi/3)=2*sqrt(3)/3, and cos(gamma)=-cos(alpha+beta) are correct. The ratio has not yet been simplified incorrectly.'),
('precalculus:243',None,'The four linear equations correctly encode the two transformations, and solving equations 1 and 3 gives a=4,b=-1. The prose referring to the first two equations is imprecise but the actual displayed calculation uses the correct pair. The line image is not yet derived.'),
('intermediate_algebra:646',None,
 'The bd term is a constant in x, so direct coefficient comparison should retain bd=0 and a+cd=1. However the displayed weaker condition a+bd+cd=1 is a valid necessary consequence of the original universal equation (set x=1), and under the full original condition bd=0 the rewritten equation is also true. The prefix has not yet claimed this weaker system is sufficient or derived an incorrect d. Conservatively classify the incomplete/misleading manipulation as unknown rather than infer a false prefix value from the final wrong answer.'),
('number_theory:49',None,'The remainder computation 200=28*7+4 is correct. The direction and number of weekday shifts have not yet been evaluated.'),
('intermediate_algebra:125',None,'The diagonal midpoint is (5,7), its length is sqrt(52), and the circumradius is sqrt(13). All deductions shown at the cut are correct; the expanded circle coefficients are not yet written.'),
('prealgebra:647',None,'Each of the three quarter-hour events occurs 24 times in the specified 24-hour interval; the counts 48,96,144 are correct. Hourly chimes and the final total have not yet been computed.'),
('intermediate_algebra:583',None,'The remainder theorem correctly uses x=-5 and f(-5)=4. The displayed substitution includes every term with the correct argument; subsequent powers/arithmetic or solving for k are not present.'),
('precalculus:405',None,'cos(70 degrees)=sin(20 degrees), cos(50 degrees)=sin(40 degrees), and cos(30 degrees)=sqrt(3)/2 are all correct. Reordering the remaining product introduces no error.'),
('geometry:469',None,'Using the two given 35-degree angles yields angle ABC=110 degrees. The prefix has not applied an inscribed-angle relation to CDA yet; no error is established by this preliminary calculation.'),
('intermediate_algebra:253','we need to ensure that they intersect at exactly one point',
 'Tangency is a local common-tangent condition and does not require exactly one intersection. For k=-3/4 these two parabolas are tangent at (-1/2,-1/2), where their slopes are both -1, and also intersect at (3/2,3/2). Thus the stated necessity is false. The subsequent algebra in the expressly conditional b!=a branch is valid and is not the basis of the error label; this label makes no claim about the eventual selected k.'),
('counting_and_probability:412',None,'The displayed counts for partitions of types 4,3+1,2+2 are respectively 1,4,3 and correctly account for unlabeled boxes. The remaining type 2+1+1 has not yet been counted or omitted from a claimed final total.'),
('counting_and_probability:329',None,'Only the correct counts of candidates for the four offices have been restated. No multiplication, factorial, or ordering inference appears at the boundary.'),
('algebra:753',None,'The geometric-term equations and r^6=13/9477 are correct, including 9477=81*117. Repeated equivalent expressions are redundant but not erroneous. No sixth-root or seventh-term calculation is present yet.'),
('intermediate_algebra:546',r'\frac{f(1) f(1)}{f(-1)}',
 'For n=1,m=0, the second denominator is f(n-m-1)=f(0), not f(-1). The valid equation actually determines f(2)=2; the prefix incorrectly rejects that allowed case as undefined. Later it corrects the separate n=2,m=1 expression, but does not retract the invalid rejection of n=1,m=0. The incorrect index substitution is already explicit.'),
('prealgebra:310',r'The quadrilateral \(PTRS\) can be divided into two triangles: \(\triangle PTS\) and \(\triangle PRT\).',
 'These triangles share PT and have S and R on the same side of PT, so they overlap and do not partition PTRS. Using the prefix coordinates, cross(PT,PS)=-l*w and cross(PT,PR)=-l*w/2. Valid diagonal decompositions are PTS+TRS or PTR+PRS. The two incorrectly proposed areas happen to sum to the quadrilateral area in this midpoint configuration, so this is a geometric decomposition error, not a claim that its area sum must be wrong or that it causes the final wrong answer.'),
('prealgebra:410',r'\text{Total area} = 20 \, \text{yd}^2 + 6 \, \text{yd}^2 = 26 \, \text{yd}^2',
 'The supplied diagram is an L-shaped orthogonal polygon with vertices (0,0),(9,0),(9,2),(4,2),(4,5),(0,5). Its area is 4*5+5*2=30 square yards, not 26. The supposed extra triangle is a rectangular 5-by-2 region. The incorrect total area is already explicitly written.'),
('geometry:95','The distance between any pair of parallel sides in a regular hexagon is equal to the height of one of these equilateral triangles.',
 'The center-to-side distance is the altitude of one of the six equilateral triangles, but the distance between opposite parallel sides is twice that altitude. For side length six it is 6*sqrt(3), not 3*sqrt(3). The factor-of-two geometric identification is already wrong before any subsequent numerical evaluation.'),
('prealgebra:753',None,'All displayed prime divisions are correct and establish 2400=2^5*3*5^2. No incorrect count of prime factors, with or without multiplicity, has yet been stated.'),
]

def main():
    root=E.OUT;dest=root/'full';request_path=dest/'cut_review_request.json'
    req=C.read_json(request_path);pilot_path=root/'first32/cut_annotations.json';pilot=C.read_json(pilot_path)
    budget=C.read_json(root/'full_budget.json');frozen=C.read_json(root/'full_budget_inputs.json')['frozen_cut_proposals']
    assert len(req['questions'])==60 and len(pilot['questions'])==32 and req['created_before_evaluation']
    assert not (dest/'cut_annotations.json').exists() and not list((dest/'responses').glob('*/*.json'))
    E.prepared();old_hash=C.sha_file(pilot_path)
    assert budget['pilot_verification_sha']==C.sha_file(root/'first32/verification.json')
    remaining=[q for q in req['questions'] if q['id'] not in pilot['questions']]
    assert [q['id'] for q in remaining]==[q[0] for q in REVIEWS]
    # Exact arithmetic checks supporting borderline/explicit review decisions.
    assert 5*6*7*8*9*10==151200 and 5**7*2**8==20000000
    assert 37*503+23>=18632 and 37*502+23<18632
    assert Q(18609,37)<503 and Q(18609,37)>502
    for x in (Q(-1,2),Q(3,2)):assert x*x-Q(3,4)==x
    assert 2*Q(-1,2)==1/(2*Q(-1,2))==-1
    assert 4*5+5*2==30 and 2400==2**5*3*5**2
    decisions={q:(quote,reason) for q,quote,reason in REVIEWS};annotations={}
    for q in req['questions']:
        qid=q['id'];student=q['student'];text=student['generation']['text'];proposal=student['cut_proposal']
        prefix=text[:proposal['prefix_end']]
        assert prefix==proposal['prefix'] and proposal['prefix_end']==frozen[qid]['prefix_end']
        assert C.sha_file(root/'student'/(qid+'.json'))==q['student_sha']==frozen[qid]['student_sha256']
        if qid in pilot['questions']:
            a=pilot['questions'][qid]
        else:
            quote,reason=decisions[qid]
            a=dict(prefix_end=proposal['prefix_end'],cohort='error_observed' if quote else 'error_unknown',
                reviewer='Codex agent prospective mathematical prefix review, 2026-11-07',reason=reason,
                student_sha256=q['student_sha'],prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                original_proposed_boundary_retained=True)
            if quote:a['error_quote']=quote
        assert M.validate_annotation(text,a)==prefix
        annotations[qid]=a
    counts=dict(Counter(a['cohort'] for a in annotations.values()))
    new_counts=dict(Counter(annotations[q['id']]['cohort'] for q in remaining))
    assert counts=={'error_unknown':36,'error_observed':24}
    assert new_counts=={'error_unknown':20,'error_observed':8}
    assert all(annotations[qid]==a for qid,a in pilot['questions'].items()) and C.sha_file(pilot_path)==old_hash
    obj=dict(request_sha=C.sha_file(request_path),stage='full',questions=annotations,counts=counts,
        new_counts=new_counts,pilot_annotations_sha256=old_hash,pilot_reused_verbatim=32,
        full_budget_sha256=C.sha_file(root/'full_budget.json'),math_source=E.fingerprint(),
        review_script_sha256=C.sha_file(Path(__file__)),reviewed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        review_method='28 new problems and frozen prefixes reviewed without inspecting Student suffix, gold solutions, Teacher screening answers, or four-arm outcomes; 32 pilot objects copied verbatim.',
        boundary_policy='All 60 original proposed boundaries retained, no error-enrichment or exclusions.',
        unknown_policy='Unknown covers incomplete derivations, ambiguous reasoning, or correct preliminary work; final wrong answers are not evidence of prefix errors.',
        limitations='Single-agent adjudication, not independent human replication. An observed mathematical or structural error need not cause the final wrong answer; for intermediate_algebra:253 the error is an asserted necessary condition, for prealgebra:310 an invalid partition whose areas coincidentally add correctly. Full inference uses registered separate cohorts.')
    draft=dest/'cut_annotations.reviewed_draft.json';C.save_json(draft,obj)
    C.save_json(dest/'cut_review_verification.json',dict(passed=True,questions=60,pilot_reused_verbatim=32,newly_reviewed=28,
        counts=counts,new_counts=new_counts,all_frozen_boundaries_retained=True,student_hashes_match=True,
        no_four_arm_results_at_review=True,pilot_file_unchanged=True,request_sha256=obj['request_sha'],
        annotations_sha256=C.sha_file(draft),review_script_sha256=obj['review_script_sha256'],math_source_unchanged=True))
    print('DRAFT_VERIFIED',counts,new_counts,flush=True)

if __name__=='__main__':main()
