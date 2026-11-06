"""Post-hoc reference-ambiguity sensitivity; never changes the primary dataset."""
from pathlib import Path
from fractions import Fraction as F
import sys
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import exp2_common as C
import math_slot_core as M

def main():
    original=H/'results/math_slot_20261106/full'
    out=H/'results/next_slot_supplement_20261106'
    rows=C.read_json(original/'aggregate_rows.json')
    qid='intermediate_algebra:253'
    assert len(rows)==60 and sum(r['id']==qid for r in rows)==1
    paths=[original/n for n in ('aggregate_rows.json','analysis.json','cut_annotations.json','outcome_review.json')]
    before={str(p):C.sha_file(p) for p in paths}
    checks=[]
    for x,k in ((F(1,2),F(1,4)),(F(-1,2),F(-3,4))):
        y=x
        assert y==x*x+k and x==y*y+k and 2*x==1/(2*y)
        checks.append(dict(x=str(x),y=str(y),k=str(k),slope=str(2*x),passed=True))
    kept=[r for r in rows if r['id']!=qid]
    primary=M.analysis(rows)
    sensitivity=M.analysis(kept)
    assert primary['cohorts']==C.read_json(original/'analysis.json')['cohorts']
    result=dict(post_hoc=True,excluded_ids=[qid],primary_questions=60,sensitivity_questions=59,
        reason='Reference omits a shared-tangent solution; final-answer-based Student eligibility is ambiguous. Exclusion is supplementary, not a replacement of the frozen primary analysis.',
        exact_rational_tangency_checks=checks,primary=primary,sensitivity=sensitivity,
        original_hashes=before,script_sha256=C.sha_file(Path(__file__)),
        limitations='One flagged reference issue does not certify all dataset answers. No new significance threshold or outcome-based sampling. Primary labels, responses and cut annotations remain unchanged.')
    assert all(C.sha_file(Path(p))==h for p,h in before.items())
    C.save_json(out/'math_full_reference_ambiguity_sensitivity.json',result)
    for name,data in [('primary',primary),('sensitivity',sensitivity)]:
        c=data['cohorts']['error_observed']
        print(name,c['questions'],c['quote_continue_minus_active_pp'])

if __name__=='__main__':main()
