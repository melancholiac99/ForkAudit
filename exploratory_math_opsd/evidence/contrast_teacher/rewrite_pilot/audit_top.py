"""Manual coding of the preselected extreme sampled-token positions, ordered as in top.json."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
top=json.loads((p/'top.json').read_text())
# Directly implicated numeral, operator, variable or relational modifier.
direct={2,4,7,12,14,15,19,22,33,34,37,48,68,76,93,102}
# Adjacent to a relevant derivation or answer; token alone does not identify the error.
adjacent={1,3,5,6,8,10,11,13,16,17,18,20,21,27,28,36,38,39,40,43,45,47,49,51,52,53,59,61,62,67,70,72,73,74,75,78,79,81,82,86,87,88,91,96,98,99,100,103,105,106,107}
marks={14:'Digit starts the altitude answer 2.4, which changes to 4 under the wrong right-angle assignment.',21:'Pronoun follows the disputed sign rearrangement.',71:'Markdown divider after restating the problem.',99:'Math delimiter before the birthday probability expression.',55:'End-of-turn marker.',36:'The word cases introduces a discussion of sufficient conditions.',13:'Discourse connective after the affected maximum-overlap computation.',62:'Digit 5 is half the sum of bases in both solutions; adjacent to the affected height.',67:'Equality sign in the common base-height formula, before the changed geometry.',97:'Introduces alternate answer formatting.'}
def label(i):
    return 'direct' if i in direct else 'nearby_or_ambiguous' if i in adjacent else 'other'
assert len(top['top'])==109 and set(marks)==set(top['example_indices'])
out=dict(definition='Single-reviewer descriptive coding of the sampled token in context. Direct includes downstream changed numerical answers. No ordinary-position comparison.',counts={k:sum(label(i)==k for i in range(109)) for k in ('direct','nearby_or_ambiguous','other')},labels=[label(i) for i in range(109)],examples=[dict(**top['top'][i],label=label(i),reason=marks[i]) for i in top['example_indices']])
(p/'top_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n')
