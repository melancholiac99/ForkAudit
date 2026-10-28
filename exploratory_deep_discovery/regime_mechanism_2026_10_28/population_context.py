"""One prespecified descriptive cross-table; no new predictive feature search."""
from pathlib import Path
import hashlib
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
source=HERE.parent/'exploratory_point_mechanisms.parquet'
p=pd.read_parquet(source)
p['old_regime']=['Dead' if not live else 'Fork' if fork else 'Indifferent'
                 for live,fork in zip(p.live,p.het)]
counts=pd.crosstab(p.old_regime,p.n_target_mention)
counts.to_csv(HERE/'population_target_arm_counts.csv')
summary={}
for name,g in p.groupby('old_regime'):
    summary[name]=dict(n=len(g),target_feedback_any=int(g.any_target_mention.sum()),
        target_feedback_any_fraction=float(g.any_target_mention.mean()),
        new_instance_feedback_any=int(g.any_new_to_history.sum()),
        new_instance_feedback_any_fraction=float(g.any_new_to_history.mean()),
        same_task_sibling_success_mean=float(g.g_task_success.mean()))
live=p[p.live]
by_count=[dict(target_arms=int(k),n=len(g),detected_fork=int(g.het.sum()),
               detected_fork_fraction=float(g.het.mean()))
          for k,g in live.groupby('n_target_mention')]
out=dict(status='exploratory descriptive; not causal or population mechanism identification',
    input=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    definitions='target feedback: original per-arm response contains task target instance; '
                'new instance: absent from full factual history. Old frozen labels retained.',
    groups=summary,live_by_number_of_target_responses=by_count,
    caution='Task/trajectory/progress confounding is uncontrolled. Do not infer a monotone law '
            'from arm-count slices; 4-target-arm slice has only 33 Live points.')
(HERE/'population_context.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
