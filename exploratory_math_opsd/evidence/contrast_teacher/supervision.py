import json
from contexts import HERE,save


def summarize(output):
    steps={n:[] for n in range(1,201)}
    for rank in range(4):
        rows=[json.loads(line) for line in (output/f'supervision-rank-{rank}.jsonl').read_text().splitlines()]
        for step in steps:
            entries=[r for r in rows if r['step']==step]
            assert len(entries)==2
            steps[step].extend(entries)
    results={}
    for k,rows in steps.items():
        supervised=sum(x['supervised'] for x in rows)
        valid=sum(x['eligible'] for x in rows)
        mass=sum(x['clipped_mass_sum'] for x in rows)
        assert 0<=supervised<=valid and 0<=mass<=supervised+1e-4
        results[k]=dict(supervised=supervised,eligible=valid,mean_supervised_per_rank_microbatch=supervised/8,
                        supervised_fraction=supervised/valid,clipped_target_mass_fraction=mass/supervised if supervised else None)
    save(HERE/'supervision_v2.json',results)
