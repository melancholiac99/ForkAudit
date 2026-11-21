"""Use the full machine for the current serial arm with the signed task placement."""
import sys
from . import d_train, e25_refresh
from .e25_layout import Layout
from .e25_measure import common, per_update
from .e25_probe import configure
from .e2511_targets import score_shard
from .e2512_native import collect, evaluate
from .e2512_routing import SERIAL, owners
from exploratory_shop_audit.grpo_forced_audit.common import read


def target_shards(l,k,a):
    tasks=read(l.folder(k,a).parent/'tasks.json')
    return [[((SERIAL,k,a),t['goal_index']) for t in tasks if owners()[t['goal_index']]==rank] for rank in range(4)]


def main():
    phase,step,arm=sys.argv[1:]
    rank=configure();k=int(step);l=Layout(SERIAL)
    if phase=='collect':collect(l,k,arm,rank)
    elif phase=='eval':evaluate(l,k,arm,rank)
    elif phase=='targets':score_shard(rank,target_shards(l,k,arm))
    elif phase=='train':
        d_train.train(k,arms=[arm],folder=l.folder,state_dir=l.state,snapshot=l.snapshot,load_batch=l.batch,
                      rates=l.config['rates'],compare_original=False)
    elif phase=='per_update':per_update(rank,l,k,[arm])
    elif phase=='common':common(rank,l,k,[arm])
    elif phase.startswith('refresh_'):
        assert arm=='P' and k%10==0
        dst=l.root/'refresh'/f'{k:03}';dst.mkdir(parents=True,exist_ok=True)
        if phase=='refresh_score':e25_refresh.score(rank,SERIAL,k,l.config['window_updates'],dst)
        elif phase=='refresh_fit':e25_refresh.fit(rank,dst)
        else:raise ValueError(phase)
    else:raise ValueError(phase)


if __name__=='__main__':
    main()
