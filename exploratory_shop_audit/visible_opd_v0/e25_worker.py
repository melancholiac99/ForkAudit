"""Execute one isolated multi-GPU E operation. Phase boundaries have durable outputs."""
import sys
from . import d_collect,d_targets,d_train,d_eval
from .e25_layout import Layout
from .e25_probe import configure
from .e25_measure import per_update,common

def main():
    phase,run,step,*arms=sys.argv[1:]
    k=int(step);rank=configure();l=Layout(run)
    if phase=='train':
        d_train.train(k,arms=arms,folder=l.folder,state_dir=l.state,snapshot=l.snapshot,load_batch=l.batch,rates=l.config['rates'])
    elif phase=='per_update':per_update(rank,l,k,arms)
    elif phase=='common':common(rank,l,k,arms)
    elif phase=='eval' and k==0:d_eval.run(k,rank,layout=l)
    elif rank<len(arms):
        arm=arms[rank]
        if phase=='collect':d_collect.collect(k,rank,arm,l.folder,l.state)
        elif phase=='targets':d_targets.build(k,arm,layout=l)
        elif phase=='eval':d_eval.run(k,rank,layout=l,arm_override=arm)
        else:raise ValueError(phase)
if __name__=='__main__':main()
