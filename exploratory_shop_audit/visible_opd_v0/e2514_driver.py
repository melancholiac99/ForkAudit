"""Run the two registered four-card groups after both migration replays have passed."""
import csv
import fcntl
import os
import sys
from .d_data import folder as d_folder, state_dir as d_state
from .e25_layout import Layout
from .e25_paths import PUBLIC, RUN, SERIAL
from .e25_behavior import report
from .e25_measure import merge_common, merge_update
from .e25_driver import cleanup, freeze, trajectory_stats
from .e2511_targets import merge
from .e2512_driver import launch
from .e2512_native import merge_tapes
from .e2512_worker import target_shards
from .e2513_adopt import adopt_prefix, copy_state, place
from .e2514_gate import control_authorized
from exploratory_shop_audit.grpo_forced_audit.common import read, save, utc

OUT=PUBLIC/'migration_2514'


def frozen_prefix(l):
    arm='P-frozen'
    table=PUBLIC.parent/'section23_20261120/training_steps.tsv'
    with table.open() as src:
        records={int(x['update']):x for x in csv.DictReader(src,delimiter='\t') if x['arm']=='P'}
    receipt=read(d_folder(10,'P')/'optimizer.json')
    assert receipt['after_sha']==records[10]['updated_sha']
    copy_state(d_state(10,'P'),l.state(10,arm),receipt)
    place(d_folder(10,'P')/'optimizer.json',l.folder(10,arm)/'optimizer.json')
    save(l.folder(10,arm)/'train_statistics.json',dict(legal_rate=float(records[10]['legal_rate']),origin='D/P@10'))
    save(l.root/'adoption'/f'{arm}.json',dict(initial_training_legal_rate=float(records[1]['legal_rate']),
        old_layout_through=10,mechanism_only=True,origin='D/P@10',projector_version=0,optimizer=receipt))


def evaluate(l,k,arm):
    dest=OUT/arm
    launch(l,'common',k,arm)
    monitor=merge_common(l,k,[arm],dest/'monitor'/f'{k:03}.json')
    if arm=='R-matched' and k==10:
        from .e2511_schedule import classify
        ratio,label=classify(monitor[arm]['stats']['all']['kl'],read(l.root/'monitor/010.json')['P']['stats']['all']['kl'])
        save(dest/'movement_match_at_10.json',dict(ratio=ratio,label=label,feedback_to_lr=False))
    launch(l,'eval',k,arm)
    endpoints={'B':0,arm:k,'P':min(k,30)}
    report(l,k,endpoints,dest/'behavior'/f'{k:03}.json')


def train(arm):
    l=Layout(SERIAL)
    group=os.environ['OPD_MIGRATION_GROUP']
    assert arm in ({'R-matched'} if group=='A' else {'P-frozen','R'})
    control_authorized(l)
    dest=OUT/arm
    end=30 if arm=='P-frozen' else 50
    l.initialize(end,arms=[arm])
    if arm=='P-frozen':
        if not (l.root/'adoption'/f'{arm}.json').exists():frozen_prefix(l)
        if not (dest/'behavior/010.json').exists():evaluate(l,10,arm)
        start=11
    else:
        if arm=='R':assert (OUT/'P-frozen/endpoint.json').exists()
        adopt_prefix(l,arm)
        start=6
    for k in range(start,end+1):
        receipt=l.root/'arms'/arm/'updates'/f'{k:03}.json'
        if receipt.exists():
            if read(receipt)['stopped']:break
            continue
        launch(l,'collect',k,arm)
        merge_tapes(l,k,arm)
        launch(l,'targets',k,arm)
        merge([(SERIAL,k,arm)],target_shards(l,k,arm))
        launch(l,'train',k,arm)
        launch(l,'per_update',k,arm)
        merge_update(l,k,[arm])
        stats=trajectory_stats(l,k,arm)
        stopped=False
        if k%10==0:
            if k>=20:
                baseline=read(l.root/'adoption'/f'{arm}.json')['initial_training_legal_rate']
                previous=read(l.folder(k-10,arm)/'train_statistics.json')['legal_rate']
                stopped=baseline-max(previous,stats['legal_rate'])>=.1-1e-12
            evaluate(l,k,arm)
        save(receipt,dict(arm=arm,update=k,stopped=stopped,physical_group=group,
             optimizer_steps=stats['optimizer_steps'],before_sha=stats['before_sha'],after_sha=stats['after_sha'],finished=utc()))
        cleanup(l,k,arm)
        if stopped:break
    freeze(dest/'endpoint.json',dict(arm=arm,step=k,early_stop=read(receipt)['stopped'],mechanism_only=arm=='P-frozen',
           adoption=read(l.root/'adoption'/f'{arm}.json'),terminal_optimizer=read(l.folder(k,arm)/'optimizer.json')))


if __name__=='__main__':
    group=os.environ['OPD_MIGRATION_GROUP']
    with (RUN/'migration_2514'/f'driver_{group}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in sys.argv[1:]:train(arm)
