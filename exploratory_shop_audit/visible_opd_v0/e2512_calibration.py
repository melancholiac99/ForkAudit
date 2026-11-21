"""Finish the already-running Section 25.11 calibration independently of the new formal layout."""
import fcntl
import os

from .e25_driver import cleanup, commit_file, freeze, trajectory_stats
from .e25_layout import Layout
from .e25_measure import merge_update
from .e25_probe import PUBLIC, RUN
from .e2511_driver import progress, run_round
from .e2511_schedule import CAL, MATCH, classify
from exploratory_shop_audit.grpo_forced_audit.common import read, save, utc


def main():
    os.umask(0o077)
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cal=Layout(CAL)
        assert all(progress(cal,i,MATCH).exists() for i in range(1,5))
        if not progress(cal,5,MATCH).exists():
            merge_update(cal,5,[MATCH])
            s=trajectory_stats(cal,5,MATCH)
            cleanup(cal,5,MATCH)
            freeze(progress(cal,5,MATCH),dict(run=CAL,update=5,arm=MATCH,before_sha=s['before_sha'],
                                               after_sha=s['after_sha'],stopped=False))
        for k in range(6,11):
            if not progress(cal,k,MATCH).exists():
                run_round(k,[(CAL,k,MATCH)],{})
        probe=read(cal.root/'monitor/010.json')[MATCH]
        ratio,label=classify(probe['stats']['all']['kl'],cal.config['target'])
        assert all(progress(cal,i,MATCH).exists() for i in range(1,11))
        result=dict(rho_double_prime=ratio,label=label,learning_rate=cal.config['rates'][MATCH],
                    frozen=True,reference_movement=cal.config['target'],readout=probe,
                    null_reference_file='movement.json',calibration_final_sha=read(cal.folder(10,MATCH)/'optimizer.json')['after_sha'])
        freeze(PUBLIC/'calibration_2511_result.json',result)
        commit_file(PUBLIC/'calibration_2511_result.json','Record the second calibration and freeze the matched rate for serial Stage E')
        save(cal.root/'complete_2512.json',dict(time=utc(),final_step=10,**result))


if __name__=='__main__':
    main()
