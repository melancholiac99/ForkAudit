"""Run the registered endpoint evaluations on the two original four-rank device sets."""
from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
import time

from .gap_run import GapLayout, LANE_DEVICES, PUBLIC, launch, qualify_pages
from exploratory_shop_audit.grpo_forced_audit.common import read


def evaluate_arm(layout, arm, endpoints, baselines):
    devices=LANE_DEVICES[arm]
    if arm=='R-gap':
        try:
            for pool,ready in baselines.items():
                launch(layout,'eval',0,'B',pool,'free',devices=devices)
                qualify_pages(layout,pool)
                ready.set_result(True)
        except BaseException as error:
            for ready in baselines.values():
                if not ready.done():
                    ready.set_exception(error)
            raise
    end=endpoints[arm]
    for k in sorted({k for k in (20,30,50) if k<=end}|{end}):
        launch(layout,'eval',k,arm,'V','free',devices=devices)
    page_models=[('B',0),(arm,end)] if arm=='R-gap' else [(arm,end)]
    for pool,ready in baselines.items():
        ready.result()
        for model,k in page_models:
            launch(layout,'eval',k,model,pool,'page',devices=devices)
            launch(layout,'mass',k,model,pool,devices=devices)


def evaluate(layout):
    endpoints={a:read(layout.root/'endpoints'/f'{a}.json')['update'] for a in LANE_DEVICES}
    baselines={pool:Future() for pool in ('T_opd_eval','V')}
    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs=[workers.submit(evaluate_arm,layout,arm,endpoints,baselines) for arm in LANE_DEVICES]
        for job in jobs:
            job.result()
    from .gap_report import report
    report(layout,endpoints)


def main():
    layout=GapLayout()
    # The original one-lane finalizer uses this same lock and checks for the finished report.
    with (layout.root/'endpoint_eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while not all((layout.root/'endpoints'/f'{arm}.json').is_file() for arm in LANE_DEVICES):
            if (layout.root/'operator_hold.json').exists():
                return
            time.sleep(8)
        if not (PUBLIC/'endpoint_report.json').is_file():
            evaluate(layout)


if __name__=='__main__':
    main()
