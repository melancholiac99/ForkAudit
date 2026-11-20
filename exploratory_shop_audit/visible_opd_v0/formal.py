"""Resume the other 92 frozen tasks after the measured budget is authorized."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
from .collect import collect
from .score import score
from .report import report
from exploratory_shop_audit.grpo_forced_audit.common import read, read_rows, digest_file

def check_receipt(run):
    receipt=read(run/'formal_authorization.json')
    amended=receipt['stage']=='section16_P_pre'
    measured_run=Path(receipt['prior_run']) if amended else run
    if receipt['stage'] not in ('remaining_92_P_pre','section16_P_pre') or receipt['smoke_summary_sha256']!=digest_file(measured_run/'smoke_summary.json'):
        raise ValueError('Approval must identify this measured pilot')
    budget=float(receipt['additional_gpu_hours'])
    if not math.isfinite(budget) or budget<=0: raise ValueError('Invalid GPU budget')
    used=sum(r['gpu_hours'] for r in read_rows(run/'preflight_cost_ledger.jsonl') if r['stage'].startswith('formal_'))
    if amended:
        prior=read(measured_run/'formal_authorization.json')
        if prior['additional_gpu_hours']!=budget or prior['smoke_summary_sha256']!=receipt['smoke_summary_sha256']:
            raise ValueError('Budget lineage mismatch')
        used=sum(r['gpu_hours'] for r in read_rows(measured_run/'preflight_cost_ledger.jsonl') if r['stage'].startswith('formal_'))+sum(r['gpu_hours'] for r in read_rows(run/'preflight_cost_ledger.jsonl'))
    if budget<=used:raise ValueError('Formal budget exhausted')
    return budget-used

def main(run):
    os.umask(0o077)
    lock=(run/'formal.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remaining=check_receipt(run)
    if (run/'formal_report/packet_selection.json').exists():raise FileExistsError('Formal selection already recorded')
    def stop(_sig,_frame): raise TimeoutError('formal P-pre GPU allowance exhausted')
    signal.signal(signal.SIGALRM,stop)
    signal.alarm(max(1,int(remaining*3600/4)))
    if read(run/'formal_authorization.json')['stage']=='section16_P_pre' and not (run/'smoke_summary.json').exists():
        collect(run)
        score(run)
        report(run)
    collect(run,formal=True)
    score(run,formal=True)
    signal.alarm(0)
    report(run,formal=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
