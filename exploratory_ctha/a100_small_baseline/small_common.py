"""Four-task namespace; unchanged numerical implementations remain explicit dependencies."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'a100_full_response'
sys.path.insert(0, str(OLD))
import protocol_common as base
from protocol_common import ENGINEERING, FAST, MODEL, cpu_model_lock, fingerprint, hardware, load, save, sha

PILOT = Path(os.environ.get('FULL_RESPONSE_PILOT', ROOT / '.runtime/runs/ctha_a100_small_baseline_r1'))
NUMERIC_SOURCES = base.sources


def sources():
    result = NUMERIC_SOURCES()
    result.update({str(p.relative_to(ROOT)): sha(p) for p in [*HERE.glob('*.py'), HERE / 'PLAN.json']})
    return dict(sorted(result.items()))


base.PILOT = PILOT
base.sources = sources
base.TB = ROOT / '.runtime/tensorboard/ctha-a100-r2/four_task_baseline'
configure = base.configure
emit = base.emit
