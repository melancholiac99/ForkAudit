"""Independent authorization path for the registered eight-card migration."""
import os
import hashlib
from pathlib import Path
from .e25_paths import RUN
from exploratory_shop_audit.grpo_forced_audit.common import read

def four_card_resume():
    return os.environ.get('OPD_FOUR_CARD_RESUME') == '1'


def resume_verification_dir():
    machine=Path('/sys/class/dmi/id/product_uuid').read_text().strip()
    return RUN/'migration_2514/four_card_resume'/hashlib.sha256(machine.encode()).hexdigest()[:16]


def group_b_devices():
    return '0,1,2,3' if four_card_resume() else '4,5,6,7'


def migration_verified():
    for group in ('A', 'B'):
        record = read(RUN/'migration_2514'/group/'result.json')
        assert record['passed'] and record['targets']['passed'] and record['replay']['passed']
        assert all(p['mismatches'] == 0 for p in record['probes'])
        assert all(p['passed'] for p in record['targets']['shards'])


def control_authorized(layout):
    if os.environ.get('OPD_MIGRATION_GROUP'):
        migration_verified()
        if four_card_resume():
            assert (os.environ['OPD_MIGRATION_GROUP'],os.environ['CUDA_VISIBLE_DEVICES'])==('B',group_b_devices())
            proof=read(resume_verification_dir()/'result.json')
            assert proof['passed'] and proof['replay']['adam_identical']
            assert all(s['passed'] for s in proof['targets']['shards'])
            return
        assert (os.environ['OPD_MIGRATION_GROUP'],os.environ['CUDA_VISIBLE_DEVICES']) in (
            ('A','0,1,2,3'),('B','4,5,6,7'))
    else:
        assert read(layout.root/'controls_gate.json')['decision']=='RUN_CONTROLS'
