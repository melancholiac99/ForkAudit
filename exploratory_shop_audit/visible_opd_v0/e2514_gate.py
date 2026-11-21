"""Independent authorization path for the registered eight-card migration."""
import os
from .e25_paths import RUN
from exploratory_shop_audit.grpo_forced_audit.common import read


def migration_verified():
    for group in ('A', 'B'):
        record = read(RUN/'migration_2514'/group/'result.json')
        assert record['passed'] and record['targets']['passed'] and record['replay']['passed']
        assert all(p['mismatches'] == 0 for p in record['probes'])
        assert all(p['passed'] for p in record['targets']['shards'])


def control_authorized(layout):
    if os.environ.get('OPD_MIGRATION_GROUP'):
        migration_verified()
        assert (os.environ['OPD_MIGRATION_GROUP'],os.environ['CUDA_VISIBLE_DEVICES']) in (
            ('A','0,1,2,3'),('B','4,5,6,7'))
    else:
        assert read(layout.root/'controls_gate.json')['decision']=='RUN_CONTROLS'
