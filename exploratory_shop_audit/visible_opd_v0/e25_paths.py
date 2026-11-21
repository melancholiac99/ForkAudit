"""Paths shared by the Stage E workers and the CPU receipt checks."""
from pathlib import Path
from exploratory_shop_audit.grpo_forced_audit.common import ROOT

RUN = ROOT/'.runtime/runs/shop_visible_opd_e25_20261121'
PUBLIC = Path(__file__).parent/'evidence/section25_20261121'
SERIAL = 'formal_2512'
