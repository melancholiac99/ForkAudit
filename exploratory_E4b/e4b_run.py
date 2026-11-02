"""Run E4's frozen training / evaluation code for E4b arms, with only the data directory redirected to exploratory_E4b/data.

  python e4b_run.py train --arm Azero --seed 0 --lr 5e-06        # -> exploratory_E4/e4_06_train.py, unchanged
  python e4b_run.py eval  --split valid_unseen --adapters Azero_s0 Agap_s0   # -> e4_07_eval.py (base included unless --no_base)

E4's games.json, sampling seeds (common random numbers), templates, LoRA/DPO settings and the
run/training_implementation_change.json provenance check all stay exactly as in E4.
"""
import importlib
import sys

import e4b_common as B
from e4b_common import E

MODULES = {"train": "e4_06_train", "eval": "e4_07_eval"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        sys.exit(__doc__)
    E.DATA = B.DATA
    E.ARMS_TRAINED = list(B.ARMS)
    if sys.argv[1] == "train":
        lr = sys.argv[sys.argv.index("--lr") + 1] if "--lr" in sys.argv else None
        assert lr is not None and float(lr) == B.LR, f"E4b uses E4's selected learning rate {B.LR}; got {lr}"
    mod = importlib.import_module(MODULES[sys.argv[1]])
    sys.argv = [mod.__file__] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
