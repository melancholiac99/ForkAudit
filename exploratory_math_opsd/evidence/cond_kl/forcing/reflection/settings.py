from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common_io import ROOT, RAW as FORCING_RAW, SEED, read, save
RAW=FORCING_RAW/'reflection'
RAW.mkdir(exist_ok=True)
MODELS=('B','T0@100','R@100','R-noclip@100','R-cond@100','P@100')
