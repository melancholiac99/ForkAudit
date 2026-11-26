import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import measure
measure.STEPS=(100,)
for arm in ('R-clip-boundary','R-clip-interior'):
    measure.score('fixed',arm,int(sys.argv[1]))
