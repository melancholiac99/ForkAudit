import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[2]),str(HERE.parent/'clip_positions')]
import evaluate_cells as cells
cells.ARMS=('R-condA',)
cells.HERE=HERE
cells.main()
