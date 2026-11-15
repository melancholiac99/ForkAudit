"""Single offline entry point: saved data -> metrics, probe, figures, report."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for script in ['analyze_saved.py', 'enrich_report.py']:
        subprocess.run([sys.executable, str(here / script), '--run', str(args.run.resolve())], check=True)


if __name__ == '__main__':
    main()
