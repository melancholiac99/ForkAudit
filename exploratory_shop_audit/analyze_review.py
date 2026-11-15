"""Single CPU-only entry for reproducing the post-review analysis."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(); p.add_argument('--revision', type=Path, required=True); args = p.parse_args()
    for script in ['review_recompute.py', 'review_early_probe.py', 'review_report.py']:
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)), '--revision', str(args.revision.resolve())], check=True)


if __name__ == '__main__': main()
