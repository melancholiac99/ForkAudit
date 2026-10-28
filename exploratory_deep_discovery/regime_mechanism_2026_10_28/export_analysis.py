"""Export a compact, hash-linked research bundle without duplicating all rollouts."""
import argparse
import shutil
from pathlib import Path
from protocol import HERE, ROOT, read, write, file_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path)
    args = ap.parse_args()
    integrity = read(args.run_dir / 'integrity.json')
    assert integrity['status'] == 'complete'
    target = HERE / 'results'
    target.mkdir(exist_ok=True)
    names = ['summary.json', 'summary.md', 'summary_points.csv', 'summary_arms.csv',
        'summary_blocks.csv', 'summary_profiles.csv', 'summary_figure.png', 'summary_figure.svg',
        'summary_figure_all_profiles.png', 'summary_figure_all_profiles.svg',
        'behavior_summary.json', 'behavior_records.csv', 'behavior_cells.csv',
        'mechanism_summary.json', 'mechanism_points.csv', 'mechanism_cases.png', 'mechanism_cases.svg',
        'integrity.json', 'progress.json', 'results_manifest.json', 'run_manifest.json',
        'launch_environment.json']
    for name in names:
        shutil.copyfile(args.run_dir / name, target / name)
    # Illustrative traces chosen after viewing complete cases. Not a random sample.
    examples = [('s005_t31', 'a1', 'native'), ('s005_t31', 'a1', 'completed'),
                ('s010_t29', 'a3', 'native'), ('s010_t29', 'a3', 'completed'),
                ('s044_t18', 'a1', 'native'), ('s044_t18', 'a1', 'completed'),
                ('s003_t18', 'a0', 'completed')]
    cases = target / 'cases'
    cases.mkdir(exist_ok=True)
    for pid, arm, condition in examples:
        name = f'{pid}__{arm}__{condition}__r000.json'
        shutil.copyfile(args.run_dir / 'results' / name, cases / name)
    outputs = [p for p in target.rglob('*') if p.is_file() and p.name != 'bundle_manifest.json']
    code = sorted(HERE.glob('*.py')) + [HERE / 'run.sh', HERE / 'README.md']
    report = ROOT / 'reports/regime_mechanism_2026-10-28.md'
    write(target / 'bundle_manifest.json', dict(
        status='complete exploratory analysis bundle', run_fingerprint=integrity['run_fingerprint'],
        source_run=str(args.run_dir.resolve().relative_to(ROOT)),
        outputs={str(p.relative_to(target)): file_hash(p) for p in sorted(outputs)},
        analysis_code={str(p.relative_to(ROOT)): file_hash(p) for p in code},
        report={str(report.relative_to(ROOT)): file_hash(report)},
        case_selection='Seven post-hoc illustrative raw records, repeat zero; no representative-frequency claim.',
        raw_rollouts='All 3840 raw records remain in source_run/results; hashes in results_manifest.json.',
        reuse='1024 explicitly tagged existing continuations; 2816 new formal continuations.'))
    print(f'Exported {len(outputs)} artifacts to {target}')


if __name__ == '__main__':
    main()
