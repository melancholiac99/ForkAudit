"""Export completed interim aggregates; retain exact trajectories on the server."""
import argparse
from pathlib import Path
from .common import DEFAULT_RUN, HERE, read, read_rows, save, digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output
    source = run / 'priority_evaluation'
    assert read(source / 'status.json')['status'] == 'complete'
    assert not output.exists(), 'Preserve previously exported reports'
    result = read(source / 'metrics.json')
    for summary in result['evaluation'].values():
        for group in summary.values():
            if isinstance(group, dict):
                group.pop('records', None)
    inputs = [source / 'metrics.json', source / 'request.json', source / 'events.jsonl',
              run / 'splits.json', run / 'full_configuration_maxima.jsonl']
    for label in ['base', result['request']['label']]:
        for decode, count in [('greedy', 200), ('sample', 800)]:
            folder = run / 'evaluation' / label / decode
            episodes = read_rows(folder / 'episodes.jsonl')
            assert len(episodes) == len({r['trajectory_id'] for r in episodes}) == count
            inputs += [folder / 'episodes.jsonl', folder / 'steps.jsonl']
    output.mkdir(parents=True)
    save(output / 'metrics.json', result)
    save(output / 'provenance.json', {'run_id': run.name, 'input_sha256': {str(p.relative_to(run)): digest_file(p) for p in inputs},
         'generator_sha256': digest_file(Path(__file__)), 'completion': read(source / 'status.json'),
         'raw_records': 'Retained on server; aggregates are exported without per-trajectory records.'})
    (output / 'REPORT.md').write_bytes((source / 'REPORT.md').read_bytes())
    with (output / 'REPORT.md').open('a') as stream:
        stream.write('\n复算：先用 priority_evaluation.analyze 对保存数据重算，再运行 export_interim_report --output 新目录。来源校验和见 provenance.json。\n')


if __name__ == '__main__':
    main()
