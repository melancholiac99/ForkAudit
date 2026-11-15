"""Execute previously unexecuted S/F requests whose complete context needs >8192.

The policy keeps its frozen 8192-token input cap. This same-weight judge uses
the checkpoint's native 32768 context, preserving identical S/F prior context.
Already generated choices (including parse failures) are never regenerated.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.runtime import Model, append, read_rows, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    outputs = {(r['state_id'], r['condition']): r for r in read_rows(run / 'judge_outputs.jsonl')}
    requests = {(r['state_id'], r['condition']): r for r in read_rows(run / 'judge_requests.jsonl')}
    pending = [requests[key] for key, out in outputs.items() if out.get('error') == 'INPUT_OVERFLOW' and 'generation' not in out]
    if not pending:
        print('No unexecuted judge requests')
        return
    if any(len(r['token_ids']) + 256 > 32768 for r in pending):
        raise RuntimeError('Complete judge input exceeds checkpoint native context; retain explicit missingness')
    model = Model(run, max_model_len=32768)
    try:
        for start in range(0, len(pending), 8):
            group = pending[start:start + 8]
            generated = model.generate(group)
            for req, out in zip(group, generated):
                match = re.fullmatch(r'\s*([1-4])\s*', out['text'])
                index = int(match.group(1)) - 1 if match else -1
                selected = req['order'][index] if 0 <= index < len(req['order']) else None
                append(run / 'judge_outputs.jsonl', {'state_id': req['state_id'], 'condition': req['condition'],
                    'selected': selected, 'generation': out, 'error': None if selected is not None else 'PARSE_FAILURE',
                    'supersedes_unexecuted_capacity_check': True, 'model_context_capacity': 32768})
        append(run / 'progress.jsonl', {'time': utc(), 'phase': 'complete_judge_contexts',
               'requests': len(pending), 'gpu_hours': model.hours()})
    finally:
        model.close()
        hours = sum(r['gpu_hours'] for r in read_rows(run / 'gpu_sessions.jsonl'))
        (run / 'costs.json').write_text(json.dumps({'gpu_hours': hours, 'gpu_count': 1, 'cap_gpu_hours': 16,
            'collection_complete': (run / 'collection_complete.json').exists()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
