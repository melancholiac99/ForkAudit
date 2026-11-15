"""Complete 35 x 2 replacement S/F, reusing frozen candidates and replay values."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_file, digest_value, write_once
from exploratory_shop_audit.runtime import HERE, MODEL, Model, append, read_rows, utc
from exploratory_shop_audit.judge_prompt import build_choice_messages, CHOICE_SYSTEM
from exploratory_shop_audit.review_recompute import verify_parent


def main():
    p = argparse.ArgumentParser(); p.add_argument('--revision', type=Path, required=True); args = p.parse_args()
    revision = args.revision.resolve(); parent = verify_parent(revision)
    if not (revision / 'offline_metrics.json').exists(): raise RuntimeError('Offline diagnostics must finish first')
    probes = json.loads((revision / 'early_probe_metrics.json').read_text())
    if set(probes) != {'after_first_search', 'after_first_open'}: raise RuntimeError('Early probes incomplete')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    points = {r['state_id']: r for r in json.loads((parent / 'state_manifest.json').read_text())['points']}
    defined = {r['state_id'] for r in read_rows(parent / 'formal_forks/state_results.jsonl') if r['defined']}
    assert len(defined) == 35
    candidates = {(r['state_id'], r['candidate']): r for r in read_rows(parent / 'formal_forks/candidates.jsonl')}
    original_requests = {(r['state_id'], r['condition']): r for r in read_rows(parent / 'judge_requests.jsonl')}
    requests = []
    for sid in points:
        if sid not in defined: continue
        for condition in ['S', 'F']:
            old = original_requests[(sid, condition)]
            arms = [candidates[(sid, cid)] for cid in old['order']]
            for c in arms:
                expected = '该动作终止回合。' if c['terminal'] else c['one_step']['state']['observation']
                assert c['clean_feedback'] == expected
            messages = build_choice_messages(points[sid]['input']['messages'], arms, condition)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            assert len(ids) + 256 <= 32768
            assert messages[0] == {'role': 'system', 'content': CHOICE_SYSTEM}
            requests.append({'state_id': sid, 'condition': condition, 'page': points[sid]['page_type'],
                'messages': messages, 'token_ids': ids, 'seed': old['seed'], 'temperature': 0, 'max_tokens': 256,
                'order': old['order'], 'prompt_version': 'dedicated-choice-v2',
                'old_request_sha256': digest_value(old), 'original_context_sha256': digest_value(points[sid]['input']['messages']),
                'candidate_feedback_sha256': digest_value([{'candidate': c['candidate'], 'action': c['action'], 'feedback': c['clean_feedback']} for c in arms])})
    assert len(requests) == 70
    encoded = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests)
    path = revision / 'judge_requests.jsonl'
    if path.exists(): assert path.read_text() == encoded
    else: path.write_text(encoded)
    write_once(revision / 'judge_freeze.json', {'requests': 70, 'states': 35, 'model': str(MODEL),
        'request_sha256': digest_file(path), 'protocol_sha256': digest_file(revision / 'protocol.md'),
        'source_hashes': {name: digest_file(HERE / name) for name in ['judge_prompt.py', 'review_judge.py', 'gpu_worker.py', 'runtime.py']},
        'parameters': {'temperature': 0, 'top_p': 1, 'max_new_tokens': 256, 'max_model_len': 32768},
        'original_episodes_or_replays_regenerated': False})
    completed = {(r['state_id'], r['condition']) for r in read_rows(revision / 'judge_outputs.jsonl')}
    pending = [r for r in requests if (r['state_id'], r['condition']) not in completed]
    if not pending: return
    inherited = sum(r['gpu_hours'] for r in read_rows(parent / 'gpu_sessions.jsonl'))
    model = Model(revision, max_model_len=32768)
    try:
        for start in range(0, len(pending), 8):
            batch = pending[start:start + 8]
            for req, generation in zip(batch, model.generate(batch)):
                match = re.fullmatch(r'\s*([1-4])\s*', generation['text'])
                index = int(match.group(1)) - 1 if match else -1
                selected = req['order'][index] if 0 <= index < len(req['order']) else None
                append(revision / 'judge_outputs.jsonl', {'state_id': req['state_id'], 'condition': req['condition'],
                    'selected': selected, 'generation': generation, 'error': None if selected is not None else 'PARSE_FAILURE'})
            print(json.dumps({'completed': min(start + 8, len(pending)), 'total': len(pending), 'cumulative_gpu_hours': model.hours()}), flush=True)
    finally:
        model.close()
        total = sum(r['gpu_hours'] for r in read_rows(revision / 'gpu_sessions.jsonl'))
        (revision / 'costs.json').write_text(json.dumps({'inherited_gpu_hours': inherited,
            'additional_gpu_hours': total - inherited, 'total_gpu_hours': total, 'limit': 16}, indent=2) + '\n')
    verify_parent(revision)


if __name__ == '__main__': main()
