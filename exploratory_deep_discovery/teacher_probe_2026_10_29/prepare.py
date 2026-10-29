"""Prepare teacher inputs without loading any counterfactual outcome rates."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'
sys.path.insert(0, str(ROOT/'scripts'))
import common

H_INSTRUCTION = """You are reconsidering the marked decision in an ALFWorld episode.
Use the retrospective evidence to choose an action that improves the original
student's chance of completing its task. You may change the recorded action;
do not merely reproduce it because it appears in the transcript.
The decision-time prompt below is the student's actual information. After the
action, the same student continues with its normal two-observation memory and
the original 50-step total budget. Retrospective evidence is for your assessment;
it is not an extra hint automatically inserted into the student's continuation.
The recorded future is one factual branch, not the outcome of every alternative.
Continue the assistant response at the marked decision in its existing format.
"""
BRANCH_INSTRUCTION = """The following are one-step environment queries from the same
decision state. Each response belongs only to its labeled action. The student
receives the response of the action actually selected, not all five responses.
These records do not contain terminal outcomes of the alternative branches.
"""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    old = ROOT/'exploratory_deep_discovery/opd_opsd_2026_10_29/prepared_v1'
    sources = {}
    def read(path, binary=False):
        b = path.read_bytes(); sources[str(path.relative_to(ROOT))] = sha(b)
        return b if binary else json.loads(b)
    contexts_path = old/'contexts/student_visible.jsonl'
    contexts = [json.loads(x) for x in read(contexts_path, True).decode().splitlines()]
    dev_path = ROOT/'exploratory_deep_discovery/regime_mechanism_2026_10_28/design.json'
    dev_ids = {p['id'] for p in read(dev_path)['points']}
    assert len(contexts) == 144 and len(dev_ids) == 24
    request_path = OUT/'requests.jsonl'
    if request_path.exists():
        raise FileExistsError('Do not overwrite a prepared experiment.')
    requests, audit, tasks = [], [], {'main':set(), 'development':set()}
    rollouts = {}
    for ctx in contexts:
        pid = ctx['point_id']; tid, turn_text = pid.split('_t'); turn = int(turn_text)
        cohort = 'development' if pid in dev_ids else 'main'
        if tid not in rollouts:
            rollouts[tid] = read(ROOT/f'data/rollout/{tid}.json')
        tr = rollouts[tid]; step = tr['steps'][turn-1]
        tasks[cohort].add(tr['task_id'])
        assert step['turn'] == turn and tr['traj_id'] == tid
        prompt = read(ROOT/step['prompt_path'], True).decode()
        assert prompt == ctx['student_prompt'] and sha(prompt.encode()) == ctx['student_prompt_sha256']
        peek = read(ROOT/f'data/peek_alts/{tid}/t{turn:02d}.json')
        peeks = [x for x in peek['peeks'] if x['role']=='factual'] + sorted(
            [x for x in peek['peeks'] if x['role']=='alt'], key=lambda x:x['alt_index'])
        candidates = ctx['candidates']
        assert len(peeks) == 5 and [p['action'] for p in peeks] == [c['action'] for c in candidates]
        assert peeks[0]['obs'] == step['obs_after'] and candidates[0]['action'] == step['action']
        assert peek['checks']['factual_obs_match'] and peek['checks']['refeed_div_total'] == 0
        assert peek['checks']['repeat_all_match']
        response = step['response']; thought_end = response.rfind('</think>') + len('</think>')
        assert thought_end >= len('</think>')
        tail = response[thought_end:]
        matches = list(common._TAGGED.finditer(tail)) or list(common._UNCLOSED.finditer(tail))
        assert matches and common.parse_action(response)[0] == step['action']
        chosen_span = matches[-1]
        opening = re.match(common._OPEN, chosen_span.group(0), re.I).group(0)
        response_prefix = response[:thought_end+chosen_span.start()] + opening
        inside = chosen_span.group(1)
        leading = inside[:len(inside)-len(inside.lstrip())]
        trailing = inside[len(inside.rstrip()):]
        # Closed actions retain observed syntax. Truncated actions receive a
        # common matching closing delimiter, disclosed in the audit.
        cm = re.search(common._CLOSE, chosen_span.group(0), re.I)
        closing = cm.group(0) if cm else ('[/action]' if opening.startswith('[') else '</action>')
        episode = dict(task=tr['task'], initial_observation=tr['initial_obs'],
            steps=[dict(turn=s['turn'],response=s['response'],executed_action=s['action'],
                        observation_after=s['obs_after']) for s in tr['steps']],
            final_reward=int(tr['won']))
        future = json.dumps(episode, ensure_ascii=False, separators=(',',':'))
        h_text = (H_INSTRUCTION + f'\nMarked decision: turn {turn}.\n'
            + '<decision_time_prompt>\n'+prompt+'\n</decision_time_prompt>\n'
            + '<factual_episode>\n'+future+'\n</factual_episode>')
        seed = int(sha(f'teacher-probe-v1:{pid}'.encode())[:16],16)
        rng = np.random.default_rng(seed)
        order = rng.permutation(5).tolist()
        shift = 1+seed%3
        mapping = [0]+[1+(i-1+shift)%4 for i in range(1,5)]
        def branches(m):
            return [dict(action=candidates[i]['action'],response=peeks[m[i]]['obs']) for i in order]
        aligned, shuffled = branches(list(range(5))), branches(mapping)
        assert sorted(x['response'] for x in aligned) == sorted(x['response'] for x in shuffled)
        assert mapping[0] == 0 and all(mapping[i] != i for i in range(1,5))
        texts = {'S':prompt, 'H':h_text,
            'A':h_text+'\n'+BRANCH_INSTRUCTION+'\n'+json.dumps(aligned,ensure_ascii=False),
            'X':h_text+'\n'+BRANCH_INSTRUCTION+'\n'+json.dumps(shuffled,ensure_ascii=False)}
        for condition, text in texts.items():
            rendered = tokenizer.apply_chat_template([dict(role='user',content=text)],
                                                     tokenize=False,add_generation_prompt=True)
            prefix = rendered + response_prefix
            preids = tokenizer.encode(prefix, add_special_tokens=False)
            fullids = [tokenizer.encode(prefix+leading+c['action']+trailing+closing+tokenizer.eos_token,
                                        add_special_tokens=False) for c in candidates]
            start = len(preids)
            for ids in fullids:
                i = 0
                while i < min(len(preids),len(ids)) and preids[i]==ids[i]: i+=1
                start = min(start,i)
            assert len(preids)-start <= 3
            assert all(ids[:start]==fullids[0][:start] and ids[-1]==tokenizer.eos_token_id for ids in fullids)
            assert max(map(len,fullids)) < 32768
            job = dict(request_id=f'{pid}__{condition}',point_id=pid,cohort=cohort,
                condition=condition,task_id=tr['task_id'],turn=turn,
                prefix_text=prefix,prefix_ids=fullids[0][:start],
                candidates=[dict(**c,suffix_ids=ids[start:]) for c,ids in zip(candidates,fullids)],
                token_boundary_backoff=len(preids)-start,
                episode_sha256=sha(future.encode()) if condition!='S' else None)
            job['request_sha256'] = sha(json.dumps(job,sort_keys=True,ensure_ascii=False).encode())
            requests.append(job)
        audit.append(dict(point_id=pid,cohort=cohort,task_id=tr['task_id'],
            response_format=step['fmt'],synthesized_closing=cm is None,
            actual_thought_sha256=sha(response_prefix.encode()),display_order=order,
            alternative_mapping=mapping,all_candidates_legal=all(c['legal_before'] for c in candidates)))
    assert not tasks['main'].intersection(tasks['development'])
    with request_path.open('w') as f:
        for row in requests:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for name,obj in [('input_manifest.json',sources),('point_audit.json',audit)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    protocol = OUT/'PROTOCOL.md'
    summary = dict(status='prepared; no model scoring yet',points=len(contexts),contexts=len(requests),
        candidate_sequences=len(requests)*5,conditions=['S','H','A','X'],
        prefix_tokens_min=min(len(r['prefix_ids']) for r in requests),
        prefix_tokens_max=max(len(r['prefix_ids']) for r in requests),
        prefix_tokens_sum=sum(len(r['prefix_ids']) for r in requests),
        main_tasks=len(tasks['main']),development_tasks=len(tasks['development']),
        main_all_legal=sum(a['all_candidates_legal'] and a['cohort']=='main' for a in audit),
        protocol_sha256=sha(protocol.read_bytes()),request_file_sha256=sha(request_path.read_bytes()),
        preparation_code_sha256=sha(Path(__file__).read_bytes()))
    (OUT/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
