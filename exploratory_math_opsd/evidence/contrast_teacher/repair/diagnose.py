"""Replay each eligible answer using the original four-example prompt widths."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from contexts import MODEL, ROOT, WORK, read, save
sys.path.insert(0, str(WORK / 'source'))
from data_collator import SelfDistillationDataCollator

def select():
    jobs = read(WORK / 'manifest.json')[:800]
    negatives = read(WORK / 'negative.json')
    samples = {v['idx']: v['answers'] for path in WORK.glob('batch-*.json') for v in read(path)}
    positives = {j['idx']: next((v['text'] for v in samples[j['idx']] if v['correct'] and r'\boxed{' in v['text']), None) for j in jobs}
    index = {j['problem']: i for i, j in enumerate(jobs)}
    rows = []
    for path in (ROOT / 'run/R-contrast/generations').glob('*.json'):
        for g in read(path)['generations']:
            if g['step'] >= 25:
                continue
            question = g['prompt'].split('Problem: ', 1)[1].rsplit('\n\nPlease reason', 1)[0]
            i = index[question]
            assert i // 32 == g['step']
            if not negatives[question] or not positives[jobs[i]['idx']]:
                continue
            text = g['completion'].replace('<|endoftext|>', '')
            if text.endswith('<|im_end|>'):
                rows.append((i, text))
    assert len({i for i, _ in rows}) == len(rows) and rows
    return jobs, negatives, samples, positives, sorted(rows)

@torch.inference_mode()
def run(lane):
    jobs, negatives, samples, positives, rows = select()
    tok = AutoTokenizer.from_pretrained(MODEL)
    col = SelfDistillationDataCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    out = []
    for i, text in rows[lane::4]:
        batch = jobs[i//4*4:i//4*4+4]
        prompts = {}
        for name in ('ref','B','negative'):
            features = []
            for j in batch:
                sol = j['solution'] if name=='ref' else (positives[j['idx']] if name=='B' else negatives[j['problem']]) or j['solution']
                features.append(dict(j,solution=sol))
            padded = col(features)['teacher_prompts']
            ids = padded[i%4].tolist()
            prompts[name] = ([x for x in ids if x!=tok.pad_token_id],len(ids))
        answers = [('train',text)] + [('B',v['text']) for v in samples[jobs[i]['idx']]]
        for source, ans in answers:
            enc = tok(ans,add_special_tokens=False,return_offsets_mapping=True)
            y = enc.input_ids
            if not y:
                continue
            masks = {'all':np.ones(len(y),bool),'EOS':np.isin(y,model.generation_config.eos_token_id)}
            for key,pattern in [('final',r'\bfinal\s+answer\b'),('boxed',r'\\boxed')]:
                spans = [m.span() for m in re.finditer(pattern,ans,re.I)]
                masks[key] = np.array([any(a<hi and b>lo for lo,hi in spans) for a,b in enc.offset_mapping])
            for layout in ('right','compact'):
                scores = {}
                for name,(raw,width) in prompts.items():
                    gap = [tok.pad_token_id]*(width-len(raw))
                    ids = torch.tensor([(raw+gap if layout=='right' else gap+raw)+y],device='cuda')
                    mask = ids.ne(tok.pad_token_id).long()
                    kw = {} if layout=='right' else {'position_ids':(mask.cumsum(-1)-1).clamp_min(0)}
                    hidden = model.model(input_ids=ids,attention_mask=mask,use_cache=False,**kw).last_hidden_state[0,width-1:-1]
                    vals = []
                    for start in range(0,len(y),64):
                        logits = model.lm_head(hidden[start:start+64]).float()/1.1
                        target = ids[0,width+start:width+start+len(logits)]
                        vals.append((logits.gather(1,target[:,None])[:,0]-logits.logsumexp(-1)).cpu())
                    scores[name] = torch.cat(vals).numpy()
                for name in ('ref','B'):
                    diff = .5*(scores[name]-scores['negative'])
                    for key,m in masks.items():
                        if m.any():
                            out.append([i,source,layout,name,key,int(m.sum()),float(diff[m].sum(dtype=np.float64))])
    save(HERE/f'lane-{lane}.json',out)

def combine():
    from collections import defaultdict
    groups = defaultdict(lambda:defaultdict(lambda:np.zeros(2)))
    for lane in range(4):
        for i,source,layout,name,key,n,total in read(HERE/f'lane-{lane}.json'):
            groups['/'.join([source,layout,name,key])][i] += [n,total]
    report = {}
    for key, by in groups.items():
        a = np.array([by[i] for i in sorted(by)])
        rng = np.random.default_rng(20261124)
        boot = a[rng.integers(len(a),size=(10000,len(a)))].sum(1)
        report[key] = dict(questions=len(a),tokens=int(a[:,0].sum()),mean=a[:,1].sum()/a[:,0].sum(),ci=np.quantile(boot[:,1]/boot[:,0],[.025,.975]).tolist())
    save(HERE/'diagnostic.json',dict(report=report,exclude_control=report['train/compact/B/EOS']['ci'][1]<-.5,
         note='B archives retain decoded text only; unrecorded terminal EOS is omitted. Widths retain original training peers; for missing B positives of peers, use their reference solution.'))
if __name__=='__main__':
    combine() if sys.argv[1]=='combine' else run(int(sys.argv[1]))
