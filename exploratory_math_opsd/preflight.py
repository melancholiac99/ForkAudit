"""Run the registered single-question prompt and token-mask check on CPU."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer


def digest(x):
    return sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def run(root):
    sys.path.insert(0,str(root/'source'))
    from data_collator import SelfDistillationDataCollator, assemble_distillation_inputs
    paths=sorted((root/'datasets/train/data').glob('*.parquet'))
    source=next(pq.ParquetFile(paths[0]).iter_batches(batch_size=1,columns=['problem','solution'])).to_pylist()[0]
    alternative=dict(source,solution='Alternative reference text for the registered masking check.')
    tok=AutoTokenizer.from_pretrained(root/'models/Qwen3-1.7B',local_files_only=True)
    texts={}
    sequences={}
    expected_user=f"Problem: {source['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    reference_t0=tok.apply_chat_template([{'role':'user','content':expected_user}],tokenize=False,
                                         add_generation_prompt=True,enable_thinking=True)
    answer=tok.encode(r'The answer is \boxed{17}.',add_special_tokens=False)
    response=torch.tensor([answer+[tok.eos_token_id,tok.pad_token_id]])
    for arm in ('R','T0'):
        collate=SelfDistillationDataCollator(tok,max_length=20000,reason_first=False,
                                             student_thinking=False,teacher_thinking=True,question_only_teacher=arm=='T0')
        item=collate([source])
        alt=collate([alternative])
        student=item['student_prompts']
        teacher=item['teacher_prompts']
        assert torch.equal(student,alt['student_prompts'])
        assert torch.equal(teacher,alt['teacher_prompts']) == (arm=='T0')
        if arm=='T0':
            assert torch.equal(teacher,tok(reference_t0,return_tensors='pt').input_ids)
        generated=torch.cat([student,response],dim=1)
        assemble_distillation_inputs(item,generated,generated.ne(tok.pad_token_id).long(),tok.pad_token_id)
        sp,tp=item['student_prompt_length'],item['teacher_prompt_length']
        selected=item['labels'][:,sp:].ne(-100)
        target=response[response.ne(tok.pad_token_id)]
        assert item['labels'][:,:sp].eq(-100).all() and torch.equal(selected,response.ne(tok.pad_token_id))
        # Official compute_loss pairs logits at [prompt_length-1:-1] with these next tokens.
        for name,offset in [('student_input_ids',sp),('teacher_input_ids',tp)]:
            full=item[name]
            assert full[:,offset-1:-1].shape == selected.shape
            assert torch.equal(full[:,offset:][selected],target)
        sequences[arm]=(digest(student.tolist()),digest(target.tolist()))
        texts[arm]={'student_sha256':sequences[arm][0],'teacher_sha256':digest(teacher.tolist()),
                    'changed_reference_teacher_sha256':digest(alt['teacher_prompts'].tolist()),
                    'supervised_tokens':len(target),'supervised_sha256':sequences[arm][1]}
    assert sequences['R']==sequences['T0']
    sizes={k:sum(pq.ParquetFile(f).metadata.num_rows for f in sorted((root/'datasets'/k/'data').glob('*.parquet')))
           for k in ('train','aime24','aime25','hmmt25')}
    assert all(sizes[k]==30 for k in ('aime24','aime25','hmmt25')) and sizes['train']>0
    return dict(passed=True,device='cpu',dataset_rows=sizes,problem_sha256=digest(source['problem']),arms=texts,
                upstream_revision=subprocess.check_output(['git','-C',str(root/'source'),'rev-parse','HEAD'],text=True).strip(),
                patch_sha256=sha256(Path(__file__).with_name('upstream.patch').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=json.dumps(run(args.root.resolve()),indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    print(result)
