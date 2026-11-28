"""Compare every vocabulary logit on every tested answer position in singleton and mixed-length batches."""
import os
import sys
import tempfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from setup_data import HERE,WORK,MODEL,read,save
sys.path.insert(0,str(WORK/'source'))
from repair_runtime import PairCollator,positions

@torch.inference_mode()
def check():
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,attn_implementation='eager').eval()
    rows=[dict(problem='Find x.',solution='x is one.'),dict(problem='Find the second prime number in the given list of positive integers.',solution='Three.')]
    mapping={rows[0]['problem']:['The answer is one.','It is two.'],rows[1]['problem']:['Checking each candidate in sequence gives three.','I computed the number four incorrectly.']}
    suffix=tok.encode(' Final Answer: 3',add_special_tokens=False)+[model.generation_config.eos_token_id[0]]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'pairs.json'
        save(path,mapping)
        os.environ['REPAIR_PAIRS']=str(path)
        collate=PairCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
        batch=collate(rows)
        result={}
        for key in ('student_prompts','teacher_prompts','negative_prompts'):
            def forward(prompt):
                width=prompt.shape[1]
                ids=torch.cat((prompt,torch.tensor([suffix]*len(prompt))),dim=1)
                mask=ids.ne(tok.pad_token_id).long()
                return model(ids,attention_mask=mask,position_ids=positions(mask),use_cache=False).logits[:,width-1:-1]
            together=forward(batch[key])
            assert batch[key][:,0].eq(tok.pad_token_id).any()
            largest=0.
            for i,r in enumerate(rows):
                single=forward(collate([r])[key])[0]
                largest=max(largest,(single-together[i]).abs().max().item())
                torch.testing.assert_close(single,together[i],atol=.0002,rtol=.00002)
            result[key]=largest
    save(HERE/'padding_test.json',dict(max_abs=result,dtype='float32',model='training base',atol=.0002,rtol=.00002,positions=len(suffix)))
if __name__=='__main__':check()
