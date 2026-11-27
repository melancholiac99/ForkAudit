import json
import os
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from contrast_objective import contrast_kl
from data_collator import SelfDistillationDataCollator


class ContrastCollator(SelfDistillationDataCollator):
    def __call__(self, features):
        if not hasattr(self,'negatives'):
            with open(os.environ['CONTRAST_NEGATIVES']) as f:
                self.negatives=json.load(f)
            review=json.loads(Path(__file__).with_name('proof_review.json').read_text())
            self.positive_proofs={r['problem']:r['proof'] for r in review.values() if r['replace_positive']}
        features=[{**f,'solution':self.positive_proofs[f['problem']]} if f['problem'] in self.positive_proofs else f for f in features]
        pos=super().__call__(features)
        known=[f['problem'] in self.negatives for f in features]
        neg=[self.negatives.get(f['problem']) for f in features]
        swapped=[{**f,'solution':v if v is not None else f['solution']} for f,v in zip(features,neg)]
        other=super().__call__(swapped)
        assert torch.equal(pos['student_prompts'],other['student_prompts'])
        pos['negative_prompts']=other['teacher_prompts']
        pos['contrast_known']=torch.tensor(known)
        pos['has_negative']=torch.tensor([v is not None for v in neg])
        return pos


def teachers(trainer, model, inputs, student_len, teacher_len):
    actor=trainer.accelerator.unwrap_model(model)
    assert trainer.fixed_teacher and hasattr(actor,'disable_adapter') and inputs['contrast_known'].all()
    with torch.no_grad(), actor.disable_adapter():
        base=actor.get_base_model()
        def forward(ids, mask, length):
            return base(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,length-1:-1,:].contiguous()
        positive=forward(inputs['teacher_input_ids'],inputs['teacher_attention_mask'],teacher_len)
        neg_prompt=inputs['negative_prompts']
        ids=torch.cat((neg_prompt,inputs['student_input_ids'][:,student_len:]),dim=1)
        mask=(ids!=trainer.processing_class.pad_token_id).long()
        negative=forward(ids,mask,neg_prompt.shape[1])
    return positive,negative


def loss(student, positive, negative, labels, has_negative, step, output):
    def subtotal(s,p,n):
        mask=torch.ones(s.shape[0],dtype=torch.bool,device=s.device)
        losses,_,mass=contrast_kl(s,p,n,mask)
        return losses.sum(),mass.sum()
    eligible=labels!=-100
    valid=(eligible & has_negative[:,None]).nonzero()
    # A connected zero participates in DDP even for a fully masked microbatch.
    value=student[:0].sum(dtype=torch.float32)
    mass=student.new_zeros((),dtype=torch.float64)
    if len(valid):
        for block in valid.split(48):
            i,j=block.unbind(1)
            part,clipped=checkpoint(subtotal,student[i,j],positive[i,j],negative[i,j],use_reentrant=False)
            value=value+part
            mass=mass+clipped.detach().double()
    path=Path(output)/('supervision-rank-'+os.environ['RANK']+'.jsonl')
    with path.open('a') as f:
        print(json.dumps(dict(step=step,eligible=int(eligible.sum()),supervised=len(valid),clipped_mass_sum=mass.item())),file=f)
    return value/max(1,len(valid))
