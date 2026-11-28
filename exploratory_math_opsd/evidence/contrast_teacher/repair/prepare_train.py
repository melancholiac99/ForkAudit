"""Apply v2 changes to a separate copy of the audited v1 trainer."""
import os
import shutil
import sys
from pathlib import Path
from setup_data import HERE,WORK,ROOT,MODEL,read,old,save
sys.path.insert(0,str(HERE.parent))
from train import replace
def edit(path,changes):
    code=path.read_text()
    for prev,new in changes:code=replace(code,prev,new)
    compile(code,str(path),'exec')
    path.write_text(code)
def prepare():
    src=WORK/'source'
    shutil.copytree(old.WORK/'source',src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy(HERE/'runtime.py',src/'repair_runtime.py')
    edit(src/'opsd_trainer.py',[
        ('from training_parts import ContrastCollator','from repair_runtime import PairCollator as ContrastCollator'),
        ('attention_mask=inputs["student_attention_mask"],','attention_mask=inputs["student_attention_mask"],\n            position_ids=(inputs["student_attention_mask"].long().cumsum(-1)-1).clamp_min(0),'),
        ('        total_completion_tokens = sum(len(ids) for ids in completion_ids)','''        end_ids=set(self._repair_eos)
        self.repair_counts[0]+=sum(bool(len(seq)) and seq[-1] in end_ids for seq in completion_ids)
        self.repair_counts[1]+=len(completion_ids)
        total_completion_tokens = sum(len(ids) for ids in completion_ids)''')])
    edit(src/'training_parts.py',[('attention_mask=mask,use_cache=False','attention_mask=mask,position_ids=(mask.long().cumsum(-1)-1).clamp_min(0),use_cache=False')])
    edit(src/'opsd_train.py',[('    trainer.train(resume_from_checkpoint=', '''    from repair_runtime import WindowGate
    import json as _json
    trainer._repair_eos=_json.loads(open(os.path.join(model_args.model_name_or_path,'generation_config.json')).read())['eos_token_id']
    trainer.repair_counts=[0,0]
    trainer.add_callback(WindowGate(trainer))
    trainer.train(resume_from_checkpoint=''')])
    edit(src/'data_collator.py',[('labels[i, :actual_prompt_len] = -100','labels[i, :student_prompt_len] = -100')])
    if read(HERE/'diagnostic.json')['exclude_control']:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODEL)
        control=sorted(set(tok.added_tokens_decoder)|set(read(MODEL/'generation_config.json')['eos_token_id']))
        edit(src/'contrast_objective.py',[('        log_target = torch.where',f'''        idx = {control!r}
        free = torch.ones(lp.shape[-1],dtype=torch.bool,device=lp.device)
        free[idx] = False
        adjusted=log_target.clone()
        adjusted[...,free] += lp.detach()[...,free].logsumexp(-1,keepdim=True) - log_target[...,free].logsumexp(-1,keepdim=True)
        adjusted[...,idx]=lp.detach()[...,idx]
        log_target=adjusted
        log_target = torch.where''')])
    output=WORK/'checkpoints'
    output.mkdir(exist_ok=True)
    link=ROOT/'run/R-contrast-v2'
    if not link.exists():link.symlink_to(output,target_is_directory=True)
    assert link.resolve()==output
if __name__=='__main__':prepare()
