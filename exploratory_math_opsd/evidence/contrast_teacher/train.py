import os
import shutil
import subprocess
from contexts import HERE,ROOT,WORK,MODEL,read,save,sha
from launch import validate_checkpoint

ARM='R-contrast'


def replace(code, old, new):
    assert code.count(old)==1,old
    return code.replace(old,new)


def prepare():
    src=WORK/'source'
    shutil.copytree(ROOT/'source',src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
    shutil.copy(HERE/'proof_review.json',src/'proof_review.json')
    shutil.copy(HERE/'objective.py',src/'contrast_objective.py')
    shutil.copy(HERE/'training_parts.py',src/'training_parts.py')
    path=src/'opsd_trainer.py'
    code=replace(path.read_text(), '            data_collator = SelfDistillationDataCollator(', '            from training_parts import ContrastCollator\n            data_collator = ContrastCollator(')
    a=code.index('        # === TEACHER FORWARD');b=code.index('            del student_logits_for_loss, teacher_logits_for_loss',a)
    code=code[:a]+'''        from training_parts import teachers, loss as contrast_loss
        assert not self.use_thinking_machines_loss and self.temperature==1.1
        teacher_logits_for_loss, negative = teachers(self, model, inputs, student_prompt_len, teacher_prompt_len)
        loss = contrast_loss(student_logits_for_loss, teacher_logits_for_loss, negative, shifted_labels, inputs["has_negative"])
        del student_logits_for_loss, teacher_logits_for_loss, negative
'''+code[code.index('\n',b)+1:]
    path.write_text(code)
    compile(code,str(path),'exec')
    return src


def main():
    src=prepare()
    neg=WORK/'negative.json'
    assert sha(neg)==read(HERE/'negative_contexts.json')['sha256']
    output=ROOT/'run'/ARM
    (WORK/'checkpoints').mkdir(exist_ok=True)
    if not output.exists():output.symlink_to(WORK/'checkpoints',target_is_directory=True)
    assert output.resolve()==WORK/'checkpoints'
    if not (output/'checkpoint-200/trainer_state.json').exists():
        cmd=['bash','scripts/run_opsd_1b.sh','--stop_after_updates','200','--seed','42','--data_seed','42',
             '--save_steps','50','--save_only_model','True','--local_train_glob',str(ROOT/'datasets/train/data/*.parquet')]
        with (ROOT/'logs/R-contrast-200-train.log').open('a') as f:
            subprocess.run(cmd,cwd=src,check=True,env={**os.environ,'OPSD_ARM':ARM,'OPSD_MODEL':str(MODEL),
                'OPSD_OUTPUT':str(output),'CONTRAST_NEGATIVES':str(neg)},stdout=f,stderr=subprocess.STDOUT)
    hashes={}
    for step in (50,100,150,200):
        ck=output/f'checkpoint-{step}'
        validate_checkpoint(ck,step)
        hashes[str(step)]=sha(ck/'adapter_model.safetensors')
    receipt=dict(arm=ARM,adapters=hashes,seed=42,negative_sha256=sha(neg))
    for p in (output/'complete.json',output/'extended.json',HERE/'training.json'):
        save(p,receipt)


if __name__=='__main__':
    import fcntl
    with (WORK/'half-gpu-handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
