"""Train only against individually accepted, qualified negative edits."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from check import HERE,DATA,ROOT,read,save
sys.path.insert(0,str(HERE.parent))
from train import replace
from launch import sha
from expand import done,pool
def main():
    assert read(HERE/'signal.json')['gate_pass']
    accepted=done()
    checked=read(DATA/'validated.json')
    verdict=read(DATA/'semantic.json')
    assert set(verdict)=={str(r['idx']) for r in checked if r['qualified']}
    for r in checked:
        if r['qualified'] and verdict[str(r['idx'])]['pass']:
            accepted[r['idx']]=r
    jobs=read(DATA.parent/'contrast/manifest.json')[:3200]
    mapping={j['problem']:None for j in jobs}
    for j in jobs:
        x=accepted.get(j['idx'])
        if x and x['qualified']:
            assert __import__('hashlib').sha256(j['solution'].encode()).hexdigest()==x['reference_sha256']
            mapping[j['problem']]=x['text']
    save(DATA/'negative.json',mapping)
    save(HERE/'coverage.json',dict(training_questions=len(jobs),with_original_negative=len(pool()),qualified=sum(v is not None for v in mapping.values()),negative_sha256=sha(DATA/'negative.json'),drafts_passed_auto=sum(r['qualified'] for r in checked),drafts_passed_review=sum(v['pass'] for v in verdict.values())))
    src=DATA/'source';base=DATA.parent/'contrast/source'
    shutil.copytree(base,src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    # The old collator has a proof-review exception. Positive contexts are verbatim for every row here.
    p=src/'training_parts.py'
    code=p.read_text()
    a=code.index('            review=');b=code.index('        pos=super()',a)
    p.write_text(code[:a]+code[b:])
    shutil.copy(HERE.parent/'repair/runtime.py',src/'repair_runtime.py')
    p=src/'opsd_trainer.py'
    p.write_text(replace(p.read_text(),'        total_completion_tokens = sum(len(ids) for ids in completion_ids)','''        self.repair_counts[0]+=sum(bool(len(seq)) and seq[-1] in self._repair_eos for seq in completion_ids)
        self.repair_counts[1]+=len(completion_ids)
        total_completion_tokens = sum(len(ids) for ids in completion_ids)'''))
    p=src/'opsd_train.py'
    p.write_text(replace(p.read_text(),'    trainer.train(resume_from_checkpoint=', '''    from repair_runtime import WindowGate
    trainer._repair_eos=set(json.loads((Path(model_args.model_name_or_path)/'generation_config.json').read_text())['eos_token_id'])
    trainer.repair_counts=[0,0]
    trainer.add_callback(WindowGate(trainer))
    trainer.train(resume_from_checkpoint='''))
    output=DATA/'checkpoints'
    output.mkdir(exist_ok=True)
    arm='R-contrast-v2-matched'
    link=ROOT/'run'/arm
    if not link.exists():link.symlink_to(output,target_is_directory=True)
    assert link.resolve()==output and not list(output.glob('end_windows*'))
    with (DATA/'training.log').open('a') as log:
        subprocess.run(['bash','scripts/run_opsd_1b.sh','--stop_after_updates','100','--seed','42','--data_seed','42','--save_steps','25','--save_only_model','True','--local_train_glob',str(ROOT/'datasets/train/data/*.parquet')],cwd=src,env={**os.environ,'OPSD_ARM':arm,'OPSD_OUTPUT':str(output),'OPSD_MODEL':str(ROOT/'models/Qwen3-1.7B'),'CONTRAST_NEGATIVES':str(DATA/'negative.json')},stdout=log,stderr=subprocess.STDOUT,check=True)
    import importlib.util
    spec=importlib.util.spec_from_file_location('finish',HERE.parent/'repair/finish.py')
    # Retain the pre-registered original AIME settings and paired grader using the matched arm.
    source=Path(spec.origin).read_text().replace('R-contrast-v2@100',arm+'@100')
    sys.path.insert(0,str(HERE.parent/'repair'))
    ns={'__name__':'registered_evaluator'}
    exec(compile(source,spec.origin,'exec'),ns)
    ns.update(WORK=DATA,HERE=HERE)
    ns['main']()
if __name__=='__main__':main()
