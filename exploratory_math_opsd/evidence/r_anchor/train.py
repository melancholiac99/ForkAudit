from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/ForkAudit/math-opsd')
OUT = ROOT/'run/R-anchor'
SRC = OUT/'source'
sys.path.insert(0, str(HERE.parents[1]))
from launch import validate_checkpoint
from anchor import gradient_check


def patch(file, old, new):
    text = file.read_text()
    assert text.count(old) == 1, old
    file.write_text(text.replace(old, new))


def train(end):
    assert end in (100, 200)
    OUT.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'source', SRC, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git','__pycache__'))
    shutil.copy(HERE/'anchor.py', SRC)
    trainer = SRC/'opsd_trainer.py'
    old = '            del student_logits_for_loss, teacher_logits_for_loss'
    patch(trainer, old, '            from anchor import augment\n            loss = augment(self, model, inputs, student_logits_for_loss, teacher_logits_for_loss, shifted_labels, loss)\n'+old)
    old = '        total_completion_tokens = sum(len(ids) for ids in completion_ids)'
    patch(trainer, old, '        self.anchor_counts[0] += sum(bool(ids) and ids[-1] in self.anchor_eos for ids in completion_ids)\n        self.anchor_counts[1] += len(completion_ids)\n'+old)
    launch = SRC/'opsd_train.py'
    old = '        if state.global_step == 10:'
    patch(launch, old, '        args.save_only_model = state.global_step != 100\n'+old)
    old = '    trainer.train(resume_from_checkpoint='
    patch(launch, old, '    from anchor import Monitor\n    trainer.add_callback(Monitor(trainer, model_args.model_name_or_path))\n'+old)
    cmd = ['bash','scripts/run_opsd_1b.sh','--stop_after_updates',str(end),'--seed','42','--data_seed','42',
           '--local_train_glob',str(ROOT/'datasets/train/data/*.parquet')]
    if end == 200:
        ck = OUT/'checkpoint-100'
        validate_checkpoint(ck, 100)
        assert len(list((ck/'global_step100').glob('*optim_states.pt'))) == 4
        assert (ck/'scheduler.pt').exists() and all((ck/f'rng_state_{i}.pth').exists() for i in range(4))
        cmd += ['--resume_from_checkpoint', str(ck)]
    elif (OUT/'anchor_windows.jsonl').exists():
        raise RuntimeError('Existing trajectory: refuse to restart over archived results')
    env = dict(os.environ, OPSD_ARM='R-anchor', OPSD_OUTPUT=str(OUT), OPSD_MODEL=str(ROOT/'models/Qwen3-1.7B'))
    with (OUT/f'train-{end}.log').open('a') as f:
        subprocess.run(cmd, cwd=SRC, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    print('gradient max error', gradient_check(), flush=True)
    train(int(sys.argv[1]))
