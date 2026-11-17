"""Freeze actual dependencies, raw-source identity, and complete hyperparameters."""
import argparse
import difflib
import subprocess
from .common import *
from .train import settings,SOURCE_VERL,MIGRATION


def freeze(run):
    versions={}
    for name,python in [('cpu',CPU_PYTHON),('gpu',GPU_PYTHON)]:
        result=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        (run/f'requirements_{name}_resolved.txt').write_text(result);versions[name]=digest_value(result)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv'],text=True)
    model_files={p.name:digest_file(p) for p in MODEL.iterdir() if p.is_file()}
    manifest={'project_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'environment_commit':'13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5','model_revision':'a09a35458c702b33eeacc393d103063234e8bc28',
        'model_path':str(MODEL.resolve()),'model_files_sha256':model_files,'hardware':hardware,'versions':versions,
        'framework_commit':'20bd331bdbc9026a5668e11362178e10ab7400c8','framework_sources':{str(p.relative_to(SOURCE_VERL)):digest_file(p) for p in SOURCE_VERL.rglob('*.py')},
        'code':{p.name:digest_file(p) for p in HERE.iterdir() if p.is_file()},'baseline_recipe_sha256':digest_file(SOURCE_VERL/'examples/grpo_trainer/run_webshop.sh'),
        'data_sha256':digest_file(SOURCE/'community_products.json'),'search_index_sha256':digest_file(SOURCE/'products.sqlite3'),
        'snapshot_hashes':{p.name:digest_file(p) for p in (run/'snapshots').glob('*.json')}}
    save(run/'environment_manifest.json',manifest)
    # Preserve the adapter as a complete additive diff, upstream files untouched.
    (run/'interface_patch.diff').write_text(''.join(difflib.unified_diff([], (HERE/'interface.py').read_text().splitlines(True),fromfile='/dev/null',tofile='grpo_forced_audit/interface.py')))
    cfg=settings(run)
    command=[str(GPU_PYTHON),'-c', '''import json,sys
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=sys.argv[1],version_base=None):
 c=compose(config_name='ppo_trainer',overrides=json.loads(sys.argv[2]))
 print(OmegaConf.to_yaml(c,resolve=True))
''',str(SOURCE_VERL/'verl/trainer/config'),json.dumps([f'{k}='+('null' if v is None else json.dumps(v,separators=(',',':')) if isinstance(v,(list,bool)) else str(v)) for k,v in cfg.items()])]
    output=subprocess.check_output(command,text=True,env=dict(os.environ,PYTHONPATH=str(SOURCE_VERL),OPENBLAS_NUM_THREADS='1'))
    (run/'resolved_train_config.yaml').write_text(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);a=p.parse_args();freeze(a.run)
