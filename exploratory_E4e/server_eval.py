"""Frozen single-card engine configuration; all scientific evaluation stays in four_arm_eval."""
import argparse,json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(H));import four_arm_eval as R

def install(changes):
    from e4_policy import LoraPolicy
    original_init=LoraPolicy.__init__
    def configured(self,*a,**kw):
        import vllm
        original=vllm.LLM
        class Engine(original):
            def __init__(self,*x,**y):y.update(changes);super().__init__(*x,**y)
        vllm.LLM=Engine
        try:original_init(self,*a,**kw)
        finally:vllm.LLM=original
    LoraPolicy.__init__=configured

def checked_plan(path):
    p=json.loads(path.read_text())
    for file,sha in p['source'].items():assert R.E.file_sha256(file)==sha,file
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    assert inventory==p['gpu_inventory'],'Hardware changed; register and check before continuation'
    assert all(os.environ.get(k)==v for k,v in p['environment'].items())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==p['gpu_uuid']
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args,rest=ap.parse_known_args()
    p=checked_plan(args.plan)
    os.environ['E4E_EXECUTION_PLAN_SHA256']=R.E.file_sha256(str(args.plan))
    install(p['engine_changes'])
    sys.argv=[str(H/'four_arm_eval.py'),*rest];R.main()
