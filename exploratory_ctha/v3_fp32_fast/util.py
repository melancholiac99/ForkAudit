import hashlib,json,os,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parent
MODEL=pathlib.Path('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
DATA=pathlib.Path(os.environ.get('CTHA_RUN_ROOT','/root/autodl-tmp/ctha_v3_engineering_fast_20261113'))
sys.path.insert(0,str(PARENT))
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def immutable(p,x):
 p=pathlib.Path(p)
 if p.exists():
  if load(p)!=x:raise ValueError(f'Immutable manifest mismatch: {p}')
 else:save(p,x)
def source_hashes():
 files=list(HERE.glob('*.py'))+[HERE/'PLAN.json']+[PARENT/n for n in ['habits.py','habits_v2.py','features_v2.py','train.py','run.py','prepare.py']]
 files += [ROOT/'scripts'/n for n in ['common.py','envpool.py','alfenv.py']]
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
