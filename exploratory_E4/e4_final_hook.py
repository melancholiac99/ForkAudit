"""Wait for the authorized E4 queue, then independently verify complete results."""
from pathlib import Path
import subprocess,sys,time,json,traceback,fcntl
H=Path(__file__).resolve().parent
from e4_pipeline import identity
import e4_common as E

def main():
 with (H/'run/final_hook.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   pid=int(sys.argv[1]);start=identity(pid);assert start
   while identity(pid)==start:time.sleep(30)
   status=json.loads((H/'run/pipeline_status.json').read_text());assert status['stage']=='completed_pending_review',status
   with (H/'run/final_verification.log').open('w') as log:subprocess.run([sys.executable,'e4_verify_final.py'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
   v=json.loads((H/'run/final_verification.json').read_text());assert v['passed']
   status.update(stage='completed_verified',verification='run/final_verification.json',manifest='run/final_artifact_manifest.json')
   E.write_json(str(H/'run/pipeline_status.json'),status)
   print(json.dumps(status),flush=True)
  except Exception as exc:
   E.write_json(str(H/'run/final_hook_error.json'),dict(type=type(exc).__name__,message=str(exc)))
   traceback.print_exc();raise
if __name__=='__main__':main()
