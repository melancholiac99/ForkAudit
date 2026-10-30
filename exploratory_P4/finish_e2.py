"""Finish the already authorized E2 queue; no push, shutdown, training or E3."""
from pathlib import Path
import os,subprocess,time,json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exploratory_P4'
def complete(path):return path.exists() and json.loads(path.read_text()).get('complete',False)
def run(cmd,log):
 print(json.dumps(dict(event='start',command=cmd)),flush=True)
 with (OUT/log).open('a') as f:subprocess.run([sys.executable,'-u',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(json.dumps(dict(event='finished',command=cmd)),flush=True)
def main():
 # 7B scoring and model download were started by the parent session.
 while not (complete(OUT/'e2_7b/run.json') and complete(OUT/'e2_32b_download.json')):time.sleep(10)
 time.sleep(5)
 run(['exploratory_P4/p4_e2_judge_score.py','--model','/root/autodl-tmp/models/Qwen2.5-32B-Instruct','--name','32b','--revision','5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'],'e2_32b.log')
 run(['exploratory_P4/p4_e2_analyze.py'],'e2_analysis.log')
 run(['exploratory_P4/verify_e2.py'],'e2_verification.log')
if __name__=='__main__':main()
