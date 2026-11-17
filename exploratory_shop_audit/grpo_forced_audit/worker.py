"""Isolated CPU engine RPC with the new frozen observation interface."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback
from .interface import Engine


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    engine=Engine(args.run)
    print(json.dumps({'ready':True,'tables':engine.hashes}),flush=True)
    def execute(req):
        cmd=req['cmd']
        if cmd=='open': return engine.open(req['key'],req['goal_index'],req.get('prefix',[]))
        if cmd=='step': return engine.step(req['key'],req['action'])
        if cmd=='close': return engine.close(req['key'])
        if cmd=='state': return engine.state(req['key'])
        if cmd=='parser_audit': return engine.parser_audit()
        if cmd=='maximum':
            from .cpu import maximum
            return maximum(engine,req['goal_index'],req['asin'],req.get('fixed'))
        if cmd=='batch': return [execute(r) for r in req['requests']]
        raise ValueError(cmd)
    for line in sys.stdin:
        req=json.loads(line)
        if req['cmd']=='quit':break
        try:
            with contextlib.redirect_stdout(sys.stderr):result=execute(req)
            reply={'ok':True,'result':result}
        except Exception:reply={'ok':False,'error':traceback.format_exc()}
        print(json.dumps(reply,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
