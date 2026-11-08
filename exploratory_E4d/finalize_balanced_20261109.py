"""Complete balanced E4d, verify exact remote artifacts, then authorized shutdown."""
import sys
from pathlib import Path
import finalize_accelerated_20261109 as M
import balance_20261109 as B
F=M.F
F.R=B.R
F.FIN=B.H/'finalization_balanced_20261109'


def command(args,**kwargs):
    words=list(map(str,args))
    if len(words)==3 and words[1:]==[str(B.H/'run_server.py'),'--verify']:
        words=[words[0],str(B.H/'balance_20261109.py'),'--verify']
    if words[:2]==['codex','queue'] and '--message' in words:
        i=words.index('--message')+1
        words[i]=words[i].replace(str(B.PREV),str(B.R)).replace('accelerate_20261108.py','balance_20261109.py').replace('finalize_accelerated_20261109.py','finalize_balanced_20261109.py')
    return M.original_command(words,**kwargs)
F.command=M.command=command


def artifacts():
    out=M._original_artifacts()
    for directory in [B.PREV,B.H/'primary_complete_20261109']:
        for p in directory.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix not in ('.lock','.tmp','.pyc','.pending-target') and '__pycache__' not in p.parts:out.append(p)
    return sorted(set(out))
F.artifacts=artifacts
_previous_live=M.live_experiment_processes


def live_experiment_processes():
    found=_previous_live()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            words=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if words and 'python' in Path(words[0]).name and any(Path(w).name in ('balance_20261109.py','primary_complete_20261109.py') for w in words[1:]) and F.alive(int(p.name)):
                found.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return sorted(set(found))
F.live_experiment_processes=live_experiment_processes
# M.recover's shutdown safety uses its module-level function as well.
M.live_experiment_processes=live_experiment_processes

if __name__=='__main__':F.main()
