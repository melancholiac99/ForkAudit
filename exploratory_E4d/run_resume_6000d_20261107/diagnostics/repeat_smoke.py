from pathlib import Path
import sys
H=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(H))
import resume_6000d_20261107 as M
if __name__=='__main__':
    i=int(sys.argv[1]); dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
    original=M.E.save_json
    def save(path,content):
        assert Path(path)==M.R/f'lora_smoke_{i}.json'
        original(dest/Path(path).name,content)
    M.E.save_json=save
    M.lora_smoke(i)
