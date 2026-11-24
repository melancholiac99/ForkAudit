"""Check the dedicated environment, official imports, pinned sources and complete inputs without training."""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys


def verify(root):
    if Path(sys.prefix).resolve()!=(root/'conda-env').resolve():
        raise RuntimeError('Unexpected Python environment')
    source=root/'source'
    sys.path.insert(0,str(source))
    import torch
    import flash_attn
    import vllm
    from trl.experimental.gold import GOLDConfig
    from opsd_trainer import OPSDTrainer
    from safetensors import safe_open
    pins=re.findall(r'^    - ([^= ]+)==(\S+)$',(source/'environment.yml').read_text(),re.M)
    installed={p:version(p) for p,_ in pins}
    for package,pin in pins:
        assert installed[package].split('+')[0]==pin,(package,pin,installed[package])
    assert flash_attn.__version__.split('+')[0]=='2.8.3'
    assert vllm.__version__.split('+')[0]=='0.11.0'
    manifest=json.loads((root/'assets.json').read_text())
    patch=Path(__file__).with_name('upstream.patch').read_bytes()
    assert subprocess.check_output(['git','-C',str(source),'diff','--binary','--unified=0'])==patch
    weights=root/'models/Qwen3-1.7B'
    weight_map=json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']
    for name in set(weight_map.values()):
        with safe_open(str(weights/name),framework='pt',device='cpu') as shard:
            assert set(shard.keys())=={k for k,v in weight_map.items() if v==name}
    return dict(python=sys.version.split()[0],conda_prefix=sys.prefix,versions=installed,flash_attn=flash_attn.__version__,
                torch_cuda=torch.version.cuda,source_revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                patch_sha256=sha256(patch).hexdigest(),assets={k:x['revision'] for k,x in manifest.items()},
                gold=GOLDConfig.__module__,trainer=OPSDTrainer.__name__,verified_tensor_names=len(weight_map))


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    data=verify(root)
    (root/'run/environment.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
