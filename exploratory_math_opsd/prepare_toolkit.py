"""Install the matching NVIDIA compiler in the experiment prefix for ZeRO CPU offload."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

REDIST='https://developer.download.nvidia.com/compute/cuda/redist/'


def setup(root):
    prefix=root/'cuda-12.8'
    prefix.mkdir(parents=True,exist_ok=True)
    info=json.load(urlopen(REDIST+'redistrib_12.8.1.json',timeout=30))
    packages=('cuda_nvcc','cuda_cudart','cuda_cccl')
    with tempfile.TemporaryDirectory() as d:
        def download(name):
            item=info[name]['linux-x86_64']
            path=Path(d)/(name+'.tar.xz')
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3',
                            REDIST+item['relative_path'],'-o',str(path)],check=True)
            if sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError(f'Bad NVIDIA archive {name}')
            return path
        with ThreadPoolExecutor(3) as pool:
            for path in pool.map(download,packages):
                subprocess.run(['tar','-xJf',str(path),'--strip-components=1','-C',str(prefix)],check=True)
    if not (prefix/'lib64').exists():
        (prefix/'lib64').symlink_to('lib',target_is_directory=True)
    vendor=root/'conda-env/lib/python3.10/site-packages/nvidia'
    for package,libs in [('cublas',['libcublas.so.12','libcublasLt.so.12']),('curand',['libcurand.so.10'])]:
        for lib in libs:
            target=prefix/'lib'/lib.split('.so')[0]
            target=target.with_name(target.name+'.so')
            if not target.is_symlink():
                target.symlink_to(vendor/package/'lib'/lib)
    subprocess.run([prefix/'bin/nvcc','--version'],check=True)


if __name__=='__main__':
    setup(Path(sys.argv[1]).resolve())
