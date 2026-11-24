"""Download pinned public OPSD inputs. Validate each LFS or Git blob before use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import time

import requests


ASSETS = {
    'model': ('models', 'Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    'train': ('datasets', 'siyanzhao/Openthoughts_math_30k_opsd', '1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62'),
    'aime24': ('datasets', 'HuggingFaceH4/aime_2024', '2fe88a2f1091d5048c0f36abc874fb997b3dd99a'),
    'aime25': ('datasets', 'yentinglin/aime_2025', '6f71d77b0b89b9dabe07ab466c51df33f514df7f'),
    'hmmt25': ('datasets', 'MathArena/hmmt_feb_2025', '6fdc4277120810ff75aa22d2d5489b91f7a262a1'),
}


def check(path, blob):
    if not path.is_file() or path.stat().st_size != blob['size']:
        return False
    if 'lfs' in blob:
        h = hashlib.sha256()
        expected = blob['lfs']['sha256']
    else:
        h = hashlib.sha1(f"blob {blob['size']}\0".encode())
        expected = blob['blobId']
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest() == expected


def fetch(entry):
    dest, blob, url = entry
    dest.parent.mkdir(parents=True, exist_ok=True)
    if check(dest, blob):
        return
    tmp = dest.with_suffix(dest.suffix+'.part')
    for attempt in range(4):
        try:
            host = 'https://huggingface.co' if attempt % 2 == 0 else 'https://hf-mirror.com'
            if blob['size'] > 250_000_000 and shutil.which('aria2c'):
                subprocess.run(['aria2c','--quiet','--continue=true','--auto-file-renaming=false',
                                '--allow-overwrite=true','--file-allocation=none','-x','12','-s','12','-k','8M',
                                '--retry-wait=1','--max-tries=3','--dir',str(tmp.parent),'--out',tmp.name,
                                host+url+'?download=true&run='+str(time.time_ns())],check=True,env={
                                    **os.environ,'https_proxy':os.environ.get('https_proxy',os.environ.get('HTTPS_PROXY',''))})
                if not check(tmp,blob):
                    tmp.unlink()
                    raise ValueError(f'Content hash mismatch for {dest.name}')
                tmp.replace(dest)
                print('verified',dest,flush=True)
                return
            with requests.get(host+url, params={'download':'true','run':str(time.time_ns())},
                              stream=True, timeout=(10,90)) as res:
                res.raise_for_status()
                with tmp.open('wb') as out:
                    for chunk in res.iter_content(2 << 20):
                        out.write(chunk)
            if not check(tmp,blob):
                raise ValueError(f'Content hash mismatch for {dest.name}')
            tmp.replace(dest)
            print('verified',dest,flush=True)
            return
        except (requests.RequestException, subprocess.CalledProcessError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main(root):
    entries = []
    manifest = {}
    for key, (kind, repo, revision) in ASSETS.items():
        url = f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}'
        response = requests.get(url,params={'blobs':'true'},timeout=30)
        response.raise_for_status()
        meta = response.json()
        assert meta['sha'] == revision
        dest = root/('models/Qwen3-1.7B' if key=='model' else 'datasets/'+key)
        prefix = '/' if key=='model' else '/datasets/'
        blobs = [b for b in meta['siblings'] if b['rfilename']!='.gitattributes' and
                 (key=='model' or b['rfilename']=='README.md' or b['rfilename'].endswith('.parquet'))]
        manifest[key] = dict(repository=repo, revision=revision, files=blobs)
        for b in blobs:
            name=b['rfilename']
            entries.append((dest/name,b,f'{prefix}{repo}/resolve/{revision}/{name}'))
    with ThreadPoolExecutor(6) as workers:
        list(workers.map(fetch,entries))
    temp=root/'assets.json.tmp'
    temp.write_text(json.dumps(manifest,indent=2)+'\n')
    temp.replace(root/'assets.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
