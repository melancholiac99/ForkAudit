"""Build and verify a host-RAM copy of the base weights in the probe's exact dtype."""
import json
import os
from pathlib import Path
import tempfile

from exploratory_shop_audit.grpo_forced_audit.common import MODEL, read, save

CACHE=Path('/dev/shm/forkaudit-e25-fp32')
SIGNATURE='verified.json'


def identity(p):
    st=p.stat()
    return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]


def verified_path(cache=CACHE, source=MODEL):
    try:
        proof=read(cache/SIGNATURE)
        index='model.safetensors.index.json'
        weight_map=read(source/index)['weight_map']
        expected={index,'config.json','generation_config.json',*weight_map.values()}
        if (proof['source_path']!=str(source.resolve()) or proof['tensor_count']!=len(weight_map)
                or set(proof['source'])!=expected or set(proof['copy'])!=expected):
            return source
        for name,stamp in proof['source'].items():
            if identity(source/name)!=stamp:
                return source
        for name,stamp in proof['copy'].items():
            if identity(cache/name)!=stamp:
                return source
    except (OSError,KeyError,ValueError):
        return source
    return cache


def prepare(cache=CACHE, source=MODEL):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from exploratory_shop_audit.grpo_forced_audit.common import digest_file
    if verified_path(cache,source)==cache:
        return read(cache/SIGNATURE)
    if cache.exists():
        raise FileExistsError(cache)
    index='model.safetensors.index.json'
    weight_map=read(source/index)['weight_map']
    shards=sorted(set(weight_map.values()))
    cpu_files=[index,'config.json','generation_config.json']
    inputs={name:identity(source/name) for name in shards+cpu_files}
    with tempfile.TemporaryDirectory(dir=cache.parent,prefix='.forkaudit-build-') as staging:
        out=Path(staging)/'weights'
        out.mkdir(mode=0o700)
        for name in cpu_files:
            (out/name).write_bytes((source/name).read_bytes())
        count=0
        output_hashes={}
        for shard in shards:
            with safe_open(source/shard,framework='pt',device='cpu') as inp:
                keys=sorted(inp.keys())
                assert set(keys)=={key for key,name in weight_map.items() if name==shard}
                weights={key:inp.get_tensor(key).float() for key in keys}
                assert all(inp.get_slice(key).get_dtype()=='BF16' for key in keys)
                save_file(weights,out/shard,metadata=inp.metadata())
                del weights
                with safe_open(out/shard,framework='pt',device='cpu') as check:
                    assert sorted(check.keys())==keys
                    for key in keys:
                        copy=check.get_tensor(key)
                        assert copy.dtype==torch.float32 and torch.equal(copy,inp.get_tensor(key).float())
                        count+=1
                output_hashes[shard]=digest_file(out/shard)
        assert inputs=={name:identity(source/name) for name in inputs}
        # The HF index also stores total payload bytes; all source tensors were BF16.
        data=read(source/index)
        data['metadata']['total_size']*=2
        save(out/index,data)
        for name in inputs:
            (out/name).chmod(0o400)
        proof=dict(source_path=str(source.resolve()),tensor_count=count,source=inputs,
                   copy={name:identity(out/name) for name in inputs},sha256=output_hashes)
        save(out/SIGNATURE,proof)
        out.rename(cache)
    assert verified_path(cache,source)==cache
    return proof


if __name__=='__main__':
    print(json.dumps(prepare()))
