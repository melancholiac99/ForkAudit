"""Export our trusted local DataProto to non-pickled numerical evidence files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.environ['VERL_AGENT_SOURCE'])
from verl import DataProto


def convert(value):
    if isinstance(value, np.ndarray):
        return convert(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): convert(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert(child) for child in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packet', type=Path, help='Only load a locally generated, trusted native packet')
    args = parser.parse_args()
    data = DataProto.load_from_disk(str(args.packet))
    tensors = {key: value.detach().cpu().numpy() for key, value in data.batch.items()}
    out = args.packet.with_suffix('.npz')
    np.savez_compressed(out, **tensors)
    metadata = dict(non_tensor_batch=convert(data.non_tensor_batch), meta_info=convert(data.meta_info),
        source_sha256=hashlib.sha256(args.packet.read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    args.packet.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(arrays=str(out), fields=sorted(tensors), rows=len(data))))


if __name__ == '__main__':
    main()
