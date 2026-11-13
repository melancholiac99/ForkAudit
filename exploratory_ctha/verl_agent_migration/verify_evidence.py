"""Verify text-split evidence without extraction or loading pickled objects."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    index = json.loads((args.folder / 'evidence_index.json').read_text())
    encoded = []
    for part in index['parts']:
        data = (args.folder / part['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == part['sha256'], part['path']
        encoded.append(b''.join(data.split()))
    archive = base64.b64decode(b''.join(encoded), validate=True)
    assert len(archive) == index['archive_bytes']
    assert hashlib.sha256(archive).hexdigest() == index['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        manifest = json.loads(tar.extractfile('MANIFEST.json').read())
        for name, expected in manifest.items():
            data = tar.extractfile(name).read()
            assert len(data) == expected['bytes']
            assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
    print(json.dumps(dict(verified=True, parts=len(index['parts']), files=len(manifest))))


if __name__ == '__main__':
    main()
