"""Export all E4 result artifacts except tensor checkpoints, with verified split archive.

Reads the original final artifact manifest. Never edits source results or models.
"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile

H = Path(__file__).resolve().parent
OUT = H/'results_archive'
PART_BYTES = 32*1024**2


class SplitWriter:
    def __init__(self):
        self.parts = []; self.handle = None; self.count = 0; self.total = 0
        self.hash = hashlib.sha256()

    def finish_part(self):
        if self.handle is not None:
            self.handle.close()
            self.parts.append(dict(path=Path(self.handle.name).name, bytes=self.count, sha256=self.hash.hexdigest()))
            self.handle = None

    def write(self, data):
        original = len(data)
        while data:
            if self.handle is None:
                self.handle = (OUT/f'e4_nonweight_results.tar.gz.part{len(self.parts):03d}').open('xb')
                self.count = 0; self.hash = hashlib.sha256()
            block = data[:PART_BYTES-self.count]; self.handle.write(block); self.hash.update(block)
            self.count += len(block); self.total += len(block); data = data[len(block):]
            if self.count == PART_BYTES:
                self.finish_part()
        return original

    def flush(self):
        if self.handle is not None: self.handle.flush()


class CheckedReader:
    def __init__(self, file):
        self.file = file; self.sha = hashlib.sha256()

    def read(self, size=-1):
        b = self.file.read(size); self.sha.update(b); return b


class JoinedReader(io.RawIOBase):
    def __init__(self, parts):
        self.paths = iter(parts); self.handle = None

    def readable(self): return True

    def readinto(self, buffer):
        while True:
            if self.handle is None:
                p = next(self.paths, None)
                if p is None: return 0
                self.handle = p.open('rb')
            n = self.handle.readinto(buffer)
            if n: return n
            self.handle.close(); self.handle = None


def main():
    source = H/'run/final_artifact_manifest.json'
    manifest = json.loads(source.read_text())
    assert json.loads((H/'run/final_verification.json').read_text())['passed']
    assert json.loads((H/'run/publication_recheck.json').read_text())['passed']
    assert not OUT.exists(), 'Archive already exists: inspect rather than overwrite'
    OUT.mkdir()
    included = {}; excluded = {}
    for name, info in sorted(manifest.items()):
        p = Path(name)
        assert p.parts[0] == 'data' and '..' not in p.parts
        assert (H/p).is_file() and not (H/p).is_symlink()
        (excluded if p.suffix == '.safetensors' else included)[name] = info
    writer = SplitWriter()
    with gzip.GzipFile(filename='', mode='wb', fileobj=writer, compresslevel=3, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for i, (name, info) in enumerate(included.items()):
                p = H/name; assert p.stat().st_size == info['bytes'], name
                entry = tarfile.TarInfo(name); entry.size = info['bytes']; entry.mode = 0o644
                entry.uid = entry.gid = entry.mtime = 0
                with p.open('rb') as f:
                    checked = CheckedReader(f); tar.addfile(entry, checked)
                    assert checked.sha.hexdigest() == info['sha256'], name
                if i % 5000 == 0: print('archived files', i, flush=True)
    writer.finish_part()
    # Independently decompress every published part and verify every member.
    verified = set()
    joined = io.BufferedReader(JoinedReader([OUT/p['path'] for p in writer.parts]))
    with tarfile.open(fileobj=joined, mode='r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in included and member.name not in verified
            h = hashlib.sha256(); stream = tar.extractfile(member)
            for b in iter(lambda: stream.read(1024*1024), b''): h.update(b)
            assert h.hexdigest() == included[member.name]['sha256'], member.name
            verified.add(member.name)
    assert verified == set(included)
    report = dict(complete_nonweight_results=True, source_manifest='run/final_artifact_manifest.json',
                  source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  included_files=len(included), uncompressed_bytes=sum(x['bytes'] for x in included.values()),
                  compressed_bytes=writer.total, parts=writer.parts, roundtrip_verified_files=len(verified),
                  excluded_checkpoint_tensors=excluded,
                  exclusion_reason='LoRA tensor checkpoints remain on server; all non-weight result artifacts are included.')
    (OUT/'archive_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    (OUT/'README.txt').write_text('''E4 complete result artifact archive (tensor checkpoints excluded)\n\nAll non-safetensors files in run/final_artifact_manifest.json are included.\nThis includes all original eval/train/tune trajectories, point/peek records,\nreplay outcomes, judge requests and scores, preference pairs, mechanism outputs,\nadapter configs and training logs. No scientific file is changed.\nThe separately published e4.json, E4_REPORT.txt and verification reports contain\nthe complete registered findings. LoRA tensor checkpoints remain on the server;\ntheir exact paths, sizes and SHA256 hashes are listed in archive_manifest.json.\n\nFrom exploratory_E4/results_archive/:\n  cat e4_nonweight_results.tar.gz.part* > /tmp/e4_nonweight_results.tar.gz\n  mkdir -p /tmp/e4_results\n  tar -xzf /tmp/e4_nonweight_results.tar.gz -C /tmp/e4_results\nThe extracted data/ tree recreates the archived non-weight experiment outputs.\nCheck each part SHA256 against archive_manifest.json before extraction; every\nmember was decompressed and checked against the original final manifest before\npublication. Checkpoints are not needed to reproduce reported statistics from\nthe recorded outcomes, but are needed to run the trained policies again.\n''')
    print(json.dumps({k: v for k, v in report.items() if k != 'excluded_checkpoint_tensors'}, indent=2), flush=True)


if __name__ == '__main__': main()
