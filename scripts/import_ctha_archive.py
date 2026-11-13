"""Verify and import original P0/P1 archives; never fabricate missing rollouts."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, default=ROOT / ".runtime/imports/incoming")
    parser.add_argument("--out", type=Path, default=ROOT / ".runtime/imports")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "exploratory_ctha/p1_results_20261113/archive_manifest.json").read_text())
    parts = []
    for spec in manifest["parts"]:
        path = args.incoming / Path(spec["path"]).name
        if not path.is_file():
            parser.error(f"Missing original archive part: {path}")
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            parser.error(f"Archive part does not match published manifest: {path}")
        parts.append(path)
    args.out.mkdir(parents=True, exist_ok=True)
    joined = args.out / "verified_raw.tar.gz"
    with joined.open("wb") as output:
        for part in parts:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    seen = set()
    with tarfile.open(joined, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in manifest["files"] or member.name in seen:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            target = (args.out / member.name).resolve()
            if not target.is_relative_to(args.out.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            spec = manifest["files"][member.name]
            data = archive.extractfile(member).read()
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"Archive member checksum mismatch: {member.name}")
            if target.exists():
                if digest(target) != spec["sha256"]:
                    raise ValueError(f"Refusing to overwrite different file: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            seen.add(member.name)
    if seen != set(manifest["files"]):
        raise ValueError("Archive is incomplete")
    report = dict(verified=True, files=len(seen), parts=[str(p) for p in parts],
                  manifest_sha256=digest(ROOT / "exploratory_ctha/p1_results_20261113/archive_manifest.json"))
    (args.out / "import_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
