"""Download the frozen Qwen revision and official ALFWorld text-only data."""
import concurrent.futures
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, direct=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        download_env = os.environ.copy()
        if direct:
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                download_env.pop(key, None)
            download_env.update(NO_PROXY='*', no_proxy='*')
        if shutil.which("aria2c"):
            subprocess.run([
                "aria2c", "--continue=true", "--auto-file-renaming=false",
                "--allow-overwrite=true", "--file-allocation=none",
                "--max-connection-per-server=16", "--split=16", "--min-split-size=8M",
                "--max-tries=5", "--retry-wait=3", "--summary-interval=30",
                "--console-log-level=warn", "--download-result=hide",
                "--dir", str(path.parent), "--out", partial.name, url,
            ], check=True, env=download_env)
        else:
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "5", "--connect-timeout", "30", "--continue-at", "-",
                "--output", str(partial), url,
            ], check=True, env=download_env)
        partial.rename(path)
    print(f"Downloaded: {path.relative_to(ROOT)}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true', help='Prepare text environment data without waiting for model weights')
    parser.add_argument('--model-direct', action='store_true', help='Bypass configured proxies for model file downloads')
    args = parser.parse_args()
    model = RUNTIME / "models/Qwen2.5-7B-Instruct"
    metadata = download(
        f"https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct/revision/{REVISION}",
        RUNTIME / "models/qwen_revision.json",
    )
    info = json.loads(metadata.read_text())
    assert info["sha"] == REVISION
    names = [f["rfilename"] for f in info["siblings"]
             if f["rfilename"].endswith((".safetensors", ".json", ".txt"))
             or f["rfilename"] in ("LICENSE", "README.md")]
    tasks = [(f"https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/{REVISION}/{n}", model / n, args.model_direct)
             for n in names]
    if args.data_only:
        tasks = []
    for release, name in [
        ("0.2.2", "json_2.1.1_json.zip"),
        ("0.2.2", "json_2.1.1_pddl.zip"),
        ("0.4.0", "json_2.1.2_tw-pddl.zip"),
    ]:
        tasks.append((f"https://github.com/alfworld/alfworld/releases/download/{release}/{name}",
                      RUNTIME / "downloads" / name))
    # Every future is inspected: one failed download must fail preparation.
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
    if failures:
        raise RuntimeError(f"Asset downloads failed: {failures}")
    checks = {}
    for line in ([] if args.data_only else (ROOT / "env/qwen_sha256.txt").read_text().splitlines()):
        parts = line.split()
        if len(parts) == 2:
            expected, name = parts
            actual = digest(model / name)
            assert actual == expected, f"Model checksum mismatch: {name}"
            checks[name] = actual
    archives = {}
    for path in sorted((RUNTIME / "downloads").glob("*.zip")):
        archives[path.name] = digest(path)
        with zipfile.ZipFile(path) as archive:
            target = (RUNTIME / "alfworld").resolve()
            for name in archive.namelist():
                if not (target / name).resolve().is_relative_to(target):
                    raise ValueError(f"Unsafe archive path: {name}")
            archive.extractall(target)
    manifest = dict(model_revision=REVISION, model_sha256=checks,
                    data_archive_sha256=archives,
                    scope="Official text-only ALFWorld assets; no detector weights or training")
    name = 'data_assets_manifest.json' if args.data_only else 'assets_manifest.json'
    (RUNTIME / name).write_text(json.dumps(manifest, indent=2) + "\n")
    print("ALFWorld text data extracted; model checksums " + ('pending.' if args.data_only else 'verified.'), flush=True)


if __name__ == "__main__":
    main()
