"""Download the official research files used by the shoe training pipeline.

The script only downloads files from the dataset owners' published URLs. It
supports HTTP resume so multi-gigabyte archives do not restart after a failure.
Dataset licenses still apply; do not commit or redistribute the downloaded data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


FASHIONPEDIA = {
    "train2020.zip": "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",
    "val_test2020.zip": "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip",
    "instances_attributes_train2020.json": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
    "instances_attributes_val2020.json": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",
}


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    if existing:
        head = requests.head(url, allow_redirects=True, timeout=(30, 120))
        remote_size = int(head.headers.get("Content-Length", 0))
        if head.ok and remote_size and existing == remote_size:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            return {"url": url, "path": str(path.resolve()), "bytes": existing, "sha256": digest}
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing and response.status_code != 206:
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing else "wb"
        expected = int(response.headers.get("Content-Length", 0)) + existing
        with path.open(mode) as output:
            current = existing
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
                    current += len(chunk)
                    if current // (100 * 1024 * 1024) != (current - len(chunk)) // (100 * 1024 * 1024):
                        print(f"{path.name}: {current / 1024**3:.2f} GiB", flush=True)
    if expected and path.stat().st_size != expected:
        raise IOError(f"Incomplete download: {path} ({path.stat().st_size} != {expected})")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"url": url, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", nargs="*", choices=tuple(FASHIONPEDIA), default=tuple(FASHIONPEDIA))
    args = parser.parse_args()
    manifest = args.output / "download_manifest.json"
    existing_records = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
    records = [download(FASHIONPEDIA[name], args.output / name) for name in args.files]
    merged = {record["path"]: record for record in existing_records}
    merged.update({record["path"]: record for record in records})
    manifest.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
    print(f"Saved {manifest}", flush=True)


if __name__ == "__main__":
    main()
