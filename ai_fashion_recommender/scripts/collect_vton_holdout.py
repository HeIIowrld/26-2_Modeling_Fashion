"""공개 웹 이미지의 신규 연구 평가 표본 수집. 인물 독립성을 자동 주장하지 않는다."""
import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from fetch_deepfashion_mm_sample import HttpRangeFile

ARCHIVE = "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip"
ANNOTATIONS = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"
ALLOWED_LICENSES = {0, 1, 3, 5, 6, 7, 8, 9, 10}  # ND 및 출처/권리 미확인은 제외


def source_site(url):
    host = urlparse(url).hostname or "unknown"
    return "flickr.com" if "flickr.com" in host else host.removeprefix("www.")


def select(data, count):
    cats = {c["id"]: c["name"] for c in data["categories"]}
    annotations = defaultdict(set)
    for a in data["annotations"]:
        annotations[a["image_id"]].add(a["category_id"])
    pools = defaultdict(list)
    for row in data["images"]:
        labels = annotations[row["id"]]
        if row["license"] not in ALLOWED_LICENSES or min(row["width"], row["height"]) < 320:
            continue
        if row["height"] < row["width"] or not labels.intersection(range(12)):
            continue
        bottom = next((cats[c] for c in (10, 11, 8, 7, 6) if c in labels), "upper_only")
        pools[source_site(row["original_url"]), bottom].append({
            **row, "source_site": source_site(row["original_url"]),
            "source_bottom": bottom, "source_clothing": [cats[c] for c in sorted(labels) if c < 12],
            "source_shoes_visible": 23 in labels,
        })
    rng = random.Random(20260916)
    for rows in pools.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(selected) < count:
                selected.append(pools[key].pop())
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, action="append", required=True)
    ap.add_argument("--count", type=int, default=120)
    args = ap.parse_args()
    root = args.root
    if (root / "manifest.json").exists():
        raise RuntimeError("동결된 표본을 덮어쓰지 않습니다. 새 평가 폴더를 사용하세요.")
    data = json.loads((root / "fashionpedia_val.json").read_text(encoding="utf-8"))
    licenses = {r["id"]: r for r in data["licenses"]}
    previous = {}
    for directory in args.exclude:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for p in directory.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                previous[hashlib.sha256(p.read_bytes()).hexdigest()] = str(p)
    candidates = select(data, args.count)
    (root / "people").mkdir(exist_ok=True)
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ARCHIVE), buffer_size=1 << 20))
    members = {Path(n).name: n for n in archive.namelist() if n.endswith(".jpg")}
    manifest, failures = [], []
    for row in candidates:
        try:
            raw = archive.read(members[row["file_name"]])
            digest = hashlib.sha256(raw).hexdigest()
            if digest in previous:
                failures.append({"id": row["id"], "reason": "duplicate", "previous": previous[digest]})
                continue
            name = "fp_" + row["file_name"]
            (root / "people" / name).write_bytes(raw)
            manifest.append({**row, "image": name, "sha256": digest,
                             "license_info": licenses[row["license"]], "download_source": ARCHIVE,
                             "identity_status": "human_verification_pending",
                             "body_shape": "unlabeled", "pose": "unlabeled", "angle": "unlabeled",
                             "background": "unlabeled", "subject_group": ""})
            previous[digest] = name
            print("COLLECT", len(manifest), name, row["source_site"], flush=True)
        except Exception as exc:
            failures.append({"id": row["id"], "reason": str(exc)})
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"requested": args.count, "selected": len(candidates), "downloaded": len(manifest),
               "source_counts": dict(Counter(r["source_site"] for r in manifest)),
               "clothing_counts": dict(Counter(r["source_bottom"] for r in manifest)),
               "failures": failures, "excluded_roots": [str(p) for p in args.exclude],
               "identity_independence": "unverified; image-disjoint only",
               "annotations_sha256": hashlib.sha256((root / "fashionpedia_val.json").read_bytes()).hexdigest()}
    (root / "collection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
