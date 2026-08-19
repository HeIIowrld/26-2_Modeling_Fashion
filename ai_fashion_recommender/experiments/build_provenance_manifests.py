"""데이터 출처 매니페스트 생성 (2~6단계).

GitHub에 올릴 수 있는 최소 매핑만 만든다. 절대경로는 저장하지 않는다.

출력
  reports/manifests/fashionpedia_train_r1_images.csv
  reports/manifests/fashionpedia_train_r2_images.csv
  reports/manifests/fashionpedia_val_seed_images.csv
  reports/manifests/fashion200k_images.csv
  reports/manifests/used_image_manifest.csv
  reports/manifests/unmapped_seed_filenames.txt   (100% 매핑 실패 시에만)
  data/fashionpedia_train/manifest_r2_corrected.json
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data"
PROVENANCE = DATA / "provenance" / "fashionpedia"
MANIFESTS = PROJECT_DIR / "reports" / "manifests"

IMAGE_COLUMNS = [
    "dataset", "official_split", "image_id", "original_filename", "source_shard",
    "augmentation_round", "selected", "valid_crop_count", "usage_split", "used_for_training",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  저장: {path.relative_to(PROJECT_DIR).as_posix()}  ({len(rows):,}행)")


def crop_stats(rows: list[dict]) -> tuple[Counter, dict[str, set[str]]]:
    """파일명별 crop 수와 등장한 split 집합."""
    counts: Counter[str] = Counter()
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        name = Path(row["image_path"].replace("\\", "/")).name
        counts[name] += 1
        splits[name].add(row["split"])
    return counts, splits


# ---------------------------------------------------------------- 2. r1 / r2

def build_round(round_number: int, subset_path: Path, selection_path: Path,
                csv_path: Path, id_to_shard: dict[int, int]) -> list[dict]:
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    id_to_name = {int(item["id"]): item["file_name"] for item in subset["images"]}
    selected_ids = json.loads(selection_path.read_text())["image_ids"]
    counts, splits = crop_stats(read_csv(csv_path))

    usage_map = {"train": "train", "val_train_split": "new_val"}
    rows = []
    for image_id in sorted(selected_ids):
        name = id_to_name.get(int(image_id), "")
        crop_count = counts.get(name, 0)
        if crop_count == 0:
            usage, used = "selected_no_valid_crop", False
        else:
            found = splits[name]
            if len(found) != 1:
                raise SystemExit(f"이미지 하나가 여러 split에 걸쳐 있습니다: {name} {found}")
            usage = usage_map[next(iter(found))]
            used = usage == "train"
        rows.append({
            "dataset": "fashionpedia",
            "official_split": "train2020",
            "image_id": int(image_id),
            "original_filename": name,
            "source_shard": id_to_shard.get(int(image_id), ""),
            "augmentation_round": round_number,
            "selected": "true",
            "valid_crop_count": crop_count,
            "usage_split": usage,
            "used_for_training": "true" if used else "false",
        })
    return rows


# ---------------------------------------------------------------- 3. seed

def build_seed(annotation_path: Path) -> tuple[list[dict], list[str], dict]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    name_to_ids: dict[str, list[int]] = defaultdict(list)
    for item in payload["images"]:
        name_to_ids[item["file_name"]].append(int(item["id"]))
    duplicates = {n: v for n, v in name_to_ids.items() if len(v) > 1}

    combined = read_csv(DATA / "fashion_attribute_annotations.csv")
    seed_rows = [r for r in combined if r["image_path"].startswith("fashionpedia_seed/")]
    counts, splits = crop_stats(seed_rows)

    rows, unmapped = [], []
    for name in sorted(counts):
        ids = name_to_ids.get(name, [])
        if not ids:
            unmapped.append(name)
        found = splits[name]
        if len(found) != 1:
            raise SystemExit(f"seed 이미지가 여러 split에 걸쳐 있습니다: {name} {found}")
        split = next(iter(found))
        rows.append({
            "dataset": "fashionpedia",
            "official_split": "val2020",
            "image_id": ids[0] if ids else "",
            "original_filename": name,
            "source_shard": "",
            "augmentation_round": 0,
            "selected": "true",
            "valid_crop_count": counts[name],
            "usage_split": "train" if split == "train" else "old_val",
            "used_for_training": "true" if split == "train" else "false",
        })
    stats = {
        "annotation_images": len(payload["images"]),
        "unique_filenames_in_csv": len(counts),
        "duplicate_filenames_in_annotation": len(duplicates),
        "mapped": len(rows) - len(unmapped),
        "unmapped": len(unmapped),
    }
    return rows, unmapped, stats


# ---------------------------------------------------------------- 4. fashion200k

FASHION200K_COLUMNS = [
    "dataset", "source_subset", "product_id", "image_number", "original_filename",
    "crop_count", "usage_split", "used_for_training",
]


def build_fashion200k() -> list[dict]:
    combined = read_csv(DATA / "fashion_attribute_annotations.csv")
    rows_by_subset: dict[str, list[dict]] = defaultdict(list)
    for row in combined:
        parts = row["image_path"].replace("\\", "/").split("/")
        if parts[0].startswith("fashion200k"):
            rows_by_subset[parts[0]].append(row)

    manifest = []
    for subset, subset_rows in sorted(rows_by_subset.items()):
        counts, splits = crop_stats(subset_rows)
        for name in sorted(counts):
            stem = Path(name).stem
            product_id, _, image_number = stem.rpartition("_")
            found = splits[name]
            if len(found) != 1:
                raise SystemExit(f"fashion200k 이미지가 여러 split에 걸쳐 있습니다: {name} {found}")
            split = next(iter(found))
            manifest.append({
                "dataset": "fashion200k",
                "source_subset": subset,
                "product_id": product_id,
                "image_number": image_number,
                "original_filename": name,
                "crop_count": counts[name],
                "usage_split": "train" if split == "train" else "old_val",
                "used_for_training": "true" if split == "train" else "false",
            })
    return manifest


# ---------------------------------------------------------------- 5. 통합

USED_COLUMNS = [
    "unique_key", "dataset", "official_split", "source_subset", "image_id",
    "original_filename", "product_id", "image_number", "source_shard",
    "augmentation_round", "crop_count", "usage_split", "used_for_training",
    "image_id_recovered",
]


def build_used(r1, r2, seed, f200k) -> list[dict]:
    rows = []
    for entry in (*r1, *r2, *seed):
        has_id = entry["image_id"] != ""
        key = (
            f"fashionpedia::{entry['official_split']}::{entry['image_id']}" if has_id
            else f"fashionpedia::{entry['official_split']}::{entry['original_filename']}"
        )
        rows.append({
            "unique_key": key,
            "dataset": "fashionpedia",
            "official_split": entry["official_split"],
            "source_subset": "",
            "image_id": entry["image_id"],
            "original_filename": entry["original_filename"],
            "product_id": "", "image_number": "",
            "source_shard": entry["source_shard"],
            "augmentation_round": entry["augmentation_round"],
            "crop_count": entry["valid_crop_count"],
            "usage_split": entry["usage_split"],
            "used_for_training": entry["used_for_training"],
            "image_id_recovered": "true" if has_id else "false",
        })
    for entry in f200k:
        rows.append({
            "unique_key": (
                f"fashion200k::{entry['source_subset']}::{entry['product_id']}::{entry['image_number']}"
            ),
            "dataset": "fashion200k",
            "official_split": "",
            "source_subset": entry["source_subset"],
            "image_id": "",
            "original_filename": entry["original_filename"],
            "product_id": entry["product_id"],
            "image_number": entry["image_number"],
            "source_shard": "", "augmentation_round": "",
            "crop_count": entry["crop_count"],
            "usage_split": entry["usage_split"],
            "used_for_training": entry["used_for_training"],
            "image_id_recovered": "true",
        })
    return rows


# ---------------------------------------------------------------- 6. manifest_r2 교정

def correct_manifest_r2(id_to_shard: dict[int, int]) -> dict:
    original_path = DATA / "fashionpedia_train" / "manifest_r2.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    selection = json.loads((DATA / "fashionpedia_train" / "selection_r2.json").read_text())
    ids = selection["image_ids"]
    shards = sorted({id_to_shard[int(i)] for i in ids})
    rows = read_csv(DATA / "fashionpedia_train" / "fashion_attribute_annotations_r2.csv")

    corrected = dict(original)
    corrected["shards_used"] = shards
    corrected["selection"] = (
        f"greedy set cover targeting {selection['target']} samples per label "
        f"with current train support < {selection['target']}, "
        f"restricted to image_id >= {selection['min_image_id']} "
        f"and excluding round-1 selection"
    )
    corrected["selection_target"] = selection["target"]
    corrected["min_image_id"] = selection["min_image_id"]
    corrected["images"] = len(ids)
    corrected["csv_rows"] = len(rows)
    corrected["split_counts"] = dict(Counter(r["split"] for r in rows))
    corrected["corrected_from"] = "manifest_r2.json"
    corrected["correction_note"] = (
        "원본은 build_train_csv.py 가 shards_used=[0,1] 과 selection 문구를 하드코딩해 "
        "2차 실행에도 1차 값이 그대로 기록됐다. shard 목록은 data/provenance/fashionpedia/"
        "shard_index.json 의 image_id→shard 색인으로 재계산했다."
    )
    corrected["evidence"] = {
        "selection_json": "data/fashionpedia_train/selection_r2.json",
        "csv": "data/fashionpedia_train/fashion_attribute_annotations_r2.csv",
        "shard_index": "data/provenance/fashionpedia/shard_index.json",
    }
    return original, corrected


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    shard_index = json.loads((PROVENANCE / "shard_index.json").read_text(encoding="utf-8"))
    id_to_shard = {int(k): v for k, v in shard_index["id_to_shard"].items()}

    print("2. Fashionpedia train 보강 매니페스트")
    r1 = build_round(1, PROVENANCE / "r1" / "instances_subset.json",
                     DATA / "fashionpedia_train" / "selection.json",
                     DATA / "fashionpedia_train" / "fashion_attribute_annotations.csv", id_to_shard)
    r2 = build_round(2, PROVENANCE / "r2" / "instances_subset.json",
                     DATA / "fashionpedia_train" / "selection_r2.json",
                     DATA / "fashionpedia_train" / "fashion_attribute_annotations_r2.csv", id_to_shard)
    write_csv(MANIFESTS / "fashionpedia_train_r1_images.csv", IMAGE_COLUMNS, r1)
    write_csv(MANIFESTS / "fashionpedia_train_r2_images.csv", IMAGE_COLUMNS, r2)

    print("\n3. Fashionpedia seed(val2020) image_id 복원")
    annotation = PROVENANCE / "seed" / "instances_attributes_val2020.json"
    seed, unmapped, seed_stats = build_seed(annotation)
    write_csv(MANIFESTS / "fashionpedia_val_seed_images.csv", IMAGE_COLUMNS, seed)
    print(f"  주석 이미지 {seed_stats['annotation_images']:,} / CSV 고유 파일명 "
          f"{seed_stats['unique_filenames_in_csv']:,}")
    print(f"  매핑 성공 {seed_stats['mapped']:,} "
          f"({seed_stats['mapped']/max(seed_stats['unique_filenames_in_csv'],1)*100:.1f}%)  "
          f"실패 {seed_stats['unmapped']}")
    print(f"  주석 내 중복 파일명 {seed_stats['duplicate_filenames_in_annotation']}")
    if unmapped:
        path = MANIFESTS / "unmapped_seed_filenames.txt"
        path.write_text("\n".join(unmapped), encoding="utf-8")
        print(f"  미매핑 목록 저장: {path.name}")

    print("\n4. Fashion200K 매니페스트")
    f200k = build_fashion200k()
    write_csv(MANIFESTS / "fashion200k_images.csv", FASHION200K_COLUMNS, f200k)

    print("\n5. 통합 매니페스트")
    used = build_used(r1, r2, seed, f200k)
    write_csv(MANIFESTS / "used_image_manifest.csv", USED_COLUMNS, used)

    print("\n6. manifest_r2 교정본")
    original, corrected = correct_manifest_r2(id_to_shard)
    target = DATA / "fashionpedia_train" / "manifest_r2_corrected.json"
    target.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {target.relative_to(PROJECT_DIR).as_posix()} (원본 manifest_r2.json 은 그대로 둠)")
    print("\n  원본 vs 교정본 차이")
    for key in sorted(set(original) | set(corrected)):
        before, after = original.get(key, "<없음>"), corrected.get(key, "<없음>")
        if before != after:
            print(f"    - {key}")
            print(f"        원본  : {before}")
            print(f"        교정본: {after}")

    (MANIFESTS / "sources.json").write_text(json.dumps({
        "fashionpedia_train2020_annotation": {
            "url": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
            "note": "로컬 보존 안 함(542MB). r1/r2 instances_subset.json 이 파생본.",
        },
        "fashionpedia_val2020_annotation": {
            "url": json.loads((DATA / "fashionpedia_seed" / "manifest.json").read_text(
                encoding="utf-8"))["annotation_source"],
            "sha256": sha256(annotation),
            "local_path": "data/provenance/fashionpedia/seed/instances_attributes_val2020.json (gitignored)",
        },
        "fashionpedia_images_parquet": {
            "url": shard_index["source"],
            "shards": shard_index["shards"],
        },
        "instances_subset": {
            "r1": {"local_path": "data/provenance/fashionpedia/r1/instances_subset.json (gitignored)",
                   "sha256": sha256(PROVENANCE / "r1" / "instances_subset.json")},
            "r2": {"local_path": "data/provenance/fashionpedia/r2/instances_subset.json (gitignored)",
                   "sha256": sha256(PROVENANCE / "r2" / "instances_subset.json")},
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  저장: reports/manifests/sources.json")


if __name__ == "__main__":
    main()
