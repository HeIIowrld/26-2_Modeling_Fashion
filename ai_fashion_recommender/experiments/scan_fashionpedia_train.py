"""7단계 A — Fashionpedia train 주석(542MB)을 스트리밍으로 훑어 희소 라벨 보유 이미지를 찾는다.

메모리가 1GB도 안 남은 환경이라 json.loads(전체 트리 수 GB)를 쓸 수 없다.
ijson으로 흘려 읽으면서 이미지당 "이 이미지가 기여할 라벨 집합"만 남긴다.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ijson

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fashion_attribute_dataset import (  # noqa: E402
    ATTRIBUTE_KEYWORDS,
    FASHIONPEDIA_CATEGORY_MAP,
    FASHIONPEDIA_GARMENT_PART_LABELS,
)
from fashion_attribute_schema import MULTI_LABEL_TASKS  # noqa: E402

SOURCE = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])


def attribute_label_map(attributes: dict[int, str]) -> dict[int, set[tuple[str, str]]]:
    """attribute_id -> {(task, 한국어 라벨)}. 300개뿐이라 미리 계산해두면 주석 순회가 빨라진다."""
    mapping: dict[int, set[tuple[str, str]]] = {}
    for attribute_id, raw_name in attributes.items():
        normalized = raw_name.lower().replace("_", "-").replace("/", " ")
        found = set()
        for task_name, keywords in ATTRIBUTE_KEYWORDS.items():
            for keyword, label in keywords.items():
                if keyword in normalized:
                    found.add((task_name, label))
        mapping[attribute_id] = found
    return mapping


def main() -> None:
    print(f"reading {SOURCE} ({SOURCE.stat().st_size / 1e6:.0f} MB)", flush=True)

    # --- 작은 섹션 먼저 (images / categories / attributes)
    images: dict[int, str] = {}
    with SOURCE.open("rb") as handle:
        for item in ijson.items(handle, "images.item", use_float=True):
            images[int(item["id"])] = item["file_name"]
    print(f"  images      {len(images):,}", flush=True)

    categories: dict[int, str] = {}
    with SOURCE.open("rb") as handle:
        for item in ijson.items(handle, "categories.item", use_float=True):
            categories[int(item["id"])] = str(item["name"]).lower().strip()
    print(f"  categories  {len(categories)}", flush=True)

    attributes: dict[int, str] = {}
    with SOURCE.open("rb") as handle:
        for item in ijson.items(handle, "attributes.item", use_float=True):
            supercategory = str(item.get("supercategory", "")).strip()
            name = str(item["name"]).strip()
            attributes[int(item["id"])] = " ".join(v for v in (supercategory, name) if v)
    print(f"  attributes  {len(attributes)}", flush=True)

    attribute_to_labels = attribute_label_map(attributes)
    part_categories = set(FASHIONPEDIA_GARMENT_PART_LABELS)

    # --- 주석 스트리밍: 이미지별 라벨 집합만 축적
    image_labels: dict[int, set[tuple[str, str]]] = defaultdict(set)
    image_annotation_count: Counter[int] = Counter()
    total = 0
    with SOURCE.open("rb") as handle:
        for annotation in ijson.items(handle, "annotations.item", use_float=True):
            total += 1
            if total % 50000 == 0:
                print(f"  annotations {total:,}", flush=True)
            image_id = int(annotation["image_id"])
            image_annotation_count[image_id] += 1
            raw_category = categories.get(int(annotation["category_id"]), "")
            found = image_labels[image_id]
            category = FASHIONPEDIA_CATEGORY_MAP.get(raw_category)
            if category:
                found.add(("category", category))
            if raw_category in part_categories:
                found.add(FASHIONPEDIA_GARMENT_PART_LABELS[raw_category])
            for attribute_id in annotation.get("attribute_ids", []):
                found |= attribute_to_labels.get(int(attribute_id), set())
    print(f"  annotations {total:,} (총 이미지 {len(image_labels):,})", flush=True)

    label_counts = Counter()
    for labels in image_labels.values():
        for pair in labels:
            label_counts[pair] += 1

    payload = {
        "source": str(SOURCE),
        "images": len(images),
        "annotations": total,
        "label_image_counts": {f"{t}|{l}": c for (t, l), c in sorted(label_counts.items())},
        "image_labels": {
            str(image_id): sorted(f"{t}|{l}" for t, l in labels)
            for image_id, labels in image_labels.items()
        },
        "image_file_names": {str(k): v for k, v in images.items()},
        "image_annotation_count": {str(k): v for k, v in image_annotation_count.items()},
    }
    OUTPUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nsaved: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")

    print("\n라벨을 보유한 이미지 수 (상위 30, 태스크|라벨):")
    for (task, label), count in label_counts.most_common(30):
        print(f"  {task:<15}{label:<14}{count:>7,}")


if __name__ == "__main__":
    main()
