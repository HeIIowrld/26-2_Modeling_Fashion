"""7단계 C — 선택 이미지의 주석만 뽑아 축소 JSON을 만들고 기존 변환기로 CSV를 생성한다.

기존 convert_fashionpedia_instances() 를 그대로 재사용해 라벨 매핑 로직이
seed 데이터와 100% 동일하도록 보장한다.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import ijson

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from fashion_attribute_dataset import convert_fashionpedia_instances  # noqa: E402

SELECTION = Path(sys.argv[1])
ANNOTATION = Path(sys.argv[2])
OUTPUT_DIR = Path(sys.argv[3])
VALIDATION_RATIO = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
# shard 목록은 반드시 인자로 받는다. 예전에는 [0, 1]로 하드코딩돼 2차 실행 manifest에
# 잘못된 값이 기록됐다(교정본: manifest_r2_corrected.json).
SHARDS = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
SEED = 45


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    wanted = set(selection["image_ids"])
    if SHARDS is None:
        raise SystemExit(
            "shard 목록을 5번째 인자로 지정하세요. 예: ... <validation_ratio> 2,3,4,5,6"
        )
    reduced_path = OUTPUT_DIR / "instances_subset.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    with ANNOTATION.open("rb") as handle:
        for item in ijson.items(handle, "images.item", use_float=True):
            if int(item["id"]) in wanted:
                images.append({"id": int(item["id"]), "file_name": item["file_name"]})
    categories = []
    with ANNOTATION.open("rb") as handle:
        for item in ijson.items(handle, "categories.item", use_float=True):
            categories.append({"id": int(item["id"]), "name": str(item["name"])})
    attributes = []
    with ANNOTATION.open("rb") as handle:
        for item in ijson.items(handle, "attributes.item", use_float=True):
            attributes.append({
                "id": int(item["id"]),
                "name": str(item["name"]),
                "supercategory": str(item.get("supercategory", "")),
            })
    annotations = []
    with ANNOTATION.open("rb") as handle:
        for item in ijson.items(handle, "annotations.item", use_float=True):
            if int(item["image_id"]) not in wanted:
                continue
            annotations.append({
                "id": int(item["id"]),
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": [float(v) for v in item["bbox"]],
                "attribute_ids": [int(v) for v in item.get("attribute_ids", [])],
            })
    print(f"images {len(images):,}  annotations {len(annotations):,} "
          f"categories {len(categories)}  attributes {len(attributes)}", flush=True)

    reduced_path.write_text(json.dumps({
        "images": images, "categories": categories,
        "attributes": attributes, "annotations": annotations,
    }), encoding="utf-8")
    print(f"reduced json: {reduced_path} ({reduced_path.stat().st_size/1e6:.1f} MB)", flush=True)

    image_ids = sorted(wanted)
    random.Random(SEED).shuffle(image_ids)
    cut = round(len(image_ids) * VALIDATION_RATIO)
    validation = set(image_ids[:cut])
    # 새 val 은 기존 val(1,195)과 별개의 평가셋이다. 이름을 val_train_split 으로 구분한다.
    splits = {i: ("val_train_split" if i in validation else "train") for i in image_ids}

    csv_path = OUTPUT_DIR / "fashion_attribute_annotations.csv"
    result = convert_fashionpedia_instances(
        reduced_path, csv_path, split="train", image_prefix="images", image_splits=splits,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    import csv as csvmod
    from collections import Counter
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csvmod.DictReader(handle))
    print(f"\nCSV rows {len(rows):,}  " + str(Counter(r["split"] for r in rows)))
    manifest = {
        "dataset": "Fashionpedia official TRAIN split, stratified subset for rare labels",
        "annotation_source":
            "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
        "image_parquet_source":
            "https://huggingface.co/datasets/detection-datasets/fashionpedia (refs/convert/parquet, default/train/0000-0001)",
        "license_note": "Fashionpedia annotations/ontology are CC BY 4.0. Image copyrights remain with original sources.",
        "selection": (
            f"greedy set cover targeting {selection.get('target', '?')} samples per label"
            + (f", restricted to image_id >= {selection['min_image_id']}"
               if "min_image_id" in selection else "")
        ),
        "selection_target": selection.get("target"),
        "min_image_id": selection.get("min_image_id"),
        "shards_used": SHARDS,
        "seed": SEED,
        "validation_ratio": VALIDATION_RATIO,
        "images": len(images),
        "conversion": result,
        "csv_rows": len(rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
