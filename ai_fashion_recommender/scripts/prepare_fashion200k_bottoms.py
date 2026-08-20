from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as parquet

import sys

# 런타임 모듈은 src/에 있다. 임포트 전에 경로를 등록한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_attribute_schema import ATTRIBUTE_TASKS


FASHION200K_BOTTOMS_SOURCES = (
    "https://huggingface.co/datasets/Marqo/fashion200k/resolve/"
    "refs%2Fconvert%2Fparquet/default/data/0004.parquet",
    "https://huggingface.co/datasets/Marqo/fashion200k/resolve/"
    "refs%2Fconvert%2Fparquet/default/data/0005.parquet",
)
LOWER_SUBTYPE_TERMS = {
    "카고 팬츠": ("cargo",),
    "트랙팬츠": ("track pant", "tracksuit bottom"),
    "조거·스웨트팬츠": ("jogger", "sweatpant", "sweat pant"),
    "요가 팬츠": ("yoga pant", "yoga legging"),
    "세일러 팬츠": ("sailor pant", "sailor trouser"),
    "청바지": ("jean", "denim pant", "denim trouser"),
    "치노 팬츠": ("chino",),
    "슬랙스": (
        "slack", "tailored trouser", "suit trouser", "formal trouser",
        "dress pant", "smart trouser",
    ),
}
PANT_LEG_SHAPE_TERMS = {
    "부츠컷": ("bootcut", "boot cut"),
    "팔라초": ("palazzo",),
    "페그": ("pegged", "peg leg", "peg-leg", "peg"),
    "테이퍼드": ("tapered",),
    "플레어": ("flare", "flared"),
    "스키니": ("skinny",),
    "스트레이트": ("straight-leg", "straight leg", "straight cut"),
    "와이드": ("wide-leg", "wide leg"),
}
LOWER_DETAIL_TERMS = {
    "5포켓": ("5-pocket", "five pocket", "five-pocket"),
    "플리츠·턱": ("pleat", "pleated", "front tuck"),
    "드로스트링": ("drawstring",),
    "밴딩 허리": ("elastic waist", "elasticated waist"),
    "사이드 스트라이프": ("side stripe", "side-stripe"),
    "밑단 커프": ("cuff", "cuffed hem", "cuff hem", "ankle cuff"),
    "스터럽": ("stirrup",),
}
MAX_PER_LABEL = 200


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}s?(?![a-z])", text) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _product_id(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]


def _preferred_rows(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["category1"] == "pants":
            grouped[_product_id(row["item_ID"])].append(row)
    return {
        product_id: min(
            values,
            key=lambda row: (not row["item_ID"].endswith("_0"), row["item_ID"]),
        )
        for product_id, values in grouped.items()
    }


def _lower_subtype(row: dict) -> str | None:
    """상품명에 명시된 유형만 약지도 라벨로 사용한다."""
    name = (row.get("category3") or "").lower()
    category2 = (row.get("category2") or "").lower()
    for label in ("카고 팬츠", "트랙팬츠", "조거·스웨트팬츠", "요가 팬츠", "세일러 팬츠"):
        if any(_contains(name, term) for term in LOWER_SUBTYPE_TERMS[label]):
            return label
    if category2 == "leggings" or _contains(name, "legging"):
        return "레깅스"
    if category2 == "harem pants" or _contains(name, "harem"):
        return "하렘 팬츠"
    for label in ("청바지", "치노 팬츠", "슬랙스"):
        if any(_contains(name, term) for term in LOWER_SUBTYPE_TERMS[label]):
            return label
    return None


def _pant_leg_shape(row: dict) -> str | None:
    name = (row.get("category3") or "").lower()
    category2 = (row.get("category2") or "").lower()
    for label, terms in PANT_LEG_SHAPE_TERMS.items():
        if any(_contains(name, term) for term in terms):
            return label
    return {
        "skinny pants": "스키니",
        "straight-leg pants": "스트레이트",
        "wide-leg and palazzo pants": "와이드",
    }.get(category2)


def _pant_length(row: dict) -> str | None:
    name = (row.get("category3") or "").lower()
    category2 = (row.get("category2") or "").lower()
    if _contains(name, "capri"):
        return "카프리·7부"
    if category2 == "cropped pants":
        return "크롭·앵클"
    if category2 == "full length pants":
        return "풀렝스"
    return None


def _lower_details(row: dict, subtype: str | None = None) -> list[str]:
    name = (row.get("category3") or "").lower()
    details = [
        label
        for label, terms in LOWER_DETAIL_TERMS.items()
        if any(_contains(name, term) for term in terms)
    ]
    if subtype == "카고 팬츠":
        details.insert(0, "카고 포켓")
    return list(dict.fromkeys(details))


def _labels(row: dict, subtype: str | None = None) -> dict[str, list[str]]:
    name = (row.get("category3") or "").lower()
    subtype = subtype or _lower_subtype(row)
    shape = _pant_leg_shape(row)
    length = _pant_length(row)
    details = _lower_details(row, subtype)
    labels: dict[str, list[str]] = {
        "category": ["청바지" if subtype == "청바지" else "팬츠"],
    }
    if subtype:
        labels["lower_subtype"] = [subtype]
    if shape:
        labels["pant_leg_shape"] = [shape]
        labels["lower_fit"] = [{
            "스키니": "슬림핏", "스트레이트": "스트레이트핏",
            "테이퍼드": "테이퍼드핏", "페그": "테이퍼드핏",
            "부츠컷": "플레어핏", "플레어": "플레어핏",
            "와이드": "와이드핏", "팔라초": "와이드핏",
        }[shape]]
    if length:
        labels["pant_length"] = [length]
        labels["lower_length"] = [
            "롱·긴바지 기장" if length == "풀렝스" else "미디·7부 기장"
        ]
    if details:
        labels["lower_detail"] = details
    if subtype == "청바지":
        labels["material"] = ["데님"]
    return labels


def _selection_keys(labels: dict[str, list[str]]) -> list[str]:
    return [
        f"{task_name}:{label}"
        for task_name in ("lower_subtype", "pant_leg_shape", "pant_length", "lower_detail")
        for label in labels.get(task_name, [])
    ]


def _select(rows: list[dict], seed: int, max_per_label: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _preferred_rows(rows).values():
        for key in _selection_keys(_labels(row)):
            grouped[key].append(row)
    selected: dict[str, dict] = {}
    for values in grouped.values():
        rng.shuffle(values)
        for row in values[:max_per_label]:
            selected[row["item_ID"]] = row
    return sorted(selected.values(), key=lambda row: row["item_ID"])


def _is_validation_product(product_id: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{product_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 5 == 0


def _combine_csv(base_csv: Path, supplement_csv: Path, output_csv: Path) -> None:
    fieldnames = ["image_path", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *ATTRIBUTE_TASKS]
    with base_csv.open(encoding="utf-8-sig", newline="") as handle:
        base_rows = [
            row for row in csv.DictReader(handle)
            if not row["image_path"].replace("\\", "/").startswith("fashion200k_bottoms/")
        ]
    with supplement_csv.open(encoding="utf-8-sig", newline="") as handle:
        supplement_rows = list(csv.DictReader(handle))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        for row in supplement_rows:
            row["image_path"] = (Path("fashion200k_bottoms") / row["image_path"]).as_posix()
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def prepare_bottoms(
    parquet_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    base_csv: str | Path,
    combined_csv: str | Path,
    seed: int = 44,
    max_per_label: int = MAX_PER_LABEL,
) -> dict:
    sources = [Path(path).expanduser().resolve() for path in parquet_paths]
    if not sources:
        raise ValueError("Fashion200K Parquet 파일을 한 개 이상 지정해야 합니다.")
    destination = Path(output_dir).expanduser().resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_index, source in enumerate(sources):
        metadata_rows = parquet.read_table(
            source,
            columns=["category1", "category2", "category3", "item_ID"],
        ).to_pylist()
        for row_index, row in enumerate(metadata_rows):
            row["_source_index"] = source_index
            row["_row_index"] = row_index
        rows.extend(metadata_rows)
    selected = _select(rows, seed, max_per_label)
    # 큰 이미지 열은 샤드 하나씩만 메모리에 올리고 선택된 행만 보관한다.
    selected_by_source: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        selected_by_source[row["_source_index"]].append(row)
    for source_index, source in enumerate(sources):
        images = parquet.read_table(source, columns=["image"])["image"]
        for row in selected_by_source[source_index]:
            row["image"] = images[row["_row_index"]].as_py()
        del images
    validation_products = {
        _product_id(row["item_ID"])
        for row in selected
        if _is_validation_product(_product_id(row["item_ID"]), seed)
    }
    label_counts: dict[str, Counter] = defaultdict(Counter)
    for row in selected:
        for task_name, values in _labels(row).items():
            if task_name in ("lower_subtype", "pant_leg_shape", "pant_length", "lower_detail"):
                label_counts[task_name].update(values)

    fieldnames = ["image_path", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *ATTRIBUTE_TASKS]
    supplement_csv = destination / "fashion_attribute_annotations.csv"
    with supplement_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            image_bytes = row["image"]["bytes"]
            if not image_bytes:
                continue
            extension = Path(row["image"].get("path") or "image.jpg").suffix or ".jpg"
            image_name = f"{row['item_ID']}{extension}"
            (image_dir / image_name).write_bytes(image_bytes)
            output = {
                "image_path": f"images/{image_name}",
                "split": "val" if _product_id(row["item_ID"]) in validation_products else "train",
                "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
            }
            labels = _labels(row)
            for task_name in ATTRIBUTE_TASKS:
                output[task_name] = "|".join(labels.get(task_name, []))
            writer.writerow(output)

    combined = Path(combined_csv).expanduser().resolve()
    _combine_csv(Path(base_csv).expanduser().resolve(), supplement_csv, combined)
    manifest = {
        "dataset": "Fashion200K multi-axis lower-garment supplement",
        "source": list(FASHION200K_BOTTOMS_SOURCES),
        "license": "Apache-2.0",
        "weak_label_note": (
            "Only explicit category2/category3 product metadata was converted. "
            "The labels are weak supervision and require external photo validation."
        ),
        "seed": seed,
        "selected_images": len(selected),
        "train_products": len(selected) - len(validation_products),
        "val_products": len(validation_products),
        "label_counts": {task_name: dict(counts) for task_name, counts in label_counts.items()},
        "supplement_csv": supplement_csv.relative_to(destination).as_posix(),
        "combined_csv": Path(os.path.relpath(combined, destination)).as_posix(),
        "source_sha256": {source.name: _sha256(source) for source in sources},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion200K에서 하의 세부 종류 학습 표본을 추출합니다.")
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-per-label", type=int, default=MAX_PER_LABEL)
    args = parser.parse_args()
    result = prepare_bottoms(
        args.parquet,
        args.output_dir,
        base_csv=args.base_csv,
        combined_csv=args.combined_csv,
        seed=args.seed,
        max_per_label=args.max_per_label,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
