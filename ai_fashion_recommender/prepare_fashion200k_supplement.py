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

from fashion_attribute_schema import ATTRIBUTE_TASKS


FASHION200K_SOURCE = (
    "https://huggingface.co/datasets/Marqo/fashion200k/resolve/"
    "refs%2Fconvert%2Fparquet/default/data/0007.parquet"
)
MATERIAL_TERMS = {
    "cotton": "코튼", "wool": "울", "chiffon": "시폰", "silk": "실크·새틴",
    "satin": "실크·새틴", "mesh": "메시", "lace": "레이스", "suede": "스웨이드",
    "leather": "가죽", "knit": "니트", "denim": "데님", "fleece": "퍼·플리스",
}
PATTERN_TERMS = {
    "stripe": "스트라이프", "striped": "스트라이프", "check": "체크", "plaid": "체크",
    "dot": "도트", "floral": "플로럴", "animal": "애니멀", "camo": "카모",
    "camouflage": "카모", "colorblock": "컬러 블록", "color-block": "컬러 블록",
}


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _product_id(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]


def _preferred_rows(rows: list[dict]) -> dict[str, tuple[int, dict]]:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_product_id(row["item_ID"])].append((index, row))
    return {
        product_id: min(values, key=lambda value: (not value[1]["item_ID"].endswith("_0"), value[1]["item_ID"]))
        for product_id, values in grouped.items()
    }


def _category(row: dict) -> str:
    name = (row["category3"] or "").lower()
    if _contains(name, "polo") and "polo-neck" not in name:
        return "폴로 셔츠"
    return {
        "blouses": "블라우스",
        "shirts": "셔츠",
        "t-shirts": "티셔츠",
    }.get(row["category2"], "탑")


def _labels(row: dict) -> dict[str, list[str]]:
    product_name = (row["category3"] or "").lower()
    labels: dict[str, list[str]] = {"category": [_category(row)]}
    if labels["category"] == ["폴로 셔츠"]:
        labels["collar"] = ["폴로 칼라"]

    if "sleeveless" in product_name or "tank" in product_name:
        labels["sleeve_length"] = ["민소매"]
    elif "short sleeve" in product_name or row["category2"] == "short sleeve tops":
        labels["sleeve_length"] = ["반팔"]
    elif "long sleeve" in product_name or row["category2"] == "long sleeved tops":
        labels["sleeve_length"] = ["긴팔"]

    if "cropped" in product_name or _contains(product_name, "crop"):
        labels["upper_length"] = ["크롭 기장"]
    elif "longline" in product_name or "tunic" in product_name:
        labels["upper_length"] = ["롱 기장"]

    neckline_terms = (
        ("v-neck", "V넥"), ("crewneck", "라운드넥"), ("crew neck", "라운드넥"),
        ("round neck", "라운드넥"), ("square neck", "스퀘어넥"),
        ("boat neck", "보트넥"), ("off-shoulder", "오프숄더"),
        ("halter", "홀터넥"), ("turtleneck", "터틀넥"),
    )
    for term, label in neckline_terms:
        if term in product_name:
            labels["neckline"] = [label]
            break

    fit_terms = (
        ("oversized", "오버핏"), ("loose", "여유핏"), ("relaxed", "여유핏"),
        ("skinny", "슬림핏"), ("slim", "슬림핏"), ("classic-fit", "레귤러핏"),
    )
    for term, label in fit_terms:
        if term in product_name:
            labels["upper_fit"] = [label]
            break

    materials = [label for term, label in MATERIAL_TERMS.items() if _contains(product_name, term)]
    if materials:
        labels["material"] = list(dict.fromkeys(materials))
    patterns = [label for term, label in PATTERN_TERMS.items() if _contains(product_name, term)]
    if patterns:
        labels["pattern"] = list(dict.fromkeys(patterns))

    details = []
    for term, label in (
        ("embroidered", "자수"), ("embroidery", "자수"), ("ruffle", "프릴·러플"),
        ("zip", "지퍼"), ("pocket", "포켓"), ("bow", "리본"), ("sequin", "스팽글"),
        ("lace", "레이스"),
    ):
        if _contains(product_name, term):
            details.append(label)
    if details:
        labels["detail"] = list(dict.fromkeys(details))
    return labels


def _select_products(rows: list[dict], seed: int, max_blouses: int, per_material: int) -> list[tuple[int, dict]]:
    preferred = _preferred_rows(rows)
    rng = random.Random(seed)
    selected: dict[str, tuple[int, dict]] = {}

    polo = [(key, value) for key, value in preferred.items() if _category(value[1]) == "폴로 셔츠"]
    for product_id, value in polo:
        selected[product_id] = value

    blouses = [(key, value) for key, value in preferred.items() if value[1]["category2"] == "blouses"]
    rng.shuffle(blouses)
    for product_id, value in blouses[:max_blouses]:
        selected[product_id] = value

    for term in MATERIAL_TERMS:
        candidates = [
            (key, value)
            for key, value in preferred.items()
            if _contains((value[1]["category3"] or "").lower(), term)
        ]
        rng.shuffle(candidates)
        for product_id, value in candidates[:per_material]:
            selected[product_id] = value
    return sorted(selected.values(), key=lambda value: value[0])


def _combine_csvs(
    base_csv: Path,
    supplement_csv: Path,
    output_csv: Path,
    *,
    base_prefix: str,
    supplement_prefix: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *ATTRIBUTE_TASKS]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for source, prefix in ((base_csv, base_prefix), (supplement_csv, supplement_prefix)):
            with source.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    path = Path(row["image_path"])
                    if not path.is_absolute():
                        row["image_path"] = (Path(prefix) / path).as_posix()
                    writer.writerow({name: row.get(name, "") for name in fieldnames})


def prepare_supplement(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    base_csv: str | Path,
    combined_csv: str | Path,
    seed: int = 43,
    max_blouses: int = 300,
    per_material: int = 50,
) -> dict:
    source = Path(parquet_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image", "category1", "category2", "category3", "text", "item_ID"]
    rows = parquet.read_table(source, columns=columns).to_pylist()
    selected = _select_products(rows, seed, max_blouses, per_material)
    product_ids = [_product_id(row["item_ID"]) for _, row in selected]
    val_products = set(random.Random(seed).sample(product_ids, max(1, round(len(product_ids) * 0.2))))

    fieldnames = ["image_path", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *ATTRIBUTE_TASKS]
    supplement_csv = destination / "fashion_attribute_annotations.csv"
    label_counts: dict[str, Counter] = {name: Counter() for name in ATTRIBUTE_TASKS}
    with supplement_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in selected:
            image_bytes = row["image"]["bytes"]
            if not image_bytes:
                continue
            extension = Path(row["image"].get("path") or "image.jpg").suffix or ".jpg"
            image_name = f"{row['item_ID']}{extension}"
            (image_dir / image_name).write_bytes(image_bytes)
            labels = _labels(row)
            output = {
                "image_path": f"images/{image_name}",
                "split": "val" if _product_id(row["item_ID"]) in val_products else "train",
                "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
            }
            for task_name in ATTRIBUTE_TASKS:
                values = labels.get(task_name, [])
                output[task_name] = "|".join(values)
                label_counts[task_name].update(values)
            writer.writerow(output)

    combined = Path(combined_csv).expanduser().resolve()
    _combine_csvs(
        Path(base_csv).expanduser().resolve(), supplement_csv, combined,
        base_prefix="fashionpedia_seed", supplement_prefix="fashion200k_supplement",
    )
    manifest = {
        "dataset": "Fashion200K strict metadata supplement for sparse Fashionpedia labels",
        "source": FASHION200K_SOURCE,
        "license": "Apache-2.0",
        "weak_label_note": (
            "Only explicit category2/category3 product metadata was converted. These labels are weaker "
            "than Fashionpedia expert annotations and should be reviewed before production use."
        ),
        "seed": seed,
        "selected_images": len(selected),
        "train_products": len(selected) - len(val_products),
        "val_products": len(val_products),
        "label_counts": {name: dict(counts) for name, counts in label_counts.items()},
        "supplement_csv": supplement_csv.relative_to(destination).as_posix(),
        "combined_csv": Path(os.path.relpath(combined, destination)).as_posix(),
        "source_sha256": _sha256(source),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion200K의 희소 라벨 보충 표본을 추출합니다.")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-blouses", type=int, default=300)
    parser.add_argument("--per-material", type=int, default=50)
    args = parser.parse_args()
    result = prepare_supplement(
        args.parquet, args.output_dir, base_csv=args.base_csv, combined_csv=args.combined_csv,
        seed=args.seed, max_blouses=args.max_blouses, per_material=args.per_material,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
