from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as parquet

import sys

# 런타임 모듈은 src/에 있다. 임포트 전에 경로를 등록한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_attribute_dataset import convert_fashionpedia_instances


FASHIONPEDIA_ANNOTATION_URL = (
    "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"
)
FASHIONPEDIA_HF_PARQUET_URL = (
    "https://huggingface.co/datasets/detection-datasets/fashionpedia/resolve/"
    "refs%2Fconvert%2Fparquet/default/val/0000.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_seed_dataset(
    parquet_path: str | Path,
    annotation_json: str | Path,
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio는 0과 1 사이여야 합니다.")
    source_parquet = Path(parquet_path).expanduser().resolve()
    source_json = Path(annotation_json).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    annotations = json.loads(source_json.read_text(encoding="utf-8"))
    file_names = {int(item["id"]): item["file_name"] for item in annotations["images"]}
    table = parquet.read_table(source_parquet, columns=["image_id", "image"])
    rows = table.to_pylist()
    parquet_ids = {int(row["image_id"]) for row in rows}
    annotation_ids = set(file_names)
    if parquet_ids != annotation_ids:
        missing_images = sorted(annotation_ids - parquet_ids)[:10]
        missing_annotations = sorted(parquet_ids - annotation_ids)[:10]
        raise ValueError(
            "Parquet 이미지와 공식 주석의 image_id가 다릅니다. "
            f"이미지 누락={missing_images}, 주석 누락={missing_annotations}"
        )

    written = 0
    for row in rows:
        image_id = int(row["image_id"])
        image_bytes = row["image"]["bytes"]
        if not image_bytes:
            raise ValueError(f"image_id={image_id}에 이미지 bytes가 없습니다.")
        output = image_dir / file_names[image_id]
        if not output.is_file() or output.stat().st_size != len(image_bytes):
            output.write_bytes(image_bytes)
        written += 1

    image_ids = sorted(annotation_ids)
    random.Random(seed).shuffle(image_ids)
    validation_count = max(1, round(len(image_ids) * validation_ratio))
    validation_ids = set(image_ids[:validation_count])
    image_splits = {
        image_id: "val" if image_id in validation_ids else "train"
        for image_id in image_ids
    }
    csv_path = destination / "fashion_attribute_annotations.csv"
    conversion = convert_fashionpedia_instances(
        source_json,
        csv_path,
        split="train",
        image_prefix="images",
        image_splits=image_splits,
    )
    manifest = {
        "dataset": "Fashionpedia official validation split used as an initial local seed set",
        "license_note": (
            "Fashionpedia annotations/ontology are CC BY 4.0. Image copyrights remain with their "
            "original sources; review Fashionpedia image terms before redistribution or commercial use."
        ),
        "annotation_source": FASHIONPEDIA_ANNOTATION_URL,
        "image_parquet_source": FASHIONPEDIA_HF_PARQUET_URL,
        "annotation_sha256": _sha256(source_json),
        "parquet_sha256": _sha256(source_parquet),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "images": written,
        "train_images": len(image_ids) - validation_count,
        "val_images": validation_count,
        "conversion": conversion,
        "annotation_csv": csv_path.relative_to(destination).as_posix(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashionpedia seed parquet을 학습 이미지와 CSV로 변환합니다.")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_seed_dataset(
        args.parquet,
        args.annotations,
        args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
