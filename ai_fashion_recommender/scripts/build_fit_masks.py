"""라벨 CSV의 상·하의 마스크를 FASHN parser로 일괄 생성하고 CSV를 갱신한다."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clothing_parser import ClothingParser
from fit_vision_dataset import fit_csv_header
from schemas import PoseAnalysis


def resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="핏 학습 이미지의 의류 마스크를 일괄 생성합니다.")
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    annotation = Path(args.annotations_csv).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    mask_root = Path(args.mask_dir).expanduser().resolve()
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    clothing_parser = ClothingParser(use_fashn=True)
    if clothing_parser.backend != "fashn-human-parser":
        raise SystemExit("FASHN parser가 활성화되지 않았습니다.")
    cache = {}
    generated = skipped = failed = 0
    empty_pose = PoseAnalysis(False, 0, "", 0, 0, 0, "", 0)
    for index, row in enumerate(rows, 1):
        image_path = resolve(row["image_path"], dataset_root)
        relative = Path(row["image_path"])
        if relative.is_absolute():
            relative = Path(image_path.name)
        output = (mask_root / row["category"] / relative).with_suffix(".png")
        if output.is_file() and not args.overwrite:
            skipped += 1
        else:
            try:
                if image_path not in cache:
                    cache[image_path] = clothing_parser.parse(image_path, empty_pose)
                parsed = cache[image_path]
                mask = parsed["upper_mask"] if row["category"] == "top" else parsed["lower_mask"]
                if int(np.asarray(mask).sum()) < 20:
                    raise ValueError("의류 마스크가 너무 작습니다")
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(output)
                generated += 1
            except Exception as error:  # 개별 실패는 목록의 나머지를 막지 않는다.
                failed += 1
                print(f"[{index}/{len(rows)}] 실패 {image_path.name}: {error}", flush=True)
                continue
        try:
            row["mask_path"] = output.relative_to(dataset_root).as_posix()
        except ValueError:
            row["mask_path"] = str(output)
        if index % 25 == 0:
            print(f"[{index}/{len(rows)}] 생성 {generated}, 재사용 {skipped}, 실패 {failed}", flush=True)
    temporary = annotation.with_suffix(annotation.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fit_csv_header())
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in fit_csv_header()} for row in rows)
    temporary.replace(annotation)
    print(f"완료: 생성 {generated}, 재사용 {skipped}, 실패 {failed} -> {annotation}")


if __name__ == "__main__":
    main()
