#!/usr/bin/env python3
"""Evaluate an isolated LIP coat/top parser on official DeepFashion-MM labels.

The provided model is not trained on this benchmark.  Every annotated image is
evaluated once; this is an external-taxonomy baseline, not train/validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from outerwear_layer_parser import LIP_COAT, LIP_UPPER_CLOTHES, OuterwearLayerParser  # noqa: E402


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(denominator.sum())
    return round(float((numerator & denominator).sum() / count), 5) if count else None


def evaluate_one(official: np.ndarray, predicted: np.ndarray) -> dict:
    if official.shape != predicted.shape:
        raise ValueError("Official and predicted masks must have matching dimensions")
    outer = official == 2
    inner = official == 1
    coat = predicted == LIP_COAT
    upper = predicted == LIP_UPPER_CLOTHES
    overlap = int((outer & coat).sum())
    union = int((outer | coat).sum())
    return {
        "official_outer_pixels": int(outer.sum()),
        "official_inner_pixels": int(inner.sum()),
        "pred_coat_pixels": int(coat.sum()),
        "outer_as_coat_recall": ratio(coat, outer),
        "outer_as_upper_fraction": ratio(upper, outer),
        "outer_coat_iou": round(overlap / union, 5) if union else None,
        "outer_coat_precision": ratio(outer, coat),
        "inner_as_coat_fraction": ratio(coat, inner),
        "inner_as_upper_fraction": ratio(upper, inner),
        "coat_on_background_fraction": ratio(coat, official == 0),
        "coat_on_skin_fraction": ratio(coat, official == 15),
        "coat_on_pants_fraction": ratio(coat, official == 5),
    }


def save_review(image: Image.Image, official: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    source = np.asarray(image.convert("RGB")).copy()
    color = source.copy()
    gt, pred = official == 2, predicted == LIP_COAT
    color[gt & ~pred] = (230, 70, 70)
    color[pred & ~gt] = (60, 110, 230)
    color[gt & pred] = (80, 205, 95)
    both = np.concatenate((source, color), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(both).save(path)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--review-dir", type=Path)
    cli.add_argument("--review-images", nargs="*", default=[])
    cli.add_argument("--limit", type=int, default=0)
    args = cli.parse_args()
    parser = OuterwearLayerParser(args.checkpoint)
    rows = []
    review = set(args.review_images)
    for mask_path in sorted((args.dataset_root / "segm").glob("*_segm.png")):
        name = mask_path.stem.removesuffix("_segm")
        image_path = args.dataset_root / "images" / f"{name}.jpg"
        if not image_path.is_file():
            continue
        with Image.open(mask_path) as opened:
            official = np.asarray(opened, dtype=np.uint8)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        predicted = parser.predict(image)
        row = {"image": image_path.name, **evaluate_one(official, predicted)}
        rows.append(row)
        if args.review_dir and image_path.name in review:
            save_review(image, official, predicted, args.review_dir / f"{name}_schp_review.jpg")
        print(f"{len(rows)} {image_path.name} outer={row['official_outer_pixels']} recall={row['outer_as_coat_recall']}", flush=True)
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        raise SystemExit("평가할 이미지·공식 마스크 쌍이 없습니다.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    positives = [row for row in rows if row["official_outer_pixels"] >= 1000]
    negatives = [row for row in rows if row["official_outer_pixels"] < 1000]
    summary = {
        "samples": len(rows), "outer_samples": len(positives),
        "non_outer_samples": len(negatives),
        "outer_detected_at_1000px": sum(row["pred_coat_pixels"] >= 1000 for row in positives),
        "non_outer_false_positive_at_1000px": sum(row["pred_coat_pixels"] >= 1000 for row in negatives),
    }
    for key in ("outer_as_coat_recall", "outer_as_upper_fraction", "outer_coat_iou",
                "outer_coat_precision", "inner_as_coat_fraction", "inner_as_upper_fraction"):
        values = [float(row[key]) for row in positives if row[key] is not None]
        summary[f"mean_{key}"] = round(float(np.mean(values)), 5) if values else None
        summary[f"median_{key}"] = round(float(np.median(values)), 5) if values else None
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
