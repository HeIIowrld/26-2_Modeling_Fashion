#!/usr/bin/env python3
"""Measure CatVTON top-mask morphology against official parsing labels.

The raw mask is an *oracle* union of visible top (1) and outerwear (2). This
isolates closing/dilation geometry; it does not measure parser errors or VTON
image quality. DeepFashion-MultiModal data is non-commercial research only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


VARIANTS = {
    "raw": (0.0, 0.0),
    "close_5": (0.05, 0.0),
    "close_5_dilate_3": (0.05, 0.03),
    "close_1_dilate_1": (0.01, 0.01),
}


def kernel_size(shape: tuple[int, int], ratio: float) -> int:
    """Use the same odd-kernel rule as CatVTON's mask helpers."""
    return max(3, int(min(shape) * ratio) | 1)


def morph_mask(raw: np.ndarray, close_ratio: float, dilate_ratio: float) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError("raw mask must be two-dimensional")
    mask = raw.astype(np.uint8)
    if close_ratio:
        side = kernel_size(mask.shape, close_ratio)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilate_ratio:
        side = kernel_size(mask.shape, dilate_ratio)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        mask = cv2.dilate(mask * 255, kernel, iterations=1)
    return mask > 0


def evaluate(segmentation: np.ndarray, image_name: str) -> list[dict]:
    if segmentation.ndim != 2:
        raise ValueError("official segmentation must be two-dimensional")
    raw = np.isin(segmentation, (1, 2))
    if not raw.any():
        raise ValueError("official segmentation has no top or outerwear pixels")
    outer_pixels = int((segmentation == 2).sum())
    rows = []
    for variant, (close_ratio, dilate_ratio) in VARIANTS.items():
        mask = morph_mask(raw, close_ratio, dilate_ratio)
        added = mask & ~raw
        rows.append({
            "image": image_name,
            "has_outer_1000px": outer_pixels >= 1000,
            "variant": variant,
            "image_pixels": int(raw.size),
            "raw_pixels": int(raw.sum()),
            "result_pixels": int(mask.sum()),
            "added_pixels": int(added.sum()),
            "added_background": int((added & (segmentation == 0)).sum()),
            "added_pants_leggings": int((added & np.isin(segmentation, (5, 6))).sum()),
            "added_skin": int((added & (segmentation == 15)).sum()),
            "added_hair": int((added & (segmentation == 13)).sum()),
            "removed_raw_pixels": int((raw & ~mask).sum()),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no rows to summarize")
    return {
        "images": len(rows),
        "mean_raw_fraction": round(float(np.mean([r["raw_pixels"] / r["image_pixels"] for r in rows])), 6),
        "mean_result_fraction": round(float(np.mean([r["result_pixels"] / r["image_pixels"] for r in rows])), 6),
        "mean_added_fraction": round(float(np.mean([r["added_pixels"] / r["image_pixels"] for r in rows])), 6),
        "mean_added_relative_to_raw": round(float(np.mean([r["added_pixels"] / r["raw_pixels"] for r in rows])), 6),
        "mean_added_background_fraction": round(float(np.mean([r["added_background"] / r["image_pixels"] for r in rows])), 6),
        "mean_added_pants_leggings_fraction": round(float(np.mean([r["added_pants_leggings"] / r["image_pixels"] for r in rows])), 6),
        "mean_added_skin_fraction": round(float(np.mean([r["added_skin"] / r["image_pixels"] for r in rows])), 6),
        "mean_added_hair_fraction": round(float(np.mean([r["added_hair"] / r["image_pixels"] for r in rows])), 6),
        "images_removing_raw_pixels": sum(r["removed_raw_pixels"] > 0 for r in rows),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--segm-dir", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    rows = []
    skipped_no_top_or_outer = 0
    for path in sorted(args.segm_dir.glob("*_segm.png")):
        with Image.open(path) as image:
            segmentation = np.asarray(image, dtype=np.uint8)
        if not np.isin(segmentation, (1, 2)).any():
            skipped_no_top_or_outer += 1
            continue
        rows.extend(evaluate(segmentation, path.name))
    if not rows:
        raise SystemExit("공식 parsing PNG를 찾지 못했습니다.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "raw_mask": "DeepFashion-MultiModal labels 1=top or 2=outer (oracle)",
        "skipped_no_top_or_outer": skipped_no_top_or_outer,
        "variants": {},
    }
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        result["variants"][variant] = {
            "all": summarize(subset),
            "outer": summarize([row for row in subset if row["has_outer_1000px"]]),
            "non_outer": summarize([row for row in subset if not row["has_outer_1000px"]]),
        }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
