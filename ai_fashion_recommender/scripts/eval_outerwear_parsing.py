#!/usr/bin/env python3
"""Compare FASHN outerwear masks with official DeepFashion-MultiModal labels.

The official parsing annotation uses 1=top and 2=outer.  It is evaluation
ground truth only: this script does not make those labels available at runtime.
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

from clothing_parser import ClothingParser  # noqa: E402
from outerwear_top_tryon import (  # noqa: E402
    build_official_layer_masks,
    build_outerwear_top_masks,
)
from pose_analyzer import PoseAnalyzer  # noqa: E402


def _fraction(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(denominator.sum())
    return round(float((numerator & denominator).sum() / count), 5) if count else None


def evaluate_one(
    official: np.ndarray,
    fashn: np.ndarray,
    style: np.ndarray,
    pose,
) -> dict[str, float | int | bool | None]:
    """Return coverage and label-confusion metrics for one annotated photo."""
    if official.shape != fashn.shape or style.shape != fashn.shape:
        raise ValueError("Official and FASHN masks must have the same dimensions")
    outer = official == 2
    inner = official == 1
    if not outer.any():
        raise ValueError("No official outerwear pixels")
    fashn_top = np.isin(fashn, (3, 4))
    result: dict[str, float | int | bool | None] = {
        "official_outer_pixels": int(outer.sum()),
        "official_inner_pixels": int(inner.sum()),
        "outer_labeled_fashn_top": _fraction(fashn_top, outer),
        "inner_labeled_fashn_top": _fraction(fashn_top, inner),
        "outer_labeled_fashn_hair": _fraction(fashn == 2, outer),
        "outer_labeled_fashn_background": _fraction(fashn == 0, outer),
    }
    masks = build_outerwear_top_masks(fashn, style, pose)
    oracle = build_official_layer_masks(official, pose)
    result.update({
        "pose_mask_valid": True,
        "outer_erase_recall": _fraction(masks.erase_mask, outer),
        "inner_erase_recall": _fraction(masks.erase_mask, inner),
        "outer_missed_protected_fraction": _fraction(~masks.erase_mask & masks.protected_mask, outer),
        "outer_missed_other_fraction": _fraction(
            ~masks.erase_mask & ~masks.protected_mask, outer
        ),
        "background_erased_fraction": _fraction(masks.erase_mask, official == 0),
        "pants_erased_fraction": _fraction(masks.erase_mask, official == 5),
        "skin_erased_fraction": _fraction(masks.erase_mask, official == 15),
        "erase_fraction": round(float(masks.erase_mask.mean()), 5),
        "oracle_outer_erase_recall": _fraction(oracle.erase_mask, outer),
        "oracle_pants_erased_fraction": _fraction(oracle.erase_mask, official == 5),
        "oracle_skin_erased_fraction": _fraction(oracle.erase_mask, official == 15),
        "oracle_background_erased_fraction": _fraction(oracle.erase_mask, official == 0),
        "oracle_erase_fraction": round(float(oracle.erase_mask.mean()), 5),
    })
    return result


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--dataset-root", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--min-outer-pixels", type=int, default=1000)
    arguments.add_argument("--limit", type=int, default=0)
    options = arguments.parse_args()
    image_dir = options.dataset_root / "images"
    seg_dir = options.dataset_root / "segm"
    if not image_dir.is_dir() or not seg_dir.is_dir():
        raise SystemExit("DeepFashion-MultiModal images/segm directories are required")

    parser = ClothingParser(use_fashn=True)
    pose_analyzer = PoseAnalyzer(model_complexity=1)
    rows: list[dict] = []
    try:
        for mask_path in sorted(seg_dir.glob("*_segm.png")):
            official = np.asarray(Image.open(mask_path), dtype=np.uint8)
            outer_count = int((official == 2).sum())
            if outer_count < options.min_outer_pixels:
                continue
            image_path = image_dir / f"{mask_path.stem.removesuffix('_segm')}.jpg"
            row: dict = {"image": image_path.name, "official_outer_pixels": outer_count}
            try:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                pose = pose_analyzer.analyze(image_path)
                parsed = parser.parse(image_path, pose)
                row.update(evaluate_one(
                    official, np.asarray(parsed["segmentation"]),
                    np.asarray(parsed["upper_style_mask"]), pose,
                ))
                row["error"] = ""
            except Exception as exc:
                row["pose_mask_valid"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"{len(rows)} {image_path.name}: {row.get('outer_erase_recall')}", flush=True)
            if options.limit and len(rows) >= options.limit:
                break
    finally:
        pose_analyzer.close()

    options.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image", "error", "pose_mask_valid", "official_outer_pixels",
               "official_inner_pixels", "outer_labeled_fashn_top", "inner_labeled_fashn_top",
               "outer_labeled_fashn_hair", "outer_labeled_fashn_background",
               "outer_erase_recall", "inner_erase_recall", "outer_missed_protected_fraction",
               "outer_missed_other_fraction", "background_erased_fraction", "erase_fraction"]
    columns.extend(("pants_erased_fraction", "skin_erased_fraction",
                    "oracle_outer_erase_recall", "oracle_pants_erased_fraction",
                    "oracle_skin_erased_fraction", "oracle_background_erased_fraction",
                    "oracle_erase_fraction"))
    with options.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row.get("pose_mask_valid")]
    summary = {"samples": len(rows), "pose_mask_valid": len(valid)}
    for key in ("outer_labeled_fashn_top", "inner_labeled_fashn_top",
                "outer_erase_recall", "outer_missed_protected_fraction",
                "outer_missed_other_fraction", "pants_erased_fraction",
                "skin_erased_fraction", "oracle_pants_erased_fraction",
                "oracle_skin_erased_fraction"):
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        summary[f"mean_{key}"] = round(float(np.mean(values)), 5) if values else None
        summary[f"median_{key}"] = round(float(np.median(values)), 5) if values else None
    summary_path = options.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
