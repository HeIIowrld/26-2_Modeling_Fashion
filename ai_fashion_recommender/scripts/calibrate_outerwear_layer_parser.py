#!/usr/bin/env python3
"""Test whether one SCHP coat-vs-top logit threshold generalizes across photos.

This is a diagnostic on a small, non-commercial research subset. It does not
fit or install a production parser. The threshold is selected only on four
folds and evaluated on the held-out fifth fold, grouped by item ID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from outerwear_layer_parser import LIP_COAT, LIP_UPPER_CLOTHES, OuterwearLayerParser  # noqa: E402


THRESHOLDS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)


def item_key(image_name: str) -> str:
    stem = Path(image_name).stem
    return re.sub(r"-\d+_\d+_[A-Za-z]+$", "", stem)


def grouped_folds(rows: list[dict], folds: int = 5) -> dict[str, int]:
    """Assign same-item images to one fold, stratified by outer presence."""
    outer_by_item: dict[str, bool] = {}
    for row in rows:
        key = item_key(row["image"])
        outer_by_item[key] = outer_by_item.get(key, False) or row["official_outer_pixels"] >= 1000
    result: dict[str, int] = {}
    for is_outer in (False, True):
        items = [key for key, present in outer_by_item.items() if present == is_outer]
        items.sort(key=lambda key: hashlib.sha256(f"20260925:{key}".encode()).digest())
        for index, key in enumerate(items):
            result[key] = index % folds
    return result


def summarize(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["official_outer_pixels"] >= 1000]
    negatives = [row for row in rows if row["official_outer_pixels"] < 1000]

    def mean_ratio(subset: list[dict], numerator: str, denominator: str) -> float | None:
        values = [row[numerator] / row[denominator] for row in subset if row[denominator]]
        return round(float(np.mean(values)), 5) if values else None

    return {
        "images": len(rows),
        "outer_images": len(positives),
        "non_outer_images": len(negatives),
        "outer_recall_macro": mean_ratio(positives, "outer_true_positive", "official_outer_pixels"),
        "outer_iou_macro": round(float(np.mean([
            row["outer_true_positive"]
            / (row["official_outer_pixels"] + row["pred_coat_pixels"] - row["outer_true_positive"])
            for row in positives if row["official_outer_pixels"] + row["pred_coat_pixels"] - row["outer_true_positive"]
        ])), 5) if positives else None,
        "inner_false_coat_macro": mean_ratio(positives, "inner_false_coat", "official_inner_pixels"),
        "non_outer_top_false_coat_macro": mean_ratio(negatives, "inner_false_coat", "official_inner_pixels"),
        "non_outer_false_positive_at_1000px": sum(row["pred_coat_pixels"] >= 1000 for row in negatives),
    }


def evaluate_thresholds(official: np.ndarray, logits: np.ndarray, image_name: str) -> list[dict]:
    if logits.shape[1:] != official.shape or logits.shape[0] != 20:
        raise ValueError("SCHP logits and official annotation dimensions differ")
    labels = logits.argmax(axis=0)
    candidate = (labels == LIP_COAT) | (labels == LIP_UPPER_CLOTHES)
    margin = logits[LIP_COAT] - logits[LIP_UPPER_CLOTHES]
    outer = official == 2
    inner = official == 1
    rows = []
    for threshold in THRESHOLDS:
        coat = candidate & (margin >= threshold)
        rows.append({
            "image": image_name,
            "item": item_key(image_name),
            "category": image_name.split("-id_")[0],
            "threshold": threshold,
            "official_outer_pixels": int(outer.sum()),
            "official_inner_pixels": int(inner.sum()),
            "pred_coat_pixels": int(coat.sum()),
            "outer_true_positive": int((outer & coat).sum()),
            "inner_false_coat": int((inner & coat).sum()),
        })
    return rows


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--limit", type=int, default=0)
    args = cli.parse_args()
    parser = OuterwearLayerParser(args.checkpoint)
    rows: list[dict] = []
    for mask_path in sorted((args.dataset_root / "segm").glob("*_segm.png")):
        name = mask_path.stem.removesuffix("_segm")
        image_path = args.dataset_root / "images" / f"{name}.jpg"
        if not image_path.is_file():
            continue
        with Image.open(mask_path) as opened:
            official = np.asarray(opened, dtype=np.uint8)
        with Image.open(image_path) as opened:
            logits = parser.predict_logits(opened.convert("RGB"))
        rows.extend(evaluate_thresholds(official, logits, image_path.name))
        print(f"{len(rows) // len(THRESHOLDS)} {image_path.name}", flush=True)
        if args.limit and len(rows) // len(THRESHOLDS) >= args.limit:
            break
    if not rows:
        raise SystemExit("평가할 이미지·공식 마스크 쌍이 없습니다.")
    image_rows = [row for row in rows if row["threshold"] == 0.0]
    fold_by_item = grouped_folds(image_rows)
    for row in rows:
        row["fold"] = fold_by_item[row["item"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    all_thresholds = {
        str(threshold): summarize([row for row in rows if row["threshold"] == threshold])
        for threshold in THRESHOLDS
    }
    heldout = []
    for fold in range(5):
        train = {threshold: summarize([
            row for row in rows if row["fold"] != fold and row["threshold"] == threshold
        ]) for threshold in THRESHOLDS}
        # Illustrative selection rule, declared before inspecting held-out data.
        # IoU captures missed outerwear and over-editing; penalize false coats
        # on photos with no outerwear so a lenient threshold cannot win by
        # simply labeling every upper garment as a coat.
        def selection_score(threshold: float) -> float:
            metrics = train[threshold]
            return (metrics["outer_iou_macro"] or 0) - 0.25 * (
                metrics["non_outer_top_false_coat_macro"] or 0
            )

        selected = max(THRESHOLDS, key=selection_score)
        for threshold, label in ((0.0, "baseline"), (selected, "selected")):
            heldout.append({
                "fold": fold,
                "label": label,
                "threshold": threshold,
                **summarize([
                    row for row in rows if row["fold"] == fold and row["threshold"] == threshold
                ]),
            })
    summary = {
        "dataset_images": len(image_rows),
        "folds": 5,
        "selection_rule": "train_macro_outer_iou - 0.25 * train_macro_non_outer_top_false_coat",
        "thresholds_all_data_diagnostic_only": all_thresholds,
        "heldout_folds": heldout,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
