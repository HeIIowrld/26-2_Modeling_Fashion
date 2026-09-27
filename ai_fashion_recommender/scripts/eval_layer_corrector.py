#!/usr/bin/env python3
"""Research-only, item-held-out coat/inner correction of SCHP-LIP predictions.

DeepFashion-MM annotations are used only in the offline experiment. The fitted
model and all image-derived feature caches must stay in a private evaluation
directory; this script does not change the production parser or VTON path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from outerwear_layer_parser import (  # noqa: E402
    LIP_COAT, LIP_UPPER_CLOTHES, OuterwearLayerParser, preprocess_schp,
)


TARGET_NAMES = (
    "WOMEN-Jackets_Coats-id_00005412-03_4_full.jpg",  # puffer
    "WOMEN-Jackets_Coats-id_00007123-02_4_full.jpg",  # fur
    "WOMEN-Jackets_Coats-id_00001582-03_4_full.jpg",  # long coat
    "MEN-Jackets_Vests-id_00005047-01_4_full.jpg",  # denim
)
VETO_PROBABILITY_FLOOR = 0.10


def item_key(name: str) -> str:
    return re.sub(r"-\d+_\d+_[A-Za-z]+$", "", Path(name).stem)


def grouped_folds(records: list[dict], folds: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("At least two folds are required")
    outer_by_item: dict[str, bool] = {}
    for record in records:
        key = item_key(record["image"])
        outer_by_item[key] = outer_by_item.get(key, False) or record["outer_pixels"] >= 1000
    assigned: dict[str, int] = {}
    for has_outer in (False, True):
        keys = sorted(
            (key for key, outer in outer_by_item.items() if outer == has_outer),
            key=lambda key: hashlib.sha256(f"layer-corrector-v1:{key}".encode()).digest(),
        )
        for index, key in enumerate(keys):
            assigned[key] = index % folds
    return assigned


def separate_targets(records: list[dict], target_names: tuple[str, ...] = TARGET_NAMES) -> tuple[list[dict], list[dict]]:
    """Keep all views of each named test item out of model fitting."""
    target_items = {item_key(name) for name in target_names}
    development = [record for record in records if item_key(record["image"]) not in target_items]
    targets = [record for record in records if item_key(record["image"]) in target_items]
    return development, targets


def feature_maps(image: Image.Image, parser: OuterwearLayerParser, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use SCHP scores, color, position and local context at a modest resolution."""
    width, height = image.size
    low_width = side
    low_height = max(16, round(side * height / width))
    rgb = cv2.resize(np.asarray(image.convert("RGB")), (low_width, low_height))
    raw = np.asarray(parser.session.run(
        ["logits"], {"pixel_values": preprocess_schp(image)}
    )[0])
    if raw.shape != (1, 20, 473, 473):
        raise RuntimeError(f"Unexpected SCHP logits shape: {raw.shape}")
    logits = cv2.resize(raw[0].transpose(1, 2, 0), (low_width, low_height))
    labels = logits.argmax(axis=2)
    candidate = (labels == LIP_COAT) | (labels == LIP_UPPER_CLOTHES)
    baseline = labels == LIP_COAT
    coat = np.clip(logits[:, :, LIP_COAT], -40, 40) / 20.0
    upper = np.clip(logits[:, :, LIP_UPPER_CLOTHES], -40, 40) / 20.0
    margin = np.clip(logits[:, :, LIP_COAT] - logits[:, :, LIP_UPPER_CLOTHES], -40, 40) / 20.0
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    local_mean = cv2.blur(rgb_float, (17, 17))
    residual = np.abs(rgb_float - local_mean)
    x = np.broadcast_to(np.linspace(0, 1, low_width, dtype=np.float32), (low_height, low_width))
    y = np.broadcast_to(np.linspace(0, 1, low_height, dtype=np.float32)[:, None], (low_height, low_width))
    coat_density = cv2.blur(baseline.astype(np.float32), (17, 17))
    upper_density = cv2.blur((labels == LIP_UPPER_CLOTHES).astype(np.float32), (17, 17))
    gray = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.clip(cv2.magnitude(grad_x, grad_y), 0, 2)
    features = np.dstack((
        rgb_float, lab, local_mean, residual, x, y,
        margin, coat, upper, coat_density, upper_density, edges,
    )).astype(np.float16)
    return features, candidate, baseline


def sample_training(cache_path: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as data:
        features = data["features"].astype(np.float32)
        candidate = data["candidate"].astype(bool)
        official = data["official"]
        baseline = data["baseline"].astype(bool)
    groups = (
        (candidate & (official == 2), 300),
        (candidate & (official == 1) & baseline, 150),
        (candidate & (official == 1) & ~baseline, 150),
        (candidate & ~np.isin(official, (1, 2)), 60),
    )
    flat = features.reshape(-1, features.shape[-1])
    choices = []
    for mask, maximum in groups:
        indices = np.flatnonzero(mask)
        if len(indices):
            choices.append(rng.choice(indices, size=min(maximum, len(indices)), replace=False))
    if not choices:
        return np.empty((0, flat.shape[1]), np.float32), np.empty(0, np.uint8)
    indices = np.concatenate(choices)
    return flat[indices], (official.ravel()[indices] == 2).astype(np.uint8)


def fit_model(cache_paths: list[Path]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(20260925)
    pieces = [sample_training(path, rng) for path in cache_paths]
    x = np.concatenate([part[0] for part in pieces if len(part[1])])
    y = np.concatenate([part[1] for part in pieces if len(part[1])])
    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both outerwear and non-outerwear")
    model = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=15, max_bins=128, learning_rate=0.07,
        l2_regularization=1.0, class_weight={0: 1.0, 1: 2.0},
        early_stopping=False, random_state=20260925,
    )
    model.fit(x, y)
    return model, {"train_pixels": len(y), "train_outer_pixels": int(y.sum())}


def predict(cache_path: Path, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path) as data:
        features = data["features"].astype(np.float32)
        candidate = data["candidate"].astype(bool)
        baseline = data["baseline"].astype(bool)
    corrected = np.zeros(candidate.shape, dtype=bool)
    vetoed = baseline.copy()
    if candidate.any():
        probability = model.predict_proba(features[candidate])[:, 1]
        corrected[candidate] = probability >= 0.5
        vetoed[candidate] &= probability >= VETO_PROBABILITY_FLOOR
    return baseline, corrected, vetoed


def score_one(official: np.ndarray, low_prediction: np.ndarray, image: str, method: str, split: str) -> dict:
    pred = cv2.resize(low_prediction.astype(np.uint8), (official.shape[1], official.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    outer = official == 2
    inner = official == 1
    intersection = int((pred & outer).sum())
    union = int((pred | outer).sum())
    return {
        "image": image, "item": item_key(image), "method": method, "split": split,
        "official_outer_pixels": int(outer.sum()), "official_inner_pixels": int(inner.sum()),
        "pred_coat_pixels": int(pred.sum()), "outer_true_positive": intersection,
        "outer_union": union, "inner_false_coat": int((pred & inner).sum()),
    }


def summarize(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["official_outer_pixels"] >= 1000]
    negative = [row for row in rows if row["official_outer_pixels"] < 1000]

    def average(subset: list[dict], numerator: str, denominator: str) -> float | None:
        values = [row[numerator] / row[denominator] for row in subset if row[denominator]]
        return round(float(np.mean(values)), 5) if values else None

    return {
        "images": len(rows), "outer_images": len(positive), "non_outer_images": len(negative),
        "outer_recall_macro": average(positive, "outer_true_positive", "official_outer_pixels"),
        "outer_iou_macro": average(positive, "outer_true_positive", "outer_union"),
        "inner_false_coat_macro": average(positive, "inner_false_coat", "official_inner_pixels"),
        "non_outer_top_false_coat_macro": average(negative, "inner_false_coat", "official_inner_pixels"),
        "non_outer_false_positive_at_1000px": sum(row["pred_coat_pixels"] >= 1000 for row in negative),
    }


def save_review(image_path: Path, mask_path: Path, baseline: np.ndarray,
                corrected: np.ndarray, vetoed: np.ndarray, output: Path) -> None:
    with Image.open(image_path) as opened:
        source = np.asarray(opened.convert("RGB"))
    with Image.open(mask_path) as opened:
        official = np.asarray(opened, dtype=np.uint8)
    gt = official == 2
    panels = [source]
    for low in (baseline, corrected, vetoed):
        pred = cv2.resize(low.astype(np.uint8), (source.shape[1], source.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = source.copy()
        overlay[gt & ~pred] = (235, 70, 70)  # missed outerwear
        overlay[pred & ~gt] = (70, 115, 235)  # over-erased area
        overlay[gt & pred] = (75, 205, 100)
        panels.append(overlay)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1)).save(output, quality=91)


def save_unlabeled_review(image_path: Path, predictions: tuple[np.ndarray, ...], output: Path) -> None:
    """Show original beside three coat proposals without implying ground truth."""
    with Image.open(image_path) as opened:
        source = np.asarray(opened.convert("RGB"))
    panels = [source]
    for low, title in zip(predictions, ("SCHP baseline", "Our corrector", "Conservative veto")):
        pred = cv2.resize(low.astype(np.uint8), (source.shape[1], source.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = source.astype(np.float32)
        overlay[pred] = overlay[pred] * 0.45 + np.asarray((230, 65, 60)) * 0.55
        panel = overlay.astype(np.uint8)
        cv2.rectangle(panel, (0, 0), (min(source.shape[1], 260), 34), (255, 255, 255), -1)
        cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (25, 25, 25), 2, cv2.LINE_AA)
        panels.append(panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1)).save(output, quality=91)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--cache-dir", type=Path, required=True)
    cli.add_argument("--review-dir", type=Path)
    cli.add_argument("--unlabeled-review-dir", type=Path)
    cli.add_argument("--side", type=int, default=192)
    cli.add_argument("--folds", type=int, default=5)
    cli.add_argument("--limit", type=int, default=0)
    args = cli.parse_args()
    if args.side < 32:
        cli.error("--side must be at least 32")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_paths = sorted((args.dataset_root / "segm").glob("*_segm.png"))
    if args.limit:
        mask_paths = mask_paths[:args.limit]
    parser = OuterwearLayerParser(args.checkpoint)
    records = []
    for index, mask_path in enumerate(mask_paths, 1):
        stem = mask_path.stem.removesuffix("_segm")
        image_path = args.dataset_root / "images" / f"{stem}.jpg"
        if not image_path.is_file():
            continue
        cache_path = args.cache_dir / f"{stem}_side{args.side}.npz"
        with Image.open(mask_path) as opened:
            official = np.asarray(opened, dtype=np.uint8)
        if not cache_path.is_file():
            with Image.open(image_path) as opened:
                features, candidate, baseline = feature_maps(opened.convert("RGB"), parser, args.side)
            reduced = cv2.resize(official, (features.shape[1], features.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(cache_path, features=features, candidate=candidate,
                                baseline=baseline, official=reduced)
        records.append({"image": image_path.name, "mask": mask_path, "image_path": image_path,
                        "cache": cache_path, "outer_pixels": int((official == 2).sum())})
        if index % 25 == 0:
            print(f"Cached {index}/{len(mask_paths)}", flush=True)
    if not records:
        raise SystemExit("No paired annotations found")
    development, target = separate_targets(records)
    fold_by_item = grouped_folds(development, args.folds)
    rows = []
    train_info = []

    def evaluate(model, subset: list[dict], split: str) -> None:
        for record in subset:
            baseline, corrected, vetoed = predict(record["cache"], model)
            with Image.open(record["mask"]) as opened:
                official = np.asarray(opened, dtype=np.uint8)
            for method, prediction in (
                ("schp_baseline", baseline), ("our_corrector", corrected),
                ("our_conservative_veto", vetoed),
            ):
                rows.append(score_one(official, prediction, record["image"], method, split))
            if args.review_dir and (record in target or record["outer_pixels"] >= 1000):
                save_review(record["image_path"], record["mask"], baseline, corrected, vetoed,
                            args.review_dir / f"{Path(record['image']).stem}.jpg")

    for fold in range(args.folds):
        train = [record["cache"] for record in development if fold_by_item[item_key(record["image"])] != fold]
        held = [record for record in development if fold_by_item[item_key(record["image"])] == fold]
        model, details = fit_model(train)
        train_info.append({"split": f"fold_{fold}", "training_images": len(train), **details})
        evaluate(model, held, f"fold_{fold}")
        print(f"Evaluated fold {fold}: {len(held)} images", flush=True)
    if target:
        model, details = fit_model([record["cache"] for record in development])
        train_info.append({"split": "dedicated_target", "training_images": len(development), **details})
        evaluate(model, target, "dedicated_target")
    unlabeled_reviews = []
    if args.unlabeled_review_dir:
        # Refit even when there are no annotated target images; these previews
        # remain strictly qualitative and have no role in the CV metrics.
        if not target:
            model, details = fit_model([record["cache"] for record in development])
            train_info.append({"split": "qualitative_target", "training_images": len(development), **details})
        annotated_names = {record["image"] for record in records}
        for name in TARGET_NAMES:
            image_path = args.dataset_root / "images" / name
            if name in annotated_names or not image_path.is_file():
                continue
            with Image.open(image_path) as opened:
                features, candidate, baseline = feature_maps(opened.convert("RGB"), parser, args.side)
            cache_path = args.cache_dir / f"{Path(name).stem}_unlabeled_side{args.side}.npz"
            np.savez_compressed(cache_path, features=features, candidate=candidate, baseline=baseline)
            predictions = predict(cache_path, model)
            output = args.unlabeled_review_dir / f"{Path(name).stem}.jpg"
            save_unlabeled_review(image_path, predictions, output)
            unlabeled_reviews.append(name)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset_images": len(records), "development_images": len(development),
        "dedicated_target_images": [record["image"] for record in target],
        "unlabeled_qualitative_reviews": unlabeled_reviews,
        "item_disjoint": True, "pixel_probability_threshold": 0.5,
        "conservative_veto_probability_floor": VETO_PROBABILITY_FLOOR,
        "resolution_side": args.side, "training": train_info,
        "development_heldout": {
            method: summarize([row for row in rows if row["split"] != "dedicated_target" and row["method"] == method])
            for method in ("schp_baseline", "our_corrector", "our_conservative_veto")
        },
        "dedicated_target": {
            method: summarize([row for row in rows if row["split"] == "dedicated_target" and row["method"] == method])
            for method in ("schp_baseline", "our_corrector", "our_conservative_veto")
        },
        "production_approved": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
