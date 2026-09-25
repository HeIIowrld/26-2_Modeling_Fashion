#!/usr/bin/env python3
"""Research-only coat-boundary growth around an existing SCHP mask.

The baseline is never eroded. GrabCut only proposes additional coat pixels in a
bounded halo; this is for fuzzy edges/straps, not a replacement for explicit
outerwear/inner instance parsing. All images and masks stay outside the repo.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CASES = (
    ("denim05047", "MEN-Jackets_Vests-id_00005047-01_4_full"),
    ("longcoat01582", "WOMEN-Jackets_Coats-id_00001582-03_4_full"),
    ("puffer05412", "WOMEN-Jackets_Coats-id_00005412-03_4_full"),
    ("fur07123", "WOMEN-Jackets_Coats-id_00007123-02_4_full"),
)


def odd_size(shape: tuple[int, int], ratio: float) -> int:
    value = max(3, round(min(shape) * ratio))
    return value if value % 2 else value + 1


def expand_coat_boundary(
    image: np.ndarray, baseline: np.ndarray, *, halo_ratio: float,
    max_added_fraction: float = 0.03, iterations: int = 3,
) -> tuple[np.ndarray, dict]:
    """Grow only immediately around a confident coat mask; reject runaway edits."""
    rgb = np.asarray(image)
    coat = np.asarray(baseline).astype(bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != coat.shape:
        raise ValueError("RGB image and binary coat mask must have matching dimensions")
    if not 0 < halo_ratio <= 0.25 or not 0 < max_added_fraction <= 0.1:
        raise ValueError("Halo and added-area limits are out of bounds")
    if coat.sum() < max(100, int(coat.size * 0.001)):
        raise ValueError("The seed coat is too small for boundary refinement")

    core_size = odd_size(coat.shape, 0.012)
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size))
    core = cv2.erode(coat.astype(np.uint8), core_kernel).astype(bool)
    if core.sum() < 100:
        raise ValueError("The coat has no stable interior seed")
    halo_size = odd_size(coat.shape, halo_ratio * 2)
    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_size, halo_size))
    allowed = cv2.dilate(coat.astype(np.uint8), halo_kernel).astype(bool)
    gc_mask = np.full(coat.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[allowed] = cv2.GC_PR_BGD
    gc_mask[coat] = cv2.GC_PR_FGD
    gc_mask[core] = cv2.GC_FGD
    cv2.grabCut(np.ascontiguousarray(rgb), gc_mask, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iterations, cv2.GC_INIT_WITH_MASK)
    proposal = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    added = proposal & allowed & ~coat
    added_fraction = float(added.mean())
    accepted = added_fraction <= max_added_fraction
    refined = coat | added if accepted else coat.copy()
    return refined, {
        "halo_ratio": halo_ratio, "baseline_fraction": round(float(coat.mean()), 6),
        "added_fraction": round(added_fraction, 6), "accepted": accepted,
        "refined_fraction": round(float(refined.mean()), 6),
    }


def score(official: np.ndarray, pred: np.ndarray) -> dict:
    if official.shape != pred.shape:
        raise ValueError("Official mask and prediction must match")
    outer = official == 2
    inner = official == 1
    tp = int((outer & pred).sum())
    union = int((outer | pred).sum())
    return {
        "outer_recall": round(tp / max(1, int(outer.sum())), 5),
        "outer_iou": round(tp / max(1, union), 5),
        "inner_false_coat": round(int((inner & pred).sum()) / max(1, int(inner.sum())), 5),
        "missed_outer_pixels": int((outer & ~pred).sum()),
        "non_outer_pred_pixels": int((~outer & pred).sum()),
    }


def propose_dark_straps(image: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, dict]:
    """Find thin dark coat extensions on a bright, nearly uniform studio backdrop.

    This intentionally abstains for light coats or complex backgrounds. It is
    an accessory proposal, never evidence of a hidden body shape.
    """
    rgb = np.asarray(image)
    coat = np.asarray(baseline).astype(bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != coat.shape:
        raise ValueError("RGB image and binary coat mask must have matching dimensions")
    height, width = coat.shape
    empty = np.zeros_like(coat)
    if not coat.any():
        return empty, {"accepted": False, "reason": "empty_coat", "pixels": 0}
    corner_h, corner_w = max(4, height // 12), max(4, width // 12)
    corners = np.concatenate([
        rgb[:corner_h, :corner_w].reshape(-1, 3),
        rgb[:corner_h, -corner_w:].reshape(-1, 3),
        rgb[-corner_h:, :corner_w].reshape(-1, 3),
        rgb[-corner_h:, -corner_w:].reshape(-1, 3),
    ])
    background = np.median(corners, axis=0)
    background_mad = float(np.median(np.abs(corners.astype(np.float32) - background)))
    if background.min() < 210 or background_mad > 20:
        return empty, {"accepted": False, "reason": "complex_background", "pixels": 0}

    ys, xs = np.where(coat)
    bottom = int(ys.max())
    band = coat & (np.arange(height)[:, None] >= bottom - round(0.25 * height)) & (
        np.arange(height)[:, None] < bottom - round(0.05 * height)
    )
    if int(band.sum()) < 100:
        return empty, {"accepted": False, "reason": "no_lower_coat_seed", "pixels": 0}
    coat_brightness = np.max(rgb[band], axis=1)
    if float(np.median(coat_brightness)) > 105:
        return empty, {"accepted": False, "reason": "not_dark_coat", "pixels": 0}
    threshold = min(100, int(np.percentile(coat_brightness, 75)) + 30)
    lower_x = np.where(band)[1]
    x_min, x_max = int(lower_x.min()), int(lower_x.max())
    span = max(1, x_max - x_min)
    y_grid = np.arange(height)[:, None]
    dark = (rgb.max(axis=2) <= threshold) & ~coat
    dark &= (y_grid >= bottom - round(0.08 * height))
    dark &= (y_grid <= bottom + round(0.35 * height))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    proposals = np.zeros_like(coat)
    selected = 0
    for component in range(1, count):
        x, y, comp_w, comp_h, area = map(int, stats[component])
        center_x = x + comp_w / 2
        on_side = center_x <= x_min + 0.25 * span or center_x >= x_max - 0.25 * span
        if (area < max(150, round(0.0002 * coat.size)) or comp_h < 0.04 * height
                or comp_w > 0.08 * width or comp_h / max(1, comp_w) < 1.8
                or y > bottom + 0.02 * height or not on_side):
            continue
        proposals |= labels == component
        selected += 1
    if proposals.mean() > 0.01:
        return empty, {"accepted": False, "reason": "oversize_proposal", "pixels": int(proposals.sum())}
    return proposals, {
        "accepted": bool(selected), "reason": "selected" if selected else "no_strap_component",
        "pixels": int(proposals.sum()), "components": selected,
        "background_mad": round(background_mad, 3), "dark_threshold": threshold,
    }


def review_image(image: np.ndarray, baseline: np.ndarray, refined: np.ndarray,
                 official: np.ndarray | None) -> Image.Image:
    panels = [image]
    for prediction in (baseline, refined):
        overlay = image.astype(np.float32)
        overlay[prediction] = overlay[prediction] * 0.45 + np.asarray((240, 60, 60)) * 0.55
        if official is not None:
            missed = (official == 2) & ~prediction
            overlay[missed] = overlay[missed] * 0.45 + np.asarray((40, 90, 240)) * 0.55
        panels.append(overlay.astype(np.uint8))
    return Image.fromarray(np.concatenate(panels, axis=1))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--images-dir", type=Path, required=True)
    cli.add_argument("--cache-dir", type=Path, required=True)
    cli.add_argument("--official-dir", type=Path)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    strap_rows = []
    for label, stem in CASES:
        image_path = args.images_dir / f"{stem}.jpg"
        caches = sorted(args.cache_dir.glob(f"{stem}*side192.npz"))
        if not image_path.is_file() or len(caches) != 1:
            raise FileNotFoundError(f"Missing image or unique SCHP cache for {label}")
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        with np.load(caches[0]) as data:
            low = data["baseline"].astype(np.uint8)
        baseline = cv2.resize(low, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        Image.fromarray(baseline.astype(np.uint8) * 255).save(
            args.output_dir / f"{label}_schp_baseline_mask.png"
        )
        official = None
        if args.official_dir:
            mask_path = args.official_dir / f"{stem}_segm.png"
            if mask_path.is_file():
                with Image.open(mask_path) as opened:
                    official = np.asarray(opened, dtype=np.uint8)
        halo_five = None
        for halo in (0.02, 0.05, 0.10, 0.20):
            refined, metrics = expand_coat_boundary(image, baseline, halo_ratio=halo)
            if halo == 0.05:
                halo_five = refined
            row = {"case": label, **metrics}
            if official is not None:
                row.update({f"baseline_{key}": value for key, value in score(official, baseline).items()})
                row.update({f"refined_{key}": value for key, value in score(official, refined).items()})
            rows.append(row)
            name = f"{label}_halo{int(halo * 100):02d}"
            Image.fromarray(refined.astype(np.uint8) * 255).save(args.output_dir / f"{name}_mask.png")
            review_image(image, baseline, refined, official).save(
                args.output_dir / f"{name}_review.jpg", quality=91
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
        straps, diagnostics = propose_dark_straps(image, baseline)
        Image.fromarray(straps.astype(np.uint8) * 255).save(
            args.output_dir / f"{label}_dark_strap_proposal.png"
        )
        for method, prior in (("straps_only", baseline), ("halo05_and_straps", halo_five)):
            combined = prior | straps
            row = {"case": label, "method": method, **diagnostics}
            if official is not None:
                row.update({f"baseline_{key}": value for key, value in score(official, baseline).items()})
                row.update({f"refined_{key}": value for key, value in score(official, combined).items()})
            strap_rows.append(row)
            review_image(image, baseline, combined, official).save(
                args.output_dir / f"{label}_{method}_review.jpg", quality=91
            )
            Image.fromarray(combined.astype(np.uint8) * 255).save(
                args.output_dir / f"{label}_{method}_mask.png"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "accessory_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in strap_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strap_rows)


if __name__ == "__main__":
    main()
