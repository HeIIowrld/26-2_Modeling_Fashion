#!/usr/bin/env python3
"""Research-only removal of generated studio-background halos.

Only parser-confirmed, border-connected background is adjusted. Skin, hair,
garments and low-confidence/dark pixels are left unchanged; this is not a
general-purpose outdoor/background restoration method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refine_background(
    source: np.ndarray, candidate: np.ndarray,
    source_labels: np.ndarray, candidate_labels: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Match altered studio background to the source's per-row color.

    Fail closed when the edge swatches are not bright and nearly neutral.
    Connectedness and foreground distance keep parser holes within the new
    garment from becoming background. Alpha feathering avoids a hard cut.
    """
    src = np.asarray(source)
    out = np.asarray(candidate)
    src_seg = np.asarray(source_labels)
    out_seg = np.asarray(candidate_labels)
    if (src.ndim != 3 or src.shape[2] != 3 or out.shape != src.shape
            or src_seg.shape != src.shape[:2] or out_seg.shape != src_seg.shape):
        raise ValueError("RGB images and 2-D parser maps must have matching sizes")
    height, width = src_seg.shape
    strip = max(8, round(width * 0.12))
    edge = np.concatenate((src[:, :strip], src[:, -strip:]), axis=1).astype(np.float32)
    edge_median = np.median(edge.reshape(-1, 3), axis=0)
    edge_variation = float(np.median(np.max(np.abs(edge - edge_median), axis=2)))
    reliable = bool(edge_median.min() >= 230 and edge_variation <= 20)
    unchanged = out.copy()
    metrics = {
        "studio_reliable": reliable,
        "changed_pixels": 0,
        "mean_background_error_before": None,
        "mean_background_error_after": None,
    }
    if not reliable:
        return unchanged, metrics

    # Left/right row swatches retain horizontal as well as vertical lighting.
    # A single row median made a gray vignette visibly white on one side.
    swatches = np.zeros((2, height, 3), dtype=np.float32)
    usable_rows = np.zeros((2, height), dtype=bool)
    for side, columns in enumerate((np.arange(strip), np.arange(width - strip, width))):
        for row in range(height):
            pixels = src[row, columns]
            safe = (src_seg[row, columns] == 0) & (pixels.min(axis=1) >= 175)
            if safe.sum() >= max(8, strip // 4):
                swatches[side, row] = np.median(pixels[safe], axis=0)
                usable_rows[side, row] = True
    if not usable_rows.all(axis=0).any():
        return unchanged, metrics
    for side in range(2):
        known = np.flatnonzero(usable_rows[side])
        if not len(known):
            return unchanged, metrics
        for channel in range(3):
            swatches[side, :, channel] = np.interp(
                np.arange(height), known, swatches[side, known, channel]
            )
    horizontal = np.clip((np.arange(width) - strip / 2) / max(1, width - strip), 0, 1)
    studio = (swatches[0, :, None, :] * (1 - horizontal[None, :, None])
              + swatches[1, :, None, :] * horizontal[None, :, None])

    background = (out_seg == 0).astype(np.uint8)
    count, components = cv2.connectedComponents(background, connectivity=8)
    border = np.concatenate((components[0], components[-1], components[:, 0], components[:, -1]))
    connected = np.isin(components, np.unique(border[border != 0])) if count > 1 else np.zeros_like(background, dtype=bool)
    foreground_distance = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    source_bg_distance = cv2.distanceTransform((src_seg == 0).astype(np.uint8), cv2.DIST_L2, 5)
    studio_distance = np.max(np.abs(out.astype(np.float32) - studio), axis=2)
    source_protected = np.isin(src_seg, (1, 2, 6, 9, 11, 13, 14, 15, 17))
    # The prior source lock already made many pixels exact. Never change them.
    changed_from_source = np.any(out != src, axis=2)
    eligible = (connected & ~source_protected & changed_from_source
                & (source_bg_distance <= max(30, width * 0.12))
                & (foreground_distance >= 2)
                & (out.min(axis=2) >= 175) & (studio_distance <= 65))
    if not eligible.any():
        return unchanged, metrics
    edge_fade = np.clip((max(30, width * 0.12) - source_bg_distance) / 16, 0, 1)
    alpha = np.clip((foreground_distance - 1) / 9, 0, 1) * edge_fade * eligible
    # Keep a narrow naturally antialiased border; elsewhere remove gray halos.
    corrected = out.astype(np.float32) * (1 - alpha[..., None]) + studio * alpha[..., None]
    result = np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
    altered = np.any(result != out, axis=2)
    metrics.update({
        "changed_pixels": int(altered.sum()),
        "changed_fraction": round(float(altered.mean()), 6),
        "mean_background_error_before": round(float(studio_distance[altered].mean()), 3),
        "mean_background_error_after": round(float(np.max(
            np.abs(result.astype(np.float32) - studio), axis=2
        )[altered].mean()), 3),
    })
    return result, metrics


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--original", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--original-labels", type=Path, required=True)
    cli.add_argument("--candidate-labels", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    inputs = (args.original, args.candidate, args.original_labels, args.candidate_labels)
    if args.output.resolve() in {path.resolve() for path in inputs}:
        cli.error("Output must not overwrite an input")
    arrays = []
    for index, path in enumerate(inputs):
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert("RGB") if index < 2 else image).copy())
    result, metrics = refine_background(*arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)
    record = {
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                   for name, path in zip(("original", "candidate", "original_labels", "candidate_labels"), inputs)},
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "metrics": metrics,
        "production_approved": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
