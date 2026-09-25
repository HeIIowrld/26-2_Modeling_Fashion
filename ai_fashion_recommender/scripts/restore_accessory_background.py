#!/usr/bin/env python3
"""Research-only studio-background harmonization around detected coat straps.

This corrects color seams left by source-locked compositing where the external
candidate already removed a narrow dark accessory. It never edits outside the
existing erase mask, protected pixels, or generated non-background pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def harmonize_accessory_background(
    original: np.ndarray, edited: np.ndarray, editable: np.ndarray,
    accessory: np.ndarray,
) -> tuple[np.ndarray, dict]:
    source = np.asarray(original)
    candidate = np.asarray(edited)
    allowed = np.asarray(editable).astype(bool)
    proposed = np.asarray(accessory).astype(bool)
    if (source.ndim != 3 or source.shape[2] != 3 or candidate.shape != source.shape
            or allowed.shape != source.shape[:2] or proposed.shape != allowed.shape):
        raise ValueError("Original, edited image and both masks must have matching dimensions")
    if source.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("Images must be RGB uint8")
    height, width = allowed.shape
    output = candidate.copy()
    if not proposed.any():
        return output, {"applied": False, "reason": "no_accessory", "changed_pixels": 0}
    if proposed.mean() > 0.01:
        return output, {"applied": False, "reason": "oversize_accessory", "changed_pixels": 0}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(proposed.astype(np.uint8), 8)
    components_used = 0
    for component in range(1, count):
        x, y, comp_w, comp_h, area = map(int, stats[component])
        if area < 100:
            continue
        center = x + comp_w / 2
        if abs(center - width / 2) < width * 0.08:
            continue
        left_side = center < width / 2
        y_min = max(0, y - 10)
        y_max = min(height, y + comp_h + 10)
        x_min, x_max = (0, min(width, x + comp_w + 20)) if left_side else (
            max(0, x - 20), width
        )
        changed_this_component = False
        for row in range(y_min, y_max):
            # The near-side backdrop better matches local studio lighting than
            # the extreme image edge. Sample outward from this specific strap.
            samples = source[row, max(0, x - 70):max(0, x - 20)] if left_side else (
                source[row, min(width, x + comp_w + 20):min(width, x + comp_w + 70)]
            )
            if len(samples) < 8:
                continue
            median = np.median(samples, axis=0)
            if median.min() < 220 or np.median(np.max(np.abs(samples.astype(float) - median), axis=1)) > 12:
                continue
            local_source = source[row, x_min:x_max]
            local_edited = candidate[row, x_min:x_max]
            local_allowed = allowed[row, x_min:x_max]
            local_accessory = labels[row, x_min:x_max] == component
            # Only background-like generated pixels qualify. Both the old
            # strap and its white cutout halo receive the *same nearby* studio
            # color; copying exact white source halo pixels leaves an outline.
            background_like = np.min(local_edited, axis=1) >= 215
            source_backdrop = np.min(local_source, axis=1) >= 220
            selected = local_allowed & background_like & (source_backdrop | local_accessory)
            if not selected.any():
                continue
            output[row, x_min:x_max][selected] = np.rint(median).astype(np.uint8)
            changed_this_component = True
        components_used += int(changed_this_component)
    changed = np.any(output != candidate, axis=2)
    if changed.any() and (changed & ~allowed).any():
        raise AssertionError("Accessory cleanup escaped the source-locked edit mask")
    return output, {
        "applied": bool(changed.any()),
        "reason": "restored" if changed.any() else "no_reliable_background",
        "changed_pixels": int(changed.sum()),
        "changed_fraction": round(float(changed.mean()), 6),
        "components_used": components_used,
        "outside_edit_exact_to_input": bool(not changed[~allowed].any()),
    }


def load_binary(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.size != size:
            raise ValueError(f"Mask {path} size differs from source image")
        return np.asarray(opened.convert("L")) >= 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--original", type=Path, required=True)
    cli.add_argument("--edited", type=Path, required=True)
    cli.add_argument("--erase-mask", type=Path, required=True)
    cli.add_argument("--protected-mask", type=Path, required=True)
    cli.add_argument("--accessory-mask", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    inputs = (args.original, args.edited, args.erase_mask,
              args.protected_mask, args.accessory_mask)
    if args.output.resolve() in {path.resolve() for path in inputs}:
        cli.error("Output must differ from all inputs")
    with Image.open(args.original) as opened:
        original = np.asarray(opened.convert("RGB"))
    with Image.open(args.edited) as opened:
        edited = np.asarray(opened.convert("RGB"))
    size = (original.shape[1], original.shape[0])
    erase = load_binary(args.erase_mask, size)
    protected = load_binary(args.protected_mask, size)
    accessory = load_binary(args.accessory_mask, size)
    result, metrics = harmonize_accessory_background(
        original, edited, erase & ~protected, accessory
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)
    record = {
        "inputs": {path.name: sha256(path) for path in inputs},
        "output": str(args.output), "output_sha256": sha256(args.output),
        "metrics": metrics, "production_approved": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
