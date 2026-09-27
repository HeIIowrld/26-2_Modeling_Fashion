#!/usr/bin/env python3
"""Offline test: keep original pixels outside an existing outerwear erase mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def guarded_composite(
    original: Image.Image,
    generated: Image.Image,
    erase_mask: Image.Image,
    *,
    feather_px: float = 8.0,
) -> tuple[Image.Image, dict]:
    if feather_px <= 0:
        raise ValueError("feather_px must be positive")
    source = np.asarray(original.convert("RGB"), dtype=np.uint8)
    candidate = np.asarray(
        generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    mask = np.asarray(erase_mask.convert("L"), dtype=np.uint8)
    if mask.shape != source.shape[:2]:
        raise ValueError("erase mask must have the original image size")
    edit = mask > 127
    if not edit.any():
        raise ValueError("erase mask is empty")
    inside_distance = cv2.distanceTransform(edit.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.minimum(inside_distance / feather_px, 1.0)[..., None]
    composed = np.rint(
        source.astype(np.float32) * (1.0 - alpha)
        + candidate.astype(np.float32) * alpha
    ).astype(np.uint8)
    outside_unchanged = bool(np.array_equal(composed[~edit], source[~edit]))
    metrics = {
        "edited_fraction": round(float(edit.mean()), 6),
        "outside_mask_preserved": outside_unchanged,
        "inside_changed_fraction": round(float(np.any(composed != source, axis=2)[edit].mean()), 6),
        "feather_px": feather_px,
        "source_size": list(original.size),
        "generated_size": list(generated.size),
    }
    return Image.fromarray(composed), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--erase-mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feather-px", type=float, default=8.0)
    args = parser.parse_args()
    with Image.open(args.original) as original, Image.open(args.generated) as generated, Image.open(args.erase_mask) as mask:
        composite, metrics = guarded_composite(
            original, generated, mask, feather_px=args.feather_px
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    composite.save(args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
