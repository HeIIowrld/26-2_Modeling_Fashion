"""Rasterize manually traced coat cords/trim for an oracle mask experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def rasterize_annotation(annotation: dict) -> np.ndarray:
    size = annotation.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("image_size must be [width, height]")
    width, height = size
    if not all(isinstance(v, int) and 0 < v <= 4096 for v in size):
        raise ValueError("image dimensions must be positive integers <= 4096")
    mask = np.zeros((height, width), dtype=np.uint8)
    polylines = annotation.get("polylines")
    if not isinstance(polylines, list) or not polylines:
        raise ValueError("at least one polyline is required")
    for line in polylines:
        points = np.asarray(line.get("points"))
        brush = line.get("width")
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.issubdtype(points.dtype, np.integer)
                or not isinstance(brush, int) or not 1 <= brush <= 64):
            raise ValueError("each polyline needs integer points and a 1..64 pixel brush")
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            raise ValueError("polyline points must lie inside image_size")
        cv2.polylines(mask, [points.astype(np.int32)], False, 255, brush, cv2.LINE_8)
    proposal = mask > 0
    if proposal.mean() > 0.025:
        raise ValueError("annotated accessory proposal is too broad")
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.annotation.resolve() == args.output.resolve():
        ap.error("--output must differ from --annotation")
    if args.output.suffix.lower() != ".png":
        ap.error("--output must be a lossless PNG mask")
    proposal = rasterize_annotation(json.loads(args.annotation.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(proposal.astype(np.uint8) * 255).save(args.output)
    print(f"saved={args.output} accessory_fraction={proposal.mean():.5f}")


if __name__ == "__main__":
    main()
