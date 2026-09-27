"""Build an experimental second-pass mask from the generated neutral garment.

Run this only after saving a first-stage ``*_flux_stage.png`` image.  SCHP's
upper-clothes label is used as a proposal, not as a trusted coat/inner mask;
``resolve_second_stage_mask`` later unions it with the pose target and excludes
protected pixels.  This script does not alter the production VTON path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from outerwear_layer_parser import LIP_UPPER_CLOTHES, OuterwearLayerParser  # noqa: E402


def generated_top_proposal(labels: np.ndarray, *, kernel_size: int = 5) -> np.ndarray:
    """Return a lightly dilated upper-clothes proposal at source resolution."""
    if labels.ndim != 2:
        raise ValueError("SCHP labels must be a 2-D image")
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    top = (labels == LIP_UPPER_CLOTHES).astype(np.uint8)
    if not np.any(top):
        raise ValueError("SCHP did not find a generated upper garment")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    proposal = cv2.dilate(top, kernel, iterations=1).astype(bool)
    if proposal.mean() >= 0.8:
        raise ValueError("Generated upper-garment proposal covers too much of the image")
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-image", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kernel-size", type=int, default=5)
    args = ap.parse_args()
    if args.stage_image.resolve() == args.output.resolve():
        ap.error("--output must differ from --stage-image")
    if args.output.suffix.lower() != ".png":
        ap.error("--output must be a lossless PNG mask")
    with Image.open(args.stage_image) as opened:
        image = opened.convert("RGB")
    labels = OuterwearLayerParser(args.checkpoint).predict(image)
    proposal = generated_top_proposal(labels, kernel_size=args.kernel_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(proposal.astype(np.uint8) * 255).save(args.output)
    print(f"saved={args.output} upper_fraction={proposal.mean():.5f}")


if __name__ == "__main__":
    main()
