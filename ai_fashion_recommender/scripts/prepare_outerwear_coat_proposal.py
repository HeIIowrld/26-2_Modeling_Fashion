"""Export an optional SCHP coat proposal for hair-boundary A/B experiments.

The consumer edits only pixels that FASHN marked as hair *and* SCHP marked as
coat, close to the existing erase mask.  A model disagreement alone does not
prove that a pixel is coat; do not use this proposal in production.
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

from outerwear_layer_parser import LIP_COAT, OuterwearLayerParser  # noqa: E402


def coat_proposal(labels: np.ndarray, *, kernel_size: int = 1) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("SCHP labels must be a 2-D image")
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    proposal = (labels == LIP_COAT).astype(np.uint8)
    if kernel_size > 1:
        proposal = cv2.dilate(
            proposal,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )
    proposal = proposal.astype(bool)
    if not proposal.any() or proposal.mean() > 0.8:
        raise ValueError("SCHP coat proposal is empty or implausibly broad")
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--person", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kernel-size", type=int, default=1, help="Odd-pixel coat-edge dilation for A/B tests")
    args = ap.parse_args()
    if args.person.resolve() == args.output.resolve():
        ap.error("--output must differ from --person")
    if args.output.suffix.lower() != ".png":
        ap.error("--output must be a lossless PNG mask")
    with Image.open(args.person) as opened:
        image = opened.convert("RGB")
    mask = coat_proposal(
        OuterwearLayerParser(args.checkpoint).predict(image), kernel_size=args.kernel_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output)
    print(f"saved={args.output} coat_fraction={mask.mean():.5f}")


if __name__ == "__main__":
    main()
